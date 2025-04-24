import cv2
import pandas as pd
import yaml

from utils_image_selector import (
    load_csv,
    load_tracking_events,
    select_images
)

from utils_feature_matching import (
    crop_fn,
    visualize_matches,
    orb_feature_matching,
    compare_all_images,
    compare_best_with_oldmap
)

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# 1. Load path
file_path = config["file_path"]
img_dir = file_path + "/images/"
csv_path = config["filtered_csv_ocr_path"]
timestamp_saved_path = config["timestamp_saved_path"]
yolo_4images_path = config["ocr_4images_path"]

# 2. Load data
yolo_data_csv = pd.read_csv(csv_path)
df = load_csv(csv_path)

# 2-1. Image cache and match cache
img_cache = {}
match_checked = set()

# 3. Extract each map's selected image lists
events = load_tracking_events(timestamp_saved_path)
n = len(events)

with open(yolo_4images_path, "w") as f:
    f.write(f"number of tracking fail = {n}\n")
    f.write(f"\n")

print (f"number of tracking fail = {n}")

for j in range (n): 
    select_newmap_images = select_images(j, csv_path, timestamp_saved_path, False)[1]
    select_oldmap_images = select_images(j, csv_path, timestamp_saved_path, False)[0]

    # 4. Choose 1 best pair in oldmap
    best_pair_final = compare_all_images(yolo_data_csv, select_oldmap_images, img_cache)

    # 5. Extract every sufficient pairs in newmap
    results = []
    for i, (score, match) in enumerate(best_pair_final):
        print(f"\n===== For index {j}: Comparing Firstmap Pair {i+1} with Nextmap... =====")
        result = compare_best_with_oldmap(yolo_data_csv, match, select_newmap_images)
        if result:
            results.append((score, result))

    # 6. Choose 1 best pair in newmap
    valid_results = [r for r in results if len(r[1]) >= 2]

    if valid_results:
        best_result = max(valid_results, key=lambda top2: (top2[1][0][0] + top2[1][1][0]) / 2)

        firstmap_score, best_pair_data = best_result

        for idx, (
            avg_score, img1_file, img2_file, bbox1, bbox2,
            old_file, old_bbox1, old_bbox2
        ) in enumerate(best_pair_data, 1):


            img1 = cv2.imread(img_dir + img1_file, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img_dir + img2_file, cv2.IMREAD_GRAYSCALE)
            old  = cv2.imread(img_dir + old_file,  cv2.IMREAD_GRAYSCALE)

            if idx == 1:
                x1, y1, x2, y2 = map(int, [bbox1["x1"], bbox1["y1"], bbox1["x2"], bbox1["y2"]])
                crop1 = crop_fn(img1, x1, y1, x2 - x1, y2 - y1, expand=30)
                x1, y1, x2, y2 = map(int, [bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]])
                crop2 = crop_fn(img2, x1, y1, x2 - x1, y2 - y1, expand=30)

                kp1, kp2, matches, _, _, _= orb_feature_matching(crop1, crop2, False)

                print(f"\nFirstmap match: {img1_file}, {img2_file} (score: {firstmap_score:.4f})")
                visualize_matches(crop1, crop2, kp1, kp2, matches, f"Firstmap crop match: {img1_file} vs {img2_file}")

                cv2.imshow(f"Firstmap image {img1_file}", img1)
                cv2.imshow(f"Firstmap image {img2_file}", img2)
            
            print(f"\nTop {idx} match: {old_file} (avg score: {avg_score:.4f})")

            # img1 vs old
            x1, y1, x2, y2 = map(int, [old_bbox1["x1"], old_bbox1["y1"], old_bbox1["x2"], old_bbox1["y2"]])
            crop_old1 = crop_fn(old, x1, y1, x2 - x1, y2 - y1, expand=30)
            x1, y1, x2, y2 = map(int, [bbox1["x1"], bbox1["y1"], bbox1["x2"], bbox1["y2"]])
            crop_img1 = crop_fn(img1, x1, y1, x2 - x1, y2 - y1, expand=30)
            kp1, kp2, matches1, _, _, _= orb_feature_matching(crop_img1, crop_old1, True)
            visualize_matches(crop_img1, crop_old1, kp1, kp2, matches1, f"Firstmap image1 vs Nextmap image{idx} crop match: {img1_file} vs {old_file}")

            # img2 vs old
            x1, y1, x2, y2 = map(int, [old_bbox2["x1"], old_bbox2["y1"], old_bbox2["x2"], old_bbox2["y2"]])
            crop_old2 = crop_fn(old, x1, y1, x2 - x1, y2 - y1, expand=30)
            x1, y1, x2, y2 = map(int, [bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]])
            crop_img2 = crop_fn(img2, x1, y1, x2 - x1, y2 - y1, expand=30)
            kp1, kp2, matches2, _, _, _= orb_feature_matching(crop_img2, crop_old2, True)
            visualize_matches(crop_img2, crop_old2, kp1, kp2, matches2, f"Firstmap image2  vs Nextmap image{idx} crop match: {img2_file} vs {old_file}")

            cv2.imshow(f"Nextmap image{idx}: {old_file}", old)
            cv2.waitKey(500)
        
        print("\n4 images for triangulation:")
        print(f"Firstmap: {best_result[1][0][1]}, {best_result[1][0][2]}")
        print(f"Nextmap: {best_result[1][0][5]}", end="")
        bbox3 = best_result[1][0][6]
        bbox4 = best_result[1][1][7]

        # 7. Save informations of selected 4 images for triangulation
        with open(yolo_4images_path, "a") as f:
            f.write(f"For {j+1}th fail...\n")
            if (((best_result[1][0][0] + best_result[1][1][0]) / 2) > 0.1):
                f.write(f"newmap 1 avg score: {best_result[1][0][0]:.4f}\n")
                f.write(f"newmap 2 avg score: {best_result[1][1][0]:.4f}\n")
                f.write(f"{best_result[1][0][1]} {bbox1['x1']},{bbox1['y1']},{bbox1['x2']},{bbox1['y2']}\n")
                f.write(f"{best_result[1][0][2]} {bbox2['x1']},{bbox2['y1']},{bbox2['x2']},{bbox2['y2']}\n")
                f.write(f"{best_result[1][0][5]} {bbox3['x1']},{bbox3['y1']},{bbox3['x2']},{bbox3['y2']}\n")
                f.write(f"{best_result[1][1][5]} {bbox4['x1']},{bbox4['y1']},{bbox4['x2']},{bbox4['y2']}\n")
                f.write("\n")
            else:
                f.write(f"Avg score of 2 images < 0.1\n")
                f.write("\n")

    else:
        with open(yolo_4images_path, "a") as f:
            f.write(f"For {j+1}th fail...\n")
            f.write(f"Cannot choose sufficient 2 images\n")
            f.write("\n")
        print("No targets available for visualization in the results")