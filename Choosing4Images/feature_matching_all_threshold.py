import itertools

import cv2
import pandas as pd
import yaml

from utils_image_selector import (
    load_csv,
    select_images
)

from utils_feature_matching import (
    crop_fn,
    visualize_matches,
    orb_feature_matching,
    compare_two_images,
    compare_bbox_with_image
)

# config.yaml 파일 로드
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# min_time_diff 이상 간격이고 유사도 점수 threshold 이상인 모든 쌍을 찾기
def compare_all_images(yolo_data, images):
    threshold = config["hyperparameters"]["firstmap_thresh"]
    min_time_diff = config["hyperparameters"]["firstmap_min_time_diff"]

    score_list = []

    for img1_file, img2_file in itertools.combinations(images, 2):
        time1 = float(img1_file.split('.')[0])
        time2 = float(img2_file.split('.')[0])
        time_diff = abs(time1 - time2)

        if time_diff < min_time_diff:
            continue  # 시간 차이가 작으면 무시
        score, match = compare_two_images(yolo_data, img1_file, img2_file, False)
        if match and score >= threshold:
            # cv2.imshow("crop1", match[4])
            # cv2.imshow("crop2", match[5])
            # cv2.waitKey(0)
            score_list.append((score, match))


    print(f"\nNumber of image pairs with similarity_score ≥ {threshold}: {len(score_list)}")
    input("Press Enter to continue... ")
    return score_list

# best 이미지 쌍과 가장 가까운 이미지 쌍 찾기
def compare_best_with_oldmap(yolo_data, best_pair, oldmap_images):
    thresh = config["hyperparameters"]["nextmap_thresh"]
    img1, img2, box1, box2 = best_pair

    scores = []
    for old in oldmap_images:
        s1, b1 = compare_bbox_with_image(yolo_data, box1, img1, old, False)
        s2, b2 = compare_bbox_with_image(yolo_data, box2, img2, old, False)
        if b1 is not None and b2 is not None:
            avg = (s1 + s2) / 2
            if avg >= thresh:
                scores.append((avg, img1, img2, box1, box2, old, b1, b2))

    return sorted(scores, key=lambda x: x[0], reverse=True)[:2] if scores else []

# 경로 설정
file_path = config["file_path"]
img_dir = file_path + "/images/"
csv_path = config["filtered_csv_path"]
triangulation_input_path = config["triangulation_input_path"]
timestamp_path = config["timestamp_path"]

# 데이터 불러오기
yolo_data_csv = pd.read_csv(csv_path)
df = load_csv(csv_path)

# 두 맵의 선별된 이미지 가져오기
n = int(input("Enter the index of the desired map: "))
select_newmap_images = select_images(n, csv_path, timestamp_path, False)[1]
select_oldmap_images = select_images(n, csv_path, timestamp_path, False)[0]

# oldmap에서 한 쌍 뽑기
best_pair_final = compare_all_images(yolo_data_csv, select_oldmap_images)

# newmap에서 여러 쌍 뽑기
results = []
for i, (score, match) in enumerate(best_pair_final):
    print(f"\n===== Comparing Firstmap Pair {i+1} with Nextmap... =====")
    result = compare_best_with_oldmap(yolo_data_csv, match, select_newmap_images)
    if result:
        results.append((score, result))

if results:
    # 각 result (top2 쌍)에 대해 두 개의 avg 점수의 평균을 기준으로 최고 쌍 선택
    best_result = max(results, key=lambda top2: (top2[1][0][0] + top2[1][1][0]) / 2)

    firstmap_score, best_pair_data = best_result

    for idx, (
        avg_score, img1_file, img2_file, bbox1, bbox2,
        old_file, old_bbox1, old_bbox2
    ) in enumerate(best_pair_data, 1):


        img1 = cv2.imread(img_dir + img1_file, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_dir + img2_file, cv2.IMREAD_GRAYSCALE)
        old  = cv2.imread(img_dir + old_file,  cv2.IMREAD_GRAYSCALE)

        if idx == 1:
            # 첫 번째 결과에서만 firstmap crop 비교 + 원본
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
        cv2.waitKey(0)
    
    print("\n4 images for triangulation:")
    print(f"Firstmap: {best_result[1][0][1]}, {best_result[1][0][2]}")
    print(f"Nextmap: {best_result[1][0][5]}, {best_result[1][1][5]}")
    bbox3 = best_result[1][0][6]
    bbox4 = best_result[1][1][7]

    # triangulation용 정보 저장
    with open(triangulation_input_path, "w") as f:
        f.write(f"{best_result[1][0][1]} {bbox1['x1']},{bbox1['y1']},{bbox1['x2']},{bbox1['y2']}\n")
        f.write(f"{best_result[1][0][2]} {bbox2['x1']},{bbox2['y1']},{bbox2['x2']},{bbox2['y2']}\n")
        f.write(f"{best_result[1][0][5]} {bbox3['x1']},{bbox3['y1']},{bbox3['x2']},{bbox3['y2']}\n")
        f.write(f"{best_result[1][1][5]} {bbox4['x1']},{bbox4['y1']},{bbox4['x2']},{bbox4['y2']}\n")

else:
    print("전체 비교 결과에서 시각화할 대상이 없어요.")

