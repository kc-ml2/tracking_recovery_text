import itertools

import cv2
import numpy as np
import pandas as pd
import yaml

from image_selector import (
    load_csv,
    select_timestamps_around_n
)

# config.yaml 파일 로드
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# bbox 기준으로 이미지 crop
def crop_fn(image, x, y, w, h, expand=0):
    h_img, w_img = image.shape[:2]
    x = max(0, x - expand)
    y = max(0, y - expand)
    w = min(w + 2 * expand, w_img - x)
    h = min(h + 2 * expand, h_img - y)
    return image[y : y + h, x : x + w]


# ORB 매칭 결과 시각화
def visualize_matches(img1, img2, kp1, kp2, matches, title="Feature Matching"):
    img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(title, img_match)
    cv2.waitKey(0)


# ORB 매칭 및 유사도 점수 계산
def orb_feature_matching(img1, img2):
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.1, nlevels=10)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        print("특징점이 충분하지 않음")
        #print("img1 des: ", len(des1))
        print("img1 des: ", des1)
        print("img2 des: ", des2)
        return kp1, kp2, None, None, 0, 0, 0, 0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        total_matches = len(matches)
        inliers = np.sum(matches_mask)
        min_features = min(len(kp1), len(kp2))
        match_ratio = total_matches / min_features if min_features > 0 else 0
        inlier_ratio = inliers / total_matches if total_matches > 0 else 0
        similarity_score = match_ratio * inlier_ratio
        print(f"전체 매칭 수: {total_matches}")
        print(f"Inlier 수 (RANSAC): {inliers}")
        print(f"정규화된 매칭 비율: {match_ratio:.2f}")
        print(f"Inlier 비율: {inlier_ratio:.2f}")
        print(f"유사도 점수: {similarity_score:.4f}\n")
        return kp1, kp2, matches, matches_mask, total_matches, inliers, match_ratio, inlier_ratio, similarity_score
    else:
        print("매칭된 특징점 부족")
        return kp1, kp2, matches, None, len(matches), 0, 0, 0, 0


# 이미지 두 개를 비교해서 가장 유사한 bbox 쌍 찾기
def compare_two_images(yolo_data, img1_file, img2_file):
    print(f"Comparing {img1_file} and {img2_file}")
    # 원본 이미지 읽어오기
    img1 = cv2.imread(img_dir + img1_file, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_dir + img2_file, cv2.IMREAD_GRAYSCALE)
    # bbox 쌍 찾기
    bboxes1 = yolo_data[yolo_data["image_filename"] == img1_file].reset_index(drop=True)
    bboxes2 = yolo_data[yolo_data["image_filename"] == img2_file].reset_index(drop=True)
    highest_score = 0
    best_match = None
    for _, bbox1 in bboxes1.iterrows():
        for _, bbox2 in bboxes2.iterrows():
            x1, y1, x2, y2 = map(int, [bbox1["x1"], bbox1["y1"], bbox1["x2"], bbox1["y2"]])
            crop1 = crop_fn(img1, x1, y1, x2 - x1, y2 - y1, expand=30)
            x1, y1, x2, y2 = map(int, [bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]])
            crop2 = crop_fn(img2, x1, y1, x2 - x1, y2 - y1, expand=30)
            _, _, _, _, _, _, _, _, score = orb_feature_matching(crop1, crop2)
            #_, _, _, _, _, _, score, _, _ = orb_feature_matching(crop1, crop2)
            if score > highest_score:
                highest_score = score
                best_match = (img1_file, img2_file, bbox1, bbox2)
    return highest_score, best_match

# 이미지에서 bbox와 가장 유사한 bbox 찾기
def compare_bbox_with_image(yolo_data, bbox, bbox_img_file, target_img_file):
    img1 = cv2.imread(img_dir + bbox_img_file, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_dir + target_img_file, cv2.IMREAD_GRAYSCALE)

    # newmap에서 고정된 bbox crop
    x1, y1, x2, y2 = map(int, [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
    crop1 = crop_fn(img1, x1, y1, x2 - x1, y2 - y1, expand=30)

    bboxes2 = yolo_data[yolo_data["image_filename"] == target_img_file].reset_index(drop=True)
    best_score = 0
    best_bbox2 = None

    for _, bbox2 in bboxes2.iterrows():
        x1, y1, x2, y2 = map(int, [bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]])
        crop2 = crop_fn(img2, x1, y1, x2 - x1, y2 - y1, expand=30)
        _, _, _, _, _, _, _, _, score = orb_feature_matching(crop1, crop2)

        if score > best_score:
            best_score = score
            best_bbox2 = bbox2

    return best_score, best_bbox2


# new map 내에서 가장 유사한 이미지 쌍 k개 찾기
def compare_all_images(yolo_data, images, top_k=3):
    score_list = []

    for img1_file, img2_file in itertools.combinations(images, 2):
        score, match = compare_two_images(yolo_data, img1_file, img2_file)
        if match:
            score_list.append((score, match))

    top_matches = sorted(score_list, key=lambda x: x[0], reverse=True)[:top_k]

    for i, (score, match) in enumerate(top_matches):
        img1_file, img2_file, bbox1, bbox2 = match
        print(f"Top {i+1} match: {img1_file} and {img2_file} (score: {score:.4f})")

        # 이미지 로드
        img1 = cv2.imread(img_dir + img1_file, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_dir + img2_file, cv2.IMREAD_GRAYSCALE)

        # crop bbox
        x1, y1, x2, y2 = map(int, [bbox1["x1"], bbox1["y1"], bbox1["x2"], bbox1["y2"]])
        crop1 = crop_fn(img1, x1, y1, x2 - x1, y2 - y1, expand=30)

        x1, y1, x2, y2 = map(int, [bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]])
        crop2 = crop_fn(img2, x1, y1, x2 - x1, y2 - y1, expand=30)

        # ORB 매칭
        kp1, kp2, matches, _, _, _, _, _, _ = orb_feature_matching(crop1, crop2)

        # 시각화
        visualize_matches(crop1, crop2, kp1, kp2, matches, f"Top{i+1} Match Crop: {img1_file} vs {img2_file}")
        cv2.imshow(f"Top{i+1} image1", img1)
        cv2.imshow(f"Top{i+1} image2", img2)
        cv2.waitKey(0)

    return [match for _, match in top_matches]


# old map 내에서 new map의 best 이미지 쌍과 가장 가까운 이미지 2개 찾기
def compare_best_with_oldmap(yolo_data, best_pair, oldmap_images):
    img1_file, img2_file, bbox1, bbox2 = best_pair
    score_list = []

    print(f"Best new match: {img1_file} and {img2_file}")

    for old_file in oldmap_images:
        score1, old_bbox1 = compare_bbox_with_image(yolo_data, bbox1, img1_file, old_file)
        score2, old_bbox2 = compare_bbox_with_image(yolo_data, bbox2, img2_file, old_file)

        if old_bbox1 is not None and old_bbox2 is not None:
            avg_score = (score1 + score2) / 2
            score_list.append((avg_score, old_file, old_bbox1, old_bbox2))

    # 평균 점수 기준으로 정렬해서 상위 2개 oldmap 선택
    score_list = sorted(score_list, key=lambda x: x[0], reverse=True)[:2]

    result_images = []
    for i, (avg_score, old_file, old_bbox1, old_bbox2) in enumerate(score_list):
        print(f"\nTop {i+1} oldmap match: {old_file} (avg score: {avg_score:.4f})")

        # 이미지 로드
        img1 = cv2.imread(img_dir + img1_file, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_dir + img2_file, cv2.IMREAD_GRAYSCALE)
        old = cv2.imread(img_dir + old_file, cv2.IMREAD_GRAYSCALE)

        # crop & 매칭 for img1
        x1, y1, x2, y2 = map(int, [bbox1["x1"], bbox1["y1"], bbox1["x2"], bbox1["y2"]])
        cropimg1 = crop_fn(img1, x1, y1, x2 - x1, y2 - y1, expand=30)
        x1, y1, x2, y2 = map(int, [old_bbox1["x1"], old_bbox1["y1"], old_bbox1["x2"], old_bbox1["y2"]])
        cropold1 = crop_fn(old, x1, y1, x2 - x1, y2 - y1, expand=30)
        kp1, kp2, matches1, _, _, _, _, _, _ = orb_feature_matching(cropimg1, cropold1)
        visualize_matches(cropimg1, cropold1, kp1, kp2, matches1, f"{img1_file} vs {old_file}")

        # crop & 매칭 for img2
        x1, y1, x2, y2 = map(int, [bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]])
        cropimg2 = crop_fn(img2, x1, y1, x2 - x1, y2 - y1, expand=30)
        x1, y1, x2, y2 = map(int, [old_bbox2["x1"], old_bbox2["y1"], old_bbox2["x2"], old_bbox2["y2"]])
        cropold2 = crop_fn(old, x1, y1, x2 - x1, y2 - y1, expand=30)
        kp1, kp2, matches2, _, _, _, _, _, _ = orb_feature_matching(cropimg2, cropold2)
        visualize_matches(cropimg2, cropold2, kp1, kp2, matches2, f"{img2_file} vs {old_file}")

        cv2.imshow(f"Top{i+1} Oldmap Image", old)
        result_images.append(old)

    cv2.waitKey(0)
    return result_images

# 경로 설정
file_path = config["file_path"]["file5"]
img_dir = file_path + "/images/"
csv_path = config["filtered_csv_path"]
#csv_path = config["file_paths"]["file2"] + "/yolo/yolo_info.csv"
# 데이터 불러오기
yolo_data_csv = pd.read_csv(csv_path)
df = load_csv(csv_path)

n=0
select_newmap_images = select_timestamps_around_n(n)[1]
select_oldmap_images = select_timestamps_around_n(n)[0]

# 매칭 수행
best_pair_final = compare_all_images(yolo_data_csv, select_oldmap_images)
best_oldmap_final = compare_best_with_oldmap(yolo_data_csv, best_pair_final[0], select_newmap_images)
best_oldmap_final = compare_best_with_oldmap(yolo_data_csv, best_pair_final[1], select_newmap_images)
best_oldmap_final = compare_best_with_oldmap(yolo_data_csv, best_pair_final[2], select_newmap_images)