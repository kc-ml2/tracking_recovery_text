import itertools

import cv2
import numpy as np
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

# new map 내에서 가장 유사한 이미지 쌍 k개 찾기
def compare_all_images(yolo_data, images, top_k=3):
    score_list = []

    for img1_file, img2_file in itertools.combinations(images, 2):
        score, match = compare_two_images(yolo_data, img1_file, img2_file, True)
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
        kp1, kp2, matches, _ = orb_feature_matching(crop1, crop2, True)

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
        score1, old_bbox1 = compare_bbox_with_image(yolo_data, bbox1, img1_file, old_file, True)
        score2, old_bbox2 = compare_bbox_with_image(yolo_data, bbox2, img2_file, old_file, True)

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
        kp1, kp2, matches1, _= orb_feature_matching(cropimg1, cropold1, True)
        visualize_matches(cropimg1, cropold1, kp1, kp2, matches1, f"{img1_file} vs {old_file}")

        # crop & 매칭 for img2
        x1, y1, x2, y2 = map(int, [bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]])
        cropimg2 = crop_fn(img2, x1, y1, x2 - x1, y2 - y1, expand=30)
        x1, y1, x2, y2 = map(int, [old_bbox2["x1"], old_bbox2["y1"], old_bbox2["x2"], old_bbox2["y2"]])
        cropold2 = crop_fn(old, x1, y1, x2 - x1, y2 - y1, expand=30)
        kp1, kp2, matches2, _= orb_feature_matching(cropimg2, cropold2, True)
        visualize_matches(cropimg2, cropold2, kp1, kp2, matches2, f"{img2_file} vs {old_file}")

        cv2.imshow(f"Top{i+1} Oldmap Image", old)
        result_images.append(old)

    cv2.waitKey(0)
    return result_images

# 경로 설정
file_path = config["file_path"]
img_dir = file_path + "/images/"
csv_path = config["filtered_csv_path"]

# 데이터 불러오기
yolo_data_csv = pd.read_csv(csv_path)
df = load_csv(csv_path)

n=0
select_newmap_images = select_images(n, True)[1]
select_oldmap_images = select_images(n, True)[0]

# 매칭 수행
best_pair_final = compare_all_images(yolo_data_csv, select_oldmap_images)
for i in range(3): 
    best_oldmap_final = compare_best_with_oldmap(yolo_data_csv, best_pair_final[0], select_newmap_images)
    best_oldmap_final = compare_best_with_oldmap(yolo_data_csv, best_pair_final[1], select_newmap_images)
    best_oldmap_final = compare_best_with_oldmap(yolo_data_csv, best_pair_final[2], select_newmap_images)