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
    compare_two_images
)

# config 파일 로드
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# new map 내에서 가장 유사한 이미지 쌍 찾기
def compare_all_images(yolo_data, images):
    best_score = 0
    best_pair = None

    for img1_file, img2_file in itertools.combinations(images, 2):
        score, match = compare_two_images(yolo_data, img1_file, img2_file, False)
        if score > best_score:
            best_score = score
            best_pair = match

    if best_pair:
        img1_file, img2_file, bbox1, bbox2 = best_pair
        print(f"Best new match: {img1_file} and {img2_file} (score: {best_score:.4f})\n")
        
        img1_best = cv2.imread(img_dir + img1_file, cv2.IMREAD_GRAYSCALE)
        img2_best = cv2.imread(img_dir + img2_file, cv2.IMREAD_GRAYSCALE)

        x1, y1, x2, y2 = map(int, [bbox1["x1"], bbox1["y1"], bbox1["x2"], bbox1["y2"]])
        img1_crop = crop_fn(img1_best, x1, y1, x2 - x1, y2 - y1, expand=30)

        x1, y1, x2, y2 = map(int, [bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]])
        img2_crop = crop_fn(img2_best, x1, y1, x2 - x1, y2 - y1, expand=30)

        kp1, kp2, matches, _ = orb_feature_matching(img1_crop, img2_crop, True)

        visualize_matches(
            img1_crop,
            img2_crop,
            kp1,
            kp2,
            matches,
            f"Best Match Crop: {img1_file} vs {img2_file}"
        )

        cv2.imshow("Top1 Newmap Image1", img1_best)
        cv2.imshow("Top1 Newmap Image2", img2_best)
        cv2.waitKey(0)
    return best_pair


# best 이미지 2장과 oldmap 이미지들 간의 유사도 비교
def compare_best_with_oldmap(yolo_data, best_pair, oldmap_images):
    img1_file, img2_file, _, _ = best_pair
    score_list = []

    print(f"Best new match: {img1_file} and {img2_file}")

    for old_file in oldmap_images:
        score1, match1 = compare_two_images(yolo_data, img1_file, old_file, False)
        score2, match2 = compare_two_images(yolo_data, img2_file, old_file, False)

        if match1 is not None and match2 is not None:
            avg_score = (score1 + score2) / 2
            score_list.append((
                avg_score,
                old_file,
                match1[2], match1[3],  # bbox1, bbox from old for img1
                match2[2], match2[3],  # bbox2, bbox from old for img2
            ))

    # 평균 점수 기준으로 정렬해서 상위 2개 oldmap 선택
    score_list = sorted(score_list, key=lambda x: x[0], reverse=True)[:2]

    result_images = []
    print(f"\nFor oldmap: {img1_file} and {img2_file}")
    for i, (avg_score, old_file, bbox11, bbox21, bbox12, bbox22) in enumerate(score_list):
        print(f"\nTop {i+1} oldmap match: {old_file} (avg score: {avg_score:.4f})")

        # 이미지 로드
        img1 = cv2.imread(img_dir + img1_file, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_dir + img2_file, cv2.IMREAD_GRAYSCALE)
        old = cv2.imread(img_dir + old_file, cv2.IMREAD_GRAYSCALE)

        # crop & 매칭 for img1
        x1, y1, x2, y2 = map(int, [bbox11["x1"], bbox11["y1"], bbox11["x2"], bbox11["y2"]])
        cropimg1 = crop_fn(img1, x1, y1, x2 - x1, y2 - y1, expand=30)
        x1, y1, x2, y2 = map(int, [bbox21["x1"], bbox21["y1"], bbox21["x2"], bbox21["y2"]])
        cropold1 = crop_fn(old, x1, y1, x2 - x1, y2 - y1, expand=30)
        kp1, kp2, matches1, _= orb_feature_matching(cropimg1, cropold1, True)
        visualize_matches(cropimg1, cropold1, kp1, kp2, matches1, f"{img1_file} vs {old_file}")

        # crop & 매칭 for img2
        x1, y1, x2, y2 = map(int, [bbox12["x1"], bbox12["y1"], bbox12["x2"], bbox12["y2"]])
        cropimg2 = crop_fn(img2, x1, y1, x2 - x1, y2 - y1, expand=30)
        x1, y1, x2, y2 = map(int, [bbox22["x1"], bbox22["y1"], bbox22["x2"], bbox22["y2"]])
        cropold2 = crop_fn(old, x1, y1, x2 - x1, y2 - y1, expand=30)
        kp1, kp2, matches2, _ = orb_feature_matching(cropimg2, cropold2, True)
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
best_pair_main= compare_all_images(yolo_data_csv, select_oldmap_images)
best_oldmap_main = compare_best_with_oldmap(yolo_data_csv, best_pair_main, select_newmap_images)

