import itertools
import time

import cv2
import numpy as np
import pandas as pd
import yaml

from image_selector import (
    load_csv,
    load_newmap_first_image,
    load_oldmap_last_image,
    select_newmap_images,
    select_oldmap_images,
)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def crop_image(image, x, y, w, h, expand=0):
    """지정된 영역을 crop (expand 옵션 추가)"""
    h_img, w_img = image.shape[:2]
    x = max(0, x - expand)
    y = max(0, y - expand)
    w = min(w + 2 * expand, w_img - x)
    h = min(h + 2 * expand, h_img - y)
    return image[y : y + h, x : x + w]


def visualize_matches(img1, img2, kp1, kp2, matches, title="Feature Matching"):
    """ORB 특징 매칭 시각화 (OpenCV 사용)"""
    img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(title, img_match)
    cv2.waitKey(0)


def orb_feature_matching(img1, img2):
    # ORB 검출기 생성
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.1, nlevels=10)

    # 특징점 검출 및 디스크립터 계산
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("특징점이 충분하지 않음")
        print("img1 des: ", len(des1))
        print("img2 des: ", des2)
        return None, None, None, None, 0, 0, 0, 0

    # Brute Force 매칭
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 매칭을 거리순으로 정렬
    matches = sorted(matches, key=lambda x: x.distance)

    # RANSAC을 통한 정제
    if len(matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # **유사도 평가**
        total_matches = len(matches)  # 전체 매칭 수
        inliers = np.sum(matchesMask)  # RANSAC을 통과한 inlier 수
        min_features = min(len(kp1), len(kp2))  # 두 이미지에서 최소 특징점 개수
        match_ratio = total_matches / min_features if min_features > 0 else 0  # 정규화된 매칭 비율
        inlier_ratio = inliers / total_matches if total_matches > 0 else 0  # Inlier 비율
        similarity_score = match_ratio * inlier_ratio  # 최종 유사도 점수

        print(f"전체 매칭 수: {total_matches}")
        print(f"Inlier 수 (RANSAC): {inliers}")
        print(f"정규화된 매칭 비율: {match_ratio:.2f}")
        print(f"Inlier 비율: {inlier_ratio:.2f}")
        print(f"유사도 점수 (Matching Ratio × Inlier Ratio): {similarity_score:.4f}")
        print("\n")

        return kp1, kp2, matches, matchesMask, total_matches, inliers, match_ratio, inlier_ratio, similarity_score
    else:
        print("매칭된 특징점 부족")
        return kp1, kp2, matches, None, len(matches), 0, 0, 0, 0


# 한쌍의 이미지 매칭 함수
def compare_images(yolo_data, img1_filename, img2_filename, crop_image, orb_feature_matching, visualize_matches):
    # .png를 .jpg로 변환 (파일 이름 수정)
    # img1_filename = img1_filename.replace('.png', '.jpg')
    # img2_filename = img2_filename.replace('.png', '.jpg')

    # 이미지를 파일 경로에서 로드
    img1 = cv2.imread(img_dir + img1_filename, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_dir + img2_filename, cv2.IMREAD_GRAYSCALE)

    # 첫 번째 이미지의 bbox 추출
    img1_bboxes = yolo_data[yolo_data["image_filename"] == img1_filename]
    img1_bboxes = img1_bboxes.reset_index(drop=True)

    # 두 번째 이미지의 bbox 추출
    img2_bboxes = yolo_data[yolo_data["image_filename"] == img2_filename]
    img2_bboxes = img2_bboxes.reset_index(drop=True)

    highest_similarity_score = 0
    best_match = None

    # img1과 img2의 bbox 조합을 비교
    for _, bbox1 in img1_bboxes.iterrows():  # img1의 bbox 순회
        for _, bbox2 in img2_bboxes.iterrows():  # img2의 bbox 순회
            # img1에서 bbox1을 기준으로 이미지 크롭
            x1, y1, x2, y2 = int(bbox1["x1"]), int(bbox1["y1"]), int(bbox1["x2"]), int(bbox1["y2"])
            img1_crop = crop_image(img1, x1, y1, x2 - x1, y2 - y1, expand=30)

            # img2에서 bbox2을 기준으로 이미지 크롭
            x1, y1, x2, y2 = int(bbox2["x1"]), int(bbox2["y1"]), int(bbox2["x2"]), int(bbox2["y2"])
            img2_crop = crop_image(img2, x1, y1, x2 - x1, y2 - y1, expand=30)

            # ORB 매칭 수행
            kp1, kp2, matches, matchesMask, total_matches, inliers, match_ratio, inlier_ratio, similarity_score = (
                orb_feature_matching(img1_crop, img2_crop)
            )

            # 가장 높은 similarity_score를 기록
            if similarity_score > highest_similarity_score:
                highest_similarity_score = similarity_score
                best_match = (img1_filename, img2_filename, bbox1, bbox2)

    return highest_similarity_score, best_match


# old map의 모든 이미지 쌍 매칭
def compare_all_images(yolo_data, selected_images, crop_image, orb_feature_matching, visualize_matches):
    most_highest_similarity_score = 0
    best_image_pair = None

    # 각 이미지 쌍을 비교
    for img1_filename, img2_filename in itertools.combinations(selected_images, 2):
        print(f"Comparing {img1_filename} and {img2_filename}")
        highest_similarity_score, best_match = compare_images(
            yolo_data, img1_filename, img2_filename, crop_image, orb_feature_matching, visualize_matches
        )

        # 가장 높은 similarity_score가 나온 이미지 쌍을 기록
        if highest_similarity_score > most_highest_similarity_score:
            most_highest_similarity_score = highest_similarity_score
            best_image_pair = best_match

    # 가장 높은 similarity_score를 가진 이미지 쌍 시각화
    if best_image_pair:
        img1_best_filename, img2_best_filename, bbox1, bbox2 = best_image_pair
        print(
            f"Best match: {img1_best_filename} vs {img2_best_filename} with similarity score: { most_highest_similarity_score:.4f}"
        )

        # 이미지를 파일 경로에서 로드
        img1 = cv2.imread(img_dir + img1_best_filename, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_dir + img2_best_filename, cv2.IMREAD_GRAYSCALE)

        # 크롭된 이미지 생성
        x1, y1, x2, y2 = int(bbox1["x1"]), int(bbox1["y1"]), int(bbox1["x2"]), int(bbox1["y2"])
        img1_crop = crop_image(img1, x1, y1, x2 - x1, y2 - y1, expand=30)

        x1, y1, x2, y2 = int(bbox2["x1"]), int(bbox2["y1"]), int(bbox2["x2"]), int(bbox2["y2"])
        img2_crop = crop_image(img2, x1, y1, x2 - x1, y2 - y1, expand=30)

        # ORB 매칭 수행
        kp1, kp2, matches, matchesMask, total_matches, inliers, match_ratio, inlier_ratio, similarity_score = (
            orb_feature_matching(img1_crop, img2_crop)
        )

        # 매칭 결과 시각화
        visualize_matches(
            img1_crop,
            img2_crop,
            kp1,
            kp2,
            matches,
            f"Best Match: {img1_best_filename} vs {img2_best_filename} - bbox1 vs bbox2",
        )

        # 원본 이미지 시각화
        img1_best = cv2.imread(img_dir + img1_best_filename, cv2.IMREAD_GRAYSCALE)
        img2_best = cv2.imread(img_dir + img2_best_filename, cv2.IMREAD_GRAYSCALE)

        cv2.imshow("Best img1", img1_best)
        cv2.imshow("Best img2", img2_best)
        cv2.waitKey(0)

    return best_image_pair


def compare_best_images_with_oldmap(
    yolo_data, best_image_pair, selected_oldmap_images, crop_image, orb_feature_matching, visualize_matches
):
    """
    best_image_pair에서 얻은 이미지 두 개와 selected_oldmap_images에서 가장 유사한 이미지 두 개를 매칭하는 함수
    """
    img1_best_filename, img2_best_filename, bbox1, bbox2 = best_image_pair
    print(f"Best match images: {img1_best_filename} and {img2_best_filename}")

    highest_similarity_score = 0
    best_match_oldmap = None

    # 두 개의 이미지를 selected_oldmap_images와 비교
    for img_old_filename in selected_oldmap_images:
        print(f"Comparing {img1_best_filename} with {img_old_filename}")
        highest_similarity_score_img1, _ = compare_images(
            yolo_data, img1_best_filename, img_old_filename, crop_image, orb_feature_matching, visualize_matches
        )

        print(f"Comparing {img2_best_filename} with {img_old_filename}")
        highest_similarity_score_img2, _ = compare_images(
            yolo_data, img2_best_filename, img_old_filename, crop_image, orb_feature_matching, visualize_matches
        )

        # 두 이미지에서 가장 높은 similarity score 기록
        if (
            highest_similarity_score_img1 > highest_similarity_score
            or highest_similarity_score_img2 > highest_similarity_score
        ):
            highest_similarity_score = max(highest_similarity_score_img1, highest_similarity_score_img2)
            best_match_oldmap = (img1_best_filename, img2_best_filename, img_old_filename)

    # 가장 높은 매칭 점수를 가진 이미지 쌍 시각화
    if best_match_oldmap:
        img1_best_filename, img2_best_filename, img_old_filename = best_match_oldmap
        print(
            f"Best match with oldmap: {img1_best_filename} and {img2_best_filename} vs {img_old_filename} with similarity score: {highest_similarity_score:.4f}"
        )

        # 이미지를 파일 경로에서 로드
        img1_best = cv2.imread(img_dir + img1_best_filename, cv2.IMREAD_GRAYSCALE)
        img2_best = cv2.imread(img_dir + img2_best_filename, cv2.IMREAD_GRAYSCALE)
        img_old_best = cv2.imread(img_dir + img_old_filename, cv2.IMREAD_GRAYSCALE)

        # 크롭된 이미지 생성
        x1, y1, x2, y2 = int(bbox1["x1"]), int(bbox1["y1"]), int(bbox1["x2"]), int(bbox1["y2"])
        img1_crop = crop_image(img1_best, x1, y1, x2 - x1, y2 - y1, expand=30)

        x1, y1, x2, y2 = int(bbox2["x1"]), int(bbox2["y1"]), int(bbox2["x2"]), int(bbox2["y2"])
        img2_crop = crop_image(img2_best, x1, y1, x2 - x1, y2 - y1, expand=30)

        # ORB 매칭 수행
        # kp1, kp2, matches, matchesMask, total_matches, inliers, match_ratio, inlier_ratio, similarity_score = (
        kp1, kp2, matches, matchesMask, total_matches, inliers, match_ratio, inlier_ratio, similarity_score = (
            orb_feature_matching(img1_crop, img_old_best)
        )

        # 매칭 결과 시각화
        visualize_matches(
            img1_crop,
            img_old_best,
            kp1,
            kp2,
            matches,
            f"Best Match with Oldmap: {img1_best_filename} vs {img_old_filename}",
        )

        # 원본 이미지 시각화
        cv2.imshow("Best img1", img1_best)
        cv2.imshow("Best img2", img2_best)
        cv2.imshow("Best oldmap", img_old_best)
        cv2.waitKey(0)

    return best_match_oldmap


# 이미지, 파일 경로
file_path = config["imgs_paths"]["img1"]
img_dir = file_path + "/images"
csv_path = file_path + "/yolo/yolo_info.csv"

# YOLO bbox 정보 읽기
yolo_data = pd.read_csv(csv_path)

# image_selector.py을 이용해 new map과 old map의 매칭 대상 이미지 리스트 만들기
df = load_csv(csv_path)
first_newmap_image = load_newmap_first_image()
selected_newmap_images = select_newmap_images(df, first_newmap_image)

last_oldmap_image = load_oldmap_last_image()
selected_oldmap_images = select_oldmap_images(df, last_oldmap_image)

# new map의 매칭 대상 이미지들 중 최적의 2개 추출
best_image_pair = compare_all_images(
    yolo_data, selected_newmap_images, crop_image, orb_feature_matching, visualize_matches
)

# old map의 매칭 대상 이미지들 중 new map의 2개 이미지와 가장 유사한 2개 추출
best_match_oldmap = compare_best_images_with_oldmap(
    yolo_data, best_image_pair, selected_oldmap_images, crop_image, orb_feature_matching, visualize_matches
)
