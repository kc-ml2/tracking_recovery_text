import cv2
import yaml
import numpy as np

# config.yaml 파일 로드
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# 경로 설정
file_path = config["file_path"]
img_dir = file_path + "/images/"
csv_path = config["filtered_csv_path"]

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
def orb_feature_matching(img1, img2, debug):
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.1, nlevels=10)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        if debug == True:
            print("특징점이 충분하지 않음")
            #print("img1 des: ", len(des1))
            print("img1 des: ", des1)
            print("img2 des: ", des2)
        return kp1, kp2, None, 0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # knn_matches = bf.knnMatch(des1, des2, k=2)
    # good_matches = []
    # for m, n in knn_matches:
    #     if m.distance < 0.75 * n.distance:
    #         good_matches.append(m)
    # matches = sorted(good_matches, key=lambda x: x.distance)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 50:
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

        if (debug==True):
            print(f"전체 매칭 수: {total_matches}")
            print(f"Inlier 수 (RANSAC): {inliers}")
            print(f"정규화된 매칭 비율: {match_ratio:.2f}")
            print(f"Inlier 비율: {inlier_ratio:.2f}")
            print(f"유사도 점수: {similarity_score:.4f}\n")
        return kp1, kp2, matches, des1, des2, similarity_score
    else:
        if (debug==True):
            print("매칭된 특징점 부족")
        return kp1, kp2, matches, des1, des2, 0

# 이미지 두 개를 비교해서 가장 유사한 bbox 쌍 찾기
def compare_two_images(yolo_data, img1_file, img2_file, debug):
    if debug == True:
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
            kp1, kp2, matches, _, _, score = orb_feature_matching(crop1, crop2, debug)
            if debug==True:
                visualize_matches(crop1, crop2, kp1, kp2, matches, f"two images crop match: {img1_file} vs {img2_file}")
            if score > highest_score:
                highest_score = score
                best_match = (img1_file, img2_file, bbox1, bbox2)
    return highest_score, best_match

# bbox와 이미지를 비교해서 이미지의 가장 유사한 bbox 찾기
def compare_bbox_with_image(yolo_data, bbox, bbox_img_file, target_img_file, debug):
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
        kp1, kp2, matches, _, _, score = orb_feature_matching(crop1, crop2, debug)
        if debug==True:
            visualize_matches(crop1, crop2, kp1, kp2, matches, f"Firstmap crop match: {bbox_img_file} vs {target_img_file}")
        if score > best_score:
            best_score = score
            best_bbox2 = bbox2

    return best_score, best_bbox2