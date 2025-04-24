import itertools

import cv2
import yaml
import numpy as np

# Load config.yaml
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Set image directory
file_path = config["file_path"]
img_dir = file_path + "/images/"

# Crop image based on bbox (optionally with padding)
def crop_fn(image, x, y, w, h, expand=0):
    h_img, w_img = image.shape[:2]
    x = max(0, x - expand)
    y = max(0, y - expand)
    w = min(w + 2 * expand, w_img - x)
    h = min(h + 2 * expand, h_img - y)
    return image[y : y + h, x : x + w]

# Visualize ORB feature matching between two images
def visualize_matches(img1, img2, kp1, kp2, matches, title="Feature Matching"):
    img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(title, img_match)
    cv2.waitKey(500)

# ORB feature matching and similarity score computation
def orb_feature_matching(img1, img2, debug):
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.1, nlevels=10)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        if debug == True:
            print("Not enough keypoints!")
            print("img1 des: ", des1)
            print("img2 des: ", des2)
        return kp1, kp2, None, None, None, 0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
            print(f"Total matches: {total_matches}")
            print(f"Inliers (RANSAC): {inliers}")
            print(f"Normalized match ratio: {match_ratio:.2f}")
            print(f"Inlier ratio: {inlier_ratio:.2f}")
            print(f"Similarity score: {similarity_score:.4f}\n")
        return kp1, kp2, matches, des1, des2, similarity_score
    else:
        if (debug==True):
            print("매칭된 특징점 부족")
        return kp1, kp2, matches, des1, des2, 0

# Compare two images and return the bbox pair with the highest similarity
def compare_two_images(yolo_data, img1_file, img2_file, debug, img_cache, orb, orb_cache, match_cache):
    if debug:
        print(f"Comparing {img1_file} and {img2_file}")

    if img1_file not in img_cache:
        img_cache[img1_file] = cv2.imread(img_dir + img1_file, cv2.IMREAD_GRAYSCALE)
    if img2_file not in img_cache:
        img_cache[img2_file] = cv2.imread(img_dir + img2_file, cv2.IMREAD_GRAYSCALE)

    img1 = img_cache[img1_file]
    img2 = img_cache[img2_file]

    bboxes1 = yolo_data[yolo_data["image_filename"] == img1_file].reset_index(drop=True)
    bboxes2 = yolo_data[yolo_data["image_filename"] == img2_file].reset_index(drop=True)

    best_score = 0
    best_match = None

    for _, bbox1 in bboxes1.iterrows():
        x1, y1, x2, y2 = map(int, [bbox1["x1"], bbox1["y1"], bbox1["x2"], bbox1["y2"]])
        crop1 = crop_fn(img1, x1, y1, x2 - x1, y2 - y1, expand=30)
        bbox1_key = f"{img1_file}_{int(bbox1['x1'])}_{int(bbox1['y1'])}_{int(bbox1['x2'])}_{int(bbox1['y2'])}"

        if bbox1_key not in orb_cache:
            kp1, des1 = orb.detectAndCompute(crop1, None)
            orb_cache[bbox1_key] = (kp1, des1)
        else:
            kp1, des1 = orb_cache[bbox1_key]

        for _, bbox2 in bboxes2.iterrows():
            x1, y1, x2, y2 = map(int, [bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]])
            crop2 = crop_fn(img2, x1, y1, x2 - x1, y2 - y1, expand=30)
            bbox2_key = f"{img2_file}_{int(bbox2['x1'])}_{int(bbox2['y1'])}_{int(bbox2['x2'])}_{int(bbox2['y2'])}"

            if bbox2_key not in orb_cache:
                kp2, des2 = orb.detectAndCompute(crop2, None)
                orb_cache[bbox2_key] = (kp2, des2)
            else:
                kp2, des2 = orb_cache[bbox2_key]

            if des1 is None or des2 is None:
                continue

            match_key = tuple(sorted([bbox1_key, bbox2_key]))
            if match_key in match_cache:
                matches, score = match_cache[match_key]
            else:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
                    score = match_ratio * inlier_ratio
                else:
                    score = 0
                match_cache[match_key] = (matches, score)

            if debug:
                visualize_matches(crop1, crop2, kp1, kp2, matches, f"{img1_file} vs {img2_file}")

            if score > best_score:
                best_score = score
                best_match = (img1_file, img2_file, bbox1, bbox2)

    return best_score, best_match

# Compare a fixed bbox with all bboxes in a target image and return the bbox with the highest similarity
def compare_bbox_with_image(yolo_data, bbox, bbox_img_file, target_img_file, debug, img_cache, orb, orb_cache, match_cache):
    if bbox_img_file not in img_cache:
        img_cache[bbox_img_file] = cv2.imread(img_dir + bbox_img_file, cv2.IMREAD_GRAYSCALE)
    if target_img_file not in img_cache:
        img_cache[target_img_file] = cv2.imread(img_dir + target_img_file, cv2.IMREAD_GRAYSCALE)

    img1 = img_cache[bbox_img_file]
    img2 = img_cache[target_img_file]

    x1, y1, x2, y2 = map(int, [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
    crop1 = crop_fn(img1, x1, y1, x2 - x1, y2 - y1, expand=30)
    bbox1_key = f"{bbox_img_file}_{x1}_{y1}_{x2}_{y2}"

    if bbox1_key not in orb_cache:
        kp1, des1 = orb.detectAndCompute(crop1, None)
        orb_cache[bbox1_key] = (kp1, des1)
    else:
        kp1, des1 = orb_cache[bbox1_key]

    bboxes2 = yolo_data[yolo_data["image_filename"] == target_img_file].reset_index(drop=True)
    best_score = 0
    best_bbox2 = None

    for _, bbox2 in bboxes2.iterrows():
        x1, y1, x2, y2 = map(int, [bbox2["x1"], bbox2["y1"], bbox2["x2"], bbox2["y2"]])
        crop2 = crop_fn(img2, x1, y1, x2 - x1, y2 - y1, expand=30)
        bbox2_key = f"{target_img_file}_{x1}_{y1}_{x2}_{y2}"

        if bbox2_key not in orb_cache:
            kp2, des2 = orb.detectAndCompute(crop2, None)
            orb_cache[bbox2_key] = (kp2, des2)
        else:
            kp2, des2 = orb_cache[bbox2_key]

        if des1 is None or des2 is None:
            continue

        match_key = tuple(sorted([bbox1_key, bbox2_key]))
        if match_key in match_cache:
            matches, score = match_cache[match_key]
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
                score = match_ratio * inlier_ratio
            else:
                score = 0
            match_cache[match_key] = (matches, score)

        if debug:
            visualize_matches(crop1, crop2, kp1, kp2, matches, f"{bbox_img_file} vs {target_img_file}")

        if score > best_score:
            best_score = score
            best_bbox2 = bbox2

    return best_score, best_bbox2


# Extract every image pairs with sufficient time gap and similarity score
def compare_all_images(yolo_data, images, img_cache, orb, orb_cache, match_cache):
    threshold = config["hyperparameters"]["firstmap_thresh"]
    min_time_diff = config["hyperparameters"]["firstmap_min_time_diff"]

    score_list = []

    for img1_file, img2_file in itertools.combinations(images, 2):
        time1 = float(img1_file.split('.')[0])
        time2 = float(img2_file.split('.')[0])
        time_diff = abs(time1 - time2)

        if time_diff < min_time_diff:
            continue

        score, match = compare_two_images(
            yolo_data, img1_file, img2_file, False, img_cache, orb, orb_cache, match_cache
        )

        if match and score >= threshold:
            score_list.append((score, match))

    print(f"\nNumber of image pairs with similarity_score ≥ {threshold}: {len(score_list)}")
    return score_list


# Choose 2 most relevant images compared to the given best pair
def compare_best_with_oldmap(yolo_data, best_pair, oldmap_images, img_cache, orb, orb_cache, match_cache):
    thresh = config["hyperparameters"]["nextmap_thresh"]
    img1, img2, box1, box2 = best_pair

    scores = []
    for old in oldmap_images:
        s1, b1 = compare_bbox_with_image(yolo_data, box1, img1, old, False, img_cache, orb, orb_cache, match_cache)
        s2, b2 = compare_bbox_with_image(yolo_data, box2, img2, old, False, img_cache, orb, orb_cache, match_cache)
        if b1 is not None and b2 is not None:
            avg = (s1 + s2) / 2
            if avg >= thresh:
                scores.append((avg, img1, img2, box1, box2, old, b1, b2))

    return sorted(scores, key=lambda x: x[0], reverse=True)[:2] if scores else []

def get_cached_orb_match(img1_file, crop1, bbox1, img2_file, crop2, bbox2, orb, orb_cache, match_cache):
    b1_key = f"{img1_file}_{bbox1['x1']}_{bbox1['y1']}_{bbox1['x2']}_{bbox1['y2']}"
    b2_key = f"{img2_file}_{bbox2['x1']}_{bbox2['y1']}_{bbox2['x2']}_{bbox2['y2']}"
    if b1_key not in orb_cache:
        orb_cache[b1_key] = orb.detectAndCompute(crop1, None)
    if b2_key not in orb_cache:
        orb_cache[b2_key] = orb.detectAndCompute(crop2, None)
    kp1, des1 = orb_cache[b1_key]
    kp2, des2 = orb_cache[b2_key]
    match_key = tuple(sorted([b1_key, b2_key]))
    if match_key in match_cache:
        matches, _ = match_cache[match_key]
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        match_cache[match_key] = (matches, None)
    return kp1, kp2, matches