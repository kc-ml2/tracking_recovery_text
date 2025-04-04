import cv2
import numpy as np
import yaml

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

# 경로 설정
file_path = config["file_path"]
img_dir = file_path + "/images/"

# 카메라 내부 파라미터
K = np.array([[641.136535, 0, 656.977416],
              [0, 640.351623, 365.402496],
              [0, 0, 1]], dtype=np.float32)

# triangulation_input.txt 파싱
def load_input(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    entries = []
    for line in lines:
        if line.strip() == "":
            continue
        fname, coords = line.strip().split(" ")
        x1, y1, x2, y2 = map(float, coords.split(","))
        entries.append((fname, [int(x1), int(y1), int(x2), int(y2)]))
    return entries

# triangulation 함수
def triangulate(img1, img2, bbox1, bbox2):
    crop1 = crop_fn(img1, bbox1[0], bbox1[1], bbox1[2] - bbox1[0], bbox1[3] - bbox1[1], expand=30)
    crop2 = crop_fn(img2, bbox2[0], bbox2[1], bbox2[2] - bbox2[0], bbox2[3] - bbox2[1], expand=30)

    kp1, kp2, matches, _ = orb_feature_matching(crop1, crop2, False)
    visualize_matches(crop1, crop2, kp1, kp2, matches, f"Two images crop matches!")

    if len(matches) < 8:
        raise ValueError("Not enough matches for triangulation")

    # crop 영역 좌표를 원래 이미지 기준으로 보정
    expand = 30 
    pts1_all = np.float32([kp1[m.queryIdx].pt for m in matches]) + np.array([bbox1[0] - expand, bbox1[1] - expand])
    pts2_all = np.float32([kp2[m.trainIdx].pt for m in matches]) + np.array([bbox2[0] - expand, bbox2[1] - expand])

    # RANSAC 필터링
    E, mask = cv2.findEssentialMat(pts1_all, pts2_all, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    _, R, t, pose_mask = cv2.recoverPose(E, pts1_all, pts2_all, K, mask=mask)

    # Inlier만 필터링
    inliers1 = pts1_all[pose_mask.ravel() == 1]
    inliers2 = pts2_all[pose_mask.ravel() == 1]
    inlier_matches = [matches[i] for i in range(len(matches)) if pose_mask.ravel()[i] == 1]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in inlier_matches]) + np.array([bbox1[0], bbox1[1] - expand])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in inlier_matches]) + np.array([bbox2[0], bbox2[1] - expand])

    if len(inliers1) < 8 or len(inliers2) < 8:
        raise ValueError("Not enough inliers after RANSAC")
    
    # Triangulate with inliers
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    pts1_norm = cv2.undistortPoints(inliers1.reshape(-1,1,2), K, None)
    pts2_norm = cv2.undistortPoints(inliers2.reshape(-1,1,2), K, None)

    pts_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T

    used_kp1 = [kp1[m.queryIdx] for m in inlier_matches]
    used_kp2 = [kp2[m.trainIdx] for m in inlier_matches]

    return pts_3d, pts1, pts2, used_kp1, used_kp2, R, t


# estimate_pose 함수
def estimate_pose(img, bbox, pts3d, used_kp1, ref_img):
    crop = crop_fn(img, bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], expand=30)
    kp2, des2 = cv2.ORB_create(1000).detectAndCompute(crop, None)
    if des2 is None or len(kp2) < 10:
        raise ValueError("Not enough keypoints detected in new image")

    # used_kp1 기준으로 descriptor를 다시 뽑음 (ref_img는 전체 이미지 기준)
    orb = cv2.ORB_create(1000)
    des1 = orb.compute(ref_img, used_kp1)[1]

    if des1 is None or len(des1) != len(used_kp1):
        raise ValueError("Descriptor extraction failed for used_kp1")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 6:
        raise ValueError("Not enough matches between reference and target")

    print(f"[DEBUG] Total matches: {len(matches)}")

    pts2d = []
    object_pts = []
    for m in matches:
        pt2 = kp2[m.trainIdx].pt
        pt2 = np.array(pt2) + np.array([bbox[0], bbox[1]])  # crop 보정
        pts2d.append(pt2)

        pt3d = pts3d[m.queryIdx]  # used_kp1 기준으로 정확히 대응
        object_pts.append(pt3d)

    pts2d = np.float32(pts2d)
    object_pts = np.float32(object_pts)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_pts, pts2d, K, None,
        reprojectionError=12.0,
        iterationsCount=100,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    print(f"[DEBUG] PnP inliers: {len(inliers) if inliers is not None else 0}")

    if not success or inliers is None or len(inliers) < 6:
        raise RuntimeError("PnP RANSAC failed or not enough inliers")

    # reprojection error 시각화
    projected_pts, _ = cv2.projectPoints(object_pts, rvec, tvec, K, None)
    projected_pts = projected_pts.reshape(-1, 2)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pt3, pt2 in zip(projected_pts, pts2d):
        cv2.circle(vis, tuple(pt2.astype(int)), 2, (0, 255, 0), -1)  # gt 2D (녹색)
        cv2.circle(vis, tuple(pt3.astype(int)), 2, (0, 0, 255), -1)  # reproj (빨강)
        cv2.line(vis, tuple(pt2.astype(int)), tuple(pt3.astype(int)), (255, 0, 0), 1)

    return rvec, tvec, vis


# 3D -> 2D projection 시각화 함수
def draw_projection(img, rvec, tvec, pts3d, color=(0, 255, 0)):
    projected_pts, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    projected_pts = projected_pts.reshape(-1, 2).astype(int)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pt in projected_pts:
        cv2.circle(vis, tuple(pt), 4, color, -1)
    return vis

def evaluate_pnp_with_gt_2d(img, rvec, tvec, pts3d, pts2d_gt, K, color_proj=(0, 0, 255), color_gt=(0, 255, 0), window_name="PnP reprojection vs GT"):
    """PnP 결과 검증: 3D pts를 투영해서 gt 2D 좌표와 비교"""
    projected, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    projected = projected.reshape(-1, 2)

    pts2d_gt = pts2d_gt.reshape(-1, 2)

    if len(projected) != len(pts2d_gt):
        print(f"[WARN] Projected points: {len(projected)}, GT points: {len(pts2d_gt)} (mismatch)")
        min_len = min(len(projected), len(pts2d_gt))
        projected = projected[:min_len]
        pts2d_gt = pts2d_gt[:min_len]

    error = np.linalg.norm(projected - pts2d_gt, axis=1)
    print(f"[DEBUG] Mean reprojection error (to GT 2D): {np.mean(error):.2f}px")

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pt_gt, pt_proj in zip(pts2d_gt, projected):
        cv2.circle(vis, tuple(np.clip(pt_gt.astype(int), 0, [img.shape[1]-1, img.shape[0]-1])), 3, color_gt, -1)
        cv2.circle(vis, tuple(np.clip(pt_proj.astype(int), 0, [img.shape[1]-1, img.shape[0]-1])), 3, color_proj, -1)
        cv2.line(vis, tuple(pt_gt.astype(int)), tuple(pt_proj.astype(int)), (255, 0, 0), 1)

    cv2.imshow(window_name, vis)
    cv2.waitKey(0)


