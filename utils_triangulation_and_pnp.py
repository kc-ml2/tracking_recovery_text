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

    kp1, kp2, matches, _, _, _ = orb_feature_matching(crop1, crop2, False)

    if len(matches) < 8:
        raise ValueError("Not enough matches for triangulation")

    # crop 영역 좌표를 원래 이미지 기준으로 보정
    expand = 30 
    pts1_all = np.float32([kp1[m.queryIdx].pt for m in matches]) + np.array([bbox1[0] - expand, bbox1[1] - expand]) # 매칭점들의 전체 이미지 기준 2D 좌표
    pts2_all = np.float32([kp2[m.trainIdx].pt for m in matches]) + np.array([bbox2[0] - expand, bbox2[1] - expand])

    # RANSAC 필터링
    E, mask = cv2.findEssentialMat(pts1_all, pts2_all, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    _, R, t, pose_mask = cv2.recoverPose(E, pts1_all, pts2_all, K, mask=mask)

    # Inlier만 필터링
    inliers1 = pts1_all[pose_mask.ravel() == 1]
    inliers2 = pts2_all[pose_mask.ravel() == 1]
    inlier_matches = [matches[i] for i in range(len(matches)) if pose_mask.ravel()[i] == 1]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in inlier_matches]) + np.array([bbox1[0] - expand, bbox1[1] - expand])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in inlier_matches]) + np.array([bbox2[0] - expand, bbox2[1] - expand])

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

    expand = 30
    offset1 = np.array([bbox1[0] - expand, bbox1[1] - expand])
    used_kp1_corrected = []
    for kp in used_kp1:
        new_kp = cv2.KeyPoint(
            kp.pt[0] + offset1[0],
            kp.pt[1] + offset1[1],
            kp.size, kp.angle, kp.response,
            kp.octave, kp.class_id
        )
        used_kp1_corrected.append(new_kp)
       
    offset2 = np.array([bbox2[0] - expand, bbox2[1] - expand])
    used_kp2_corrected = []
    for kp in used_kp2:
        new_kp = cv2.KeyPoint(
            kp.pt[0] + offset2[0],
            kp.pt[1] + offset2[1],
            kp.size, kp.angle, kp.response,
            kp.octave, kp.class_id
        )
        used_kp2_corrected.append(new_kp)

    return pts_3d, pts1, pts2, used_kp1_corrected, used_kp2_corrected, R, t


# estimate_pose 함수
def estimate_pose(img, bbox, pts3d, used_kp1_corrected, ref_img):
    # 1. crop된 이미지에서 feature 추출
    crop = crop_fn(img, bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], expand=30)
    orb = cv2.ORB_create(1000)
    kp2, des2 = orb.detectAndCompute(crop, None)

    # 2. reference 이미지에서 used_kp1_corrected의 descriptor 계산
    des1 = orb.compute(ref_img, used_kp1_corrected)[1]

    if des1 is None or len(des1) != len(used_kp1_corrected):
        raise ValueError("Descriptor extraction failed for used_kp1_corrected")

    # 3. 매칭
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    if len(matches) < 6:
        raise ValueError("Not enough matches between reference and target")

    if des2 is None or len(kp2) < 10:
        raise ValueError("Not enough keypoints detected in new image")

    # 4. 좌표계 원본 이미지에 맞추기
    expand = 30
    offset = np.array([bbox[0] - expand, bbox[1] - expand])

    kp2_corrected = []
    for kp in kp2:
        new_kp = cv2.KeyPoint(
            kp.pt[0] + offset[0],
            kp.pt[1] + offset[1],
            kp.size, kp.angle, kp.response,
            kp.octave, kp.class_id
        )
        kp2_corrected.append(new_kp)

    # 1~4 Debug
    # img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for kp in kp2_corrected:
    #     pt = tuple(np.round(kp.pt).astype(int))
    #     cv2.circle(img_vis, pt, 2, (0, 255, 0), -1)
    # cv2.imshow("kp2 distribution", img_vis)
    # cv2.waitKey(0)
    # visualize_matches(ref_img, img, used_kp1_corrected, kp2_corrected, matches, title="Feature Matching")

    # 5. match 기반 2D-3D correspondence
    pts2d = [] # 2D
    object_pts = [] # 3D
    for m in matches:
        pt2 = kp2_corrected[m.trainIdx].pt
        pts2d.append(pt2)
        object_pts.append(pts3d[m.queryIdx])

    pts2d = np.float32(pts2d)
    object_pts = np.float32(object_pts)

    # 6. PnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_pts, pts2d, K, None,
        reprojectionError=8.0,
        iterationsCount=100,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success or inliers is None or len(inliers) < 6:
        raise RuntimeError("PnP RANSAC failed or not enough inliers")

    print(f"\n[DEBUG] PnP inliers: {len(inliers) if inliers is not None else 0}")

    # 7. 시각화 (inlier만 사용)
    inlier_idx = inliers.ravel()
    pts2d_in = pts2d[inlier_idx]
    object_pts_in = object_pts[inlier_idx]
    projected_pts, _ = cv2.projectPoints(object_pts_in, rvec, tvec, K, None)
    projected_pts = projected_pts.reshape(-1, 2)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pt_gt, pt_proj in zip(pts2d_in, projected_pts):
        cv2.circle(vis, tuple(pt_gt.astype(int)), 2, (0, 255, 0), -1) # 초록: 실제 2D 점
        cv2.circle(vis, tuple(pt_proj.astype(int)), 2, (0, 0, 255), -1) # 빨강: reprojection 결과
        cv2.line(vis, tuple(pt_gt.astype(int)), tuple(pt_proj.astype(int)), (255, 0, 0), 1)

    cv2.imshow("PnP reprojection", vis)
    cv2.waitKey(0)

    error = np.linalg.norm(pts2d_in - projected_pts.reshape(-1, 2), axis=1)
    print(f"[DEBUG] Mean reprojection error: {np.mean(error):.2f}px")

    return rvec, tvec, vis
