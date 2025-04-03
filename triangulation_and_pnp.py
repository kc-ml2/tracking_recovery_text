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
    visualize_matches(crop1, crop2, kp1, kp2, matches, f"two images crop matches!")

    if len(matches) < 8:
        raise ValueError("Not enough matches for triangulation")

    # crop 영역 좌표를 원래 이미지 기준으로 보정
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]) + np.array([bbox1[0], bbox1[1]])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]) + np.array([bbox2[0], bbox2[1]])

    E, _ = cv2.findEssentialMat(pts1, pts2, K)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    pts1_norm = cv2.undistortPoints(pts1.reshape(-1,1,2), K, None)
    pts2_norm = cv2.undistortPoints(pts2.reshape(-1,1,2), K, None)

    pts_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T

    return pts_3d, pts1

def estimate_pose(img, bbox, pts3d):
    # crop 이미지
    crop = crop_fn(img, bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], expand=30)

    # ORB 특징점 추출
    kp, des = cv2.ORB_create(1000).detectAndCompute(crop, None)

    if des is None or len(kp) < 10:
        raise ValueError("Not enough keypoints detected")

    # 2D 포인트는 crop된 키포인트에 bbox 오프셋 더해서 원본 좌표로
    pts2d = np.float32([kp[i].pt for i in range(min(len(kp), len(pts3d)))])
    pts2d += np.array([bbox[0], bbox[1]])

    # 대응되는 3D 포인트 수만큼 맞춰 자르기
    object_pts = pts3d[:len(pts2d)]

    # PnP 수행
    success, rvec, tvec = cv2.solvePnP(object_pts, pts2d, K, None)
    if not success:
        raise RuntimeError("PnP failed")

    return rvec, tvec

# 3D -> 2D projection 시각화 함수
def draw_projection(img, rvec, tvec, pts3d, color=(0, 255, 0)):
    projected_pts, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    projected_pts = projected_pts.reshape(-1, 2).astype(int)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pt in projected_pts:
        cv2.circle(vis, tuple(pt), 4, color, -1)
    return vis

# 실행
entries = load_input("triangulation_input.txt")

img1 = cv2.imread(img_dir + entries[0][0], cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img_dir + entries[1][0], cv2.IMREAD_GRAYSCALE)

pts3d, pts1 = triangulate(img1, img2, entries[0][1], entries[1][1])
print("Triangulated 3D points:\n", pts3d)

# pts3d: (N, 3), 기준은 첫 번째 이미지
pts3d_hom = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))  # (N, 4)
proj_matrix = K @ np.hstack((np.eye(3), np.zeros((3,1))))     # P1

reprojected = (proj_matrix @ pts3d_hom.T).T  # (N, 3)
reprojected = reprojected[:, :2] / reprojected[:, 2, np.newaxis]  # Normalize

error = np.linalg.norm(reprojected - pts1, axis=1)  # pts1은 원래 이미지 1의 2D 점
print(f"평균 reprojection error: {np.mean(error):.2f}px")

img3 = cv2.imread(img_dir + entries[2][0], cv2.IMREAD_GRAYSCALE)
bbox3 = entries[2][1]

rvec3, tvec3 = estimate_pose(img3, bbox3, pts3d)
print("Nextmap pose 1 (rvec, tvec):")
print(rvec3.T)
print(tvec3.T)

# Nextmap 이미지 불러온 후 시각화
vis_img3 = draw_projection(img3, rvec3, tvec3, pts3d)
cv2.imshow("Projected 3D points on Nextmap image", vis_img3)
cv2.waitKey(0)


