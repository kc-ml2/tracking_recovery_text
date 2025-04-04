import cv2
import numpy as np
import yaml

from utils_triangulation_and_pnp import (
    load_input,
    triangulate,
    estimate_pose,
    draw_projection,
    evaluate_pnp_with_gt_2d
)

# Load config.yaml
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Load paths
file_path = config["file_path"]
img_dir = file_path + "/images/"

# Intrinsic parameter matrix
K = np.array([[641.136535, 0, 656.977416],
              [0, 640.351623, 365.402496],
              [0, 0, 1]], dtype=np.float32)

# Load bbox information
entries = load_input("triangulation_input.txt")

# Load 4 images
img1 = cv2.imread(img_dir + entries[0][0], cv2.IMREAD_GRAYSCALE) # firstmap image 1
bbox1 = entries[0][1]
img2 = cv2.imread(img_dir + entries[1][0], cv2.IMREAD_GRAYSCALE) # firstmap image 2
bbox2 = entries[1][1]
img3 = cv2.imread(img_dir + entries[2][0], cv2.IMREAD_GRAYSCALE) # nextmap image 1
bbox3 = entries[2][1]
img4 = cv2.imread(img_dir + entries[3][0], cv2.IMREAD_GRAYSCALE) # nextmap image 2
bbox4 = entries[3][1]

# Triangulation
pts3d, pts1, pts2, used_kp1, used_kp2, R_12, t_12 = triangulate(img1, img2, bbox1, bbox2)
print(f"R_12 = {R_12.T}")
print(f"t_12 = {t_12.T}")

# Triangulation debug
# 3D 점 → image1에 reproject (camera 좌표계 기준)
pts3d_cam = pts3d.T  # shape: (3, N)
pts3d_cam = pts3d_cam.reshape(3, -1)

# normalize
projected = pts3d_cam[:2] / pts3d_cam[2]  # shape: (2, N)
projected = projected.T.reshape(-1, 1, 2)

# 다시 pixel 좌표계로 변환
projected_img = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K, None)[0].reshape(-1, 2)

# pts1도 undistort 해줘서 동일 좌표계로 맞추기
pts1_undistort = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)

# error 계산 (같은 좌표계 기준에서)
error = np.linalg.norm(pts1_undistort - projected.reshape(-1, 2), axis=1)
print("Mean undistorted reprojection error:", np.mean(error))

z = pts3d[:, 2]
print("Depth min:", z.min(), " max:", z.max(), " mean:", z.mean())
print("Positive depths ratio:", np.mean(z > 0) * 100, "%")

vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
h, w = img1.shape

for pt in pts1.astype(int):
    pt = np.clip(pt, 0, [w - 1, h - 1])
    pt = pt.ravel()
    cv2.circle(vis, tuple(pt), 2, (0, 255, 0), -1)  # 원래 2D 좌표 (녹색)

for pt in projected_img.astype(int):
    pt = np.clip(pt, 0, [w - 1, h - 1])
    pt = pt.ravel()
    cv2.circle(vis, tuple(pt), 2, (0, 0, 255), -1)  # reproject된 점 (빨강)

cv2.imshow("pts1 (green) vs reprojected (red)", vis)
cv2.waitKey(0)

# PnP
rvec3, tvec3, vis3 = estimate_pose(img3, bbox3, pts3d, used_kp1, img1)
print("Nextmap pose 1 (rvec, tvec):")
cv2.imshow("PnP img3 reprojection error", vis3)
print(f"R_13 = {rvec3.T}")
print(f"t_13 = {tvec3.T}")

rvec4, tvec4, vis4 = estimate_pose(img4, bbox4, pts3d, used_kp1, img1)
print("Nextmap pose 2 (rvec, tvec):")
cv2.imshow("PnP img4 reprojection error", vis4)
cv2.waitKey(0)
print(f"R_14 = {rvec4.T}")
print(f"t_14 = {tvec4.T}")

# # Nextmap image reprojection visualization
# vis_img3 = draw_projection(img3, rvec3, tvec3, pts3d)
# cv2.imshow("Projected 3D points on Nextmap image 1", vis_img3)
# cv2.waitKey(0)

# vis_img4 = draw_projection(img4, rvec4, tvec4, pts3d)
# cv2.imshow("Projected 3D points on Nextmap image 2", vis_img4)
# cv2.waitKey(0)

# evaluate_pnp_with_gt_2d(img3, rvec3, tvec3, pts3d, pts2, K, window_name="PnP img3 reprojection error")
# evaluate_pnp_with_gt_2d(img4, rvec4, tvec4, pts3d, pts2, K, window_name="PnP img4 reprojection error")
