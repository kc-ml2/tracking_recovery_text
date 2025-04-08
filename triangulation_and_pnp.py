import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils_triangulation_and_pnp import (
    load_input,
    triangulate,
    estimate_pose
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

R2 = np.eye(3)
t2 = np.zeros((3, 1))

# Triangulation
pts3d, pts1, pts2, des1_used, des2_used, used_kp1_corrected, used_kp2_corrected, R3, t3 = triangulate(img2, img3, bbox2, bbox3)
print(f"\nR3 = {R3}")
print(f"t3 = {t3.T}")

# _, _, _, _, _, _, _, R1, t1 = triangulate(img3, img1, bbox3, bbox1)
# print(f"\nR1 = {R1}")
# print(f"t1 = {t1.T}")

# _, _, _, _, _, _, _, R4, t4 = triangulate(img2, img4, bbox2, bbox4)
# R4 = R4 @ R3
# t4 = t4 + t3
# print(f"\nR4 = {R4}")
# print(f"t4 = {t4.T}")

# # Triangulation Debug
# # 1. depths 값 확인
# z = pts3d[:, 2]
# print("[DEBUG] Depth (z) range:", z.min(), "~", z.max())
# print("[DEBUG] Positive depth ratio:", np.mean(z > 0) * 100, "%")

# # 2. Reprojecition error 시각화
# projected, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K, None)
# projected = projected.reshape(-1, 2)

# pts1_undistort = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)

# error = np.linalg.norm(pts1_undistort - projected / K[[0,1],[0,1]], axis=1)
# print("[DEBUG] Mean reprojection error:", np.mean(error), "px")

# # 3. 3d -> 2d 시각화
# def draw_projection(img, pts3d, color=(0, 255, 0)):
#     projected, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K, None)
#     projected = projected.reshape(-1, 2).astype(int)

#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     for pt in projected:
#         pt = tuple(np.clip(pt, 0, [img.shape[1] - 1, img.shape[0] - 1]))
#         cv2.circle(vis, pt, 3, color, -1)
#     return vis
# cv2.imshow("Projected 3D pts on img2", draw_projection(img2, pts3d))
# cv2.waitKey(0)

# PnP
rvec4, t4, vis4 = estimate_pose(img4, bbox4, pts3d, des1_used, used_kp1_corrected, img2)
R4, _ = cv2.Rodrigues(rvec4)
print(f"\nR4 = {R4}")
print(f"t4 = {t4.T}")

# rvec1, t1, vis1 = estimate_pose(img1, bbox1, pts3d, des1_used, used_kp1_corrected, img2)
# R1, _ = cv2.Rodrigues(rvec1)
# print(f"\nR1 = {R1}")
# print(f"t1 = {t1.T}")

# PnP Debug
# visualize all images
# -------------------- 시각화 함수 --------------------
def camera_position(R, t):
    return -R.T @ t

def draw_camera(ax, R, t, scale=50.0, label=None, color='r'):
    cam_axes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale
    cam_axes_world = (R.T @ cam_axes.T).T + (-R.T @ t).reshape(1, 3)
    origin = cam_axes_world[0]
    ax.plot(*zip(origin, cam_axes_world[1]), color='r')
    ax.plot(*zip(origin, cam_axes_world[2]), color='g')
    ax.plot(*zip(origin, cam_axes_world[3]), color='b')
    if label:
        ax.text(*origin, label, color=color)

R_list = [R1, R2, R3, R4]
t_list = [t1, t2, t3, t4]
labels = ['img1', 'img2', 'img3', 'img4']

# 카메라 위치 계산
cam1_pos = camera_position(R1, t1).flatten()
cam2_pos = camera_position(R2, t2).flatten()
cam3_pos = camera_position(R3, t3).flatten()
cam4_pos = camera_position(R4, t4).flatten()

# ----------------------- Triangulation 시각화 --------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1)3D 포인트
ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c='cyan', s=10, label='Triangulated 3D Points')

# 2)카메라 위치
#ax.scatter(*cam1_pos, c='red', s=50, label='Camera 1 (img1)')
ax.scatter(*cam2_pos, c='orange', s=50, label='Camera 2 (img2)')
ax.scatter(*cam3_pos, c='blue', s=50, label='Camera 3 (img3)')
#ax.scatter(*cam4_pos, c='green', s=50, label='Camera 4 (img4)')

cam_positions = np.array([cam2_pos, cam3_pos])
ax.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], c='black', linewidth=2, label='Camera Path')

# 축 라벨 및 시점 설정
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Triangulated 3D Points + 2Camera Positions")
ax.view_init(elev=20, azim=-60)

plt.legend()
plt.tight_layout()
plt.show()

# -------------------------- Triangulation + PnP 시각화 ------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1)3D 포인트
ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c='cyan', s=10, label='Triangulated 3D Points')

# 2)카메라 위치
ax.scatter(*cam1_pos, c='red', s=50, label='Camera 1 (img1)')
ax.scatter(*cam2_pos, c='orange', s=50, label='Camera 2 (img2)')
ax.scatter(*cam3_pos, c='blue', s=50, label='Camera 3 (img3)')
ax.scatter(*cam4_pos, c='green', s=50, label='Camera 4 (img4)')

cam_positions = np.array([cam1_pos, cam2_pos, cam3_pos, cam4_pos])
ax.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], c='black', linewidth=2, label='Camera Path')

# 축 라벨 및 시점 설정
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Triangulated 3D Points + 4 Camera Positions")
ax.view_init(elev=20, azim=-60)

plt.legend()
plt.tight_layout()
plt.show()