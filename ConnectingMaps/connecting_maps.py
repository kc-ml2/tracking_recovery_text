import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R, Slerp

from utils_connecting_maps import (
    load_image_parsed,
    load_keyframe_trajectory,
    interpolate_pose,
    compute_relative_transformation,
    scale_calibrated_keyframe_trajectory,
    transform_and_save_nextmap_trajectory
)

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

image_parsed_path = config["txt_parsed_path"]
firstmap_trajectory_path = config["firstmap_trajectory_path"]
nextmap_trajectory_path = config["nextmap_trajectory_path"]
firstmap_new_trajectory_path = config["firstmap_new_trajectory_path"]
nextmap_new_trajectory_path = config["nextmap_new_trajectory_path"]

# 1. Q, t of image1~4 at image1 coordinate
image_parsed = load_image_parsed(image_parsed_path)

if (len(image_parsed)==4):
    print("=== Colmap can well be made ===") # long sequence 확인용
    Q1, t1 = np.array([0, 0, 0, 1]), np.array([0, 0, 0])
    Q21, t21 = compute_relative_transformation(image_parsed[1], image_parsed[2])
    Q31, t31 = compute_relative_transformation(image_parsed[1], image_parsed[3])
    Q41, t41 = compute_relative_transformation(image_parsed[1], image_parsed[4])

    # 2. Q, t of image1 and 2 at A coordinate, image3 and 4 at B coordinate (interploarization)
    firstmap_trajectory = load_keyframe_trajectory(firstmap_trajectory_path)
    nextmap_trajectory = load_keyframe_trajectory(nextmap_trajectory_path)

    timestamp1 = float(image_parsed[1]["timestamp"].replace(".png", ""))
    timestamp2 = float(image_parsed[2]["timestamp"].replace(".png", ""))
    timestamp3 = float(image_parsed[3]["timestamp"].replace(".png", ""))
    timestamp4 = float(image_parsed[4]["timestamp"].replace(".png", ""))

    Q1A_temp, t1A_temp = interpolate_pose(timestamp1, firstmap_trajectory)
    Q2A_temp, t2A_temp = interpolate_pose(timestamp2, firstmap_trajectory)
    Q3C_temp, t3C_temp = interpolate_pose(timestamp3, nextmap_trajectory)
    Q4C_temp, t4C_temp = interpolate_pose(timestamp4, nextmap_trajectory)

    # 3. scale correction
    # 1--2 scale
    R21_temp = (R.from_quat(Q2A_temp) * R.from_quat(Q1A_temp).inv()).as_matrix() 
    t21_temp = t2A_temp - t1A_temp

    len_t21_temp = np.linalg.norm(t21_temp)
    len_t21 = np.linalg.norm(t21)
    firstmap_scale = len_t21 / len_t21_temp

    # print("\nFirstmap scale") # debug
    # print(firstmap_scale)

    # 3--4 scale
    R43_temp = (R.from_quat(Q4C_temp) * R.from_quat(Q3C_temp).inv()).as_matrix() 
    t43_temp = t4C_temp - t3C_temp
    R43 = (R.from_quat(Q41) * R.from_quat(Q31).inv()).as_matrix() 
    t43 = t4C_temp - t3C_temp

    len_t43_temp = np.linalg.norm(t43_temp)
    len_t43 = np.linalg.norm(t43)
    nextmap_scale = len_t43 / len_t43_temp

    # print("\nNextmap scale") # debug
    # print(nextmap_scale)

    # print("\ncomparing R") # debug
    # print(R12_temp)
    # print(R.from_quat(Q2).as_matrix())
    # print(t12_temp)
    # print(t2)

    # save to trajectory file
    scale_calibrated_keyframe_trajectory(firstmap_trajectory, firstmap_scale, firstmap_new_trajectory_path)
    scale_calibrated_keyframe_trajectory(nextmap_trajectory, nextmap_scale, nextmap_new_trajectory_path)

    # 4. change coordinate of nextmap and apply to trajectory file
    firstmap_new_trajectory = load_keyframe_trajectory(firstmap_new_trajectory_path)
    nextmap_new_trajectory = load_keyframe_trajectory(nextmap_new_trajectory_path)

    Q1A, t1A = interpolate_pose(timestamp1, firstmap_new_trajectory)
    Q2A, t2A = interpolate_pose(timestamp2, firstmap_new_trajectory)
    Q3C, t3C = interpolate_pose(timestamp3, nextmap_new_trajectory)
    Q4C, t4C = interpolate_pose(timestamp4, nextmap_new_trajectory)

    QBA, tBA = np.array(firstmap_new_trajectory[-1]["quaternion"]), np.array(firstmap_new_trajectory[-1]["translation"])
    RBA = R.from_quat(QBA).inv().as_matrix()

    RC3 = R.from_quat(Q3C).inv()
    R32 = (R.from_quat(Q31) * R.from_quat(Q21).inv())
    R21 = R.from_quat(Q21)
    R1A = R.from_quat(Q1A)
    RCA = RC3 * R32 * R21 * R1A
    RAC = RCA.inv().as_matrix()

    transform_and_save_nextmap_trajectory(nextmap_new_trajectory, RAC, tBA, config["firstmap_new_trajectory_path"], config["final_trajectory_path"])

elif (2 in image_parsed and 3 in image_parsed):
    print("=== Colmap can well be made for only orb2, orb3 ===")

else:
    print("[ERROR] Insufficient Colmap!")




