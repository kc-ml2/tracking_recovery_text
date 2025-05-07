import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R, Slerp

from utils_connecting_maps import (
    load_image_parsed,
    load_keyframe_trajectory,
    interpolate_pose,
    compute_relative_transformation,
    scale_calibrated_keyframe_trajectory,
    transform_and_save_nextmap_trajectory,
)

# Load config file
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Load key paths
image_parsed_path = config["txt_parsed_path"]
firstmap_trajectory_path = config["firstmap_trajectory_path"]
nextmap_trajectory_path = config["nextmap_trajectory_path"]
firstmap_new_trajectory_path = config["firstmap_new_trajectory_path"]
nextmap_new_trajectory_path = config["nextmap_new_trajectory_path"]

# Load poses of selected 4 frames in unknown coordinate from COLMAP result
image_parsed = load_image_parsed(image_parsed_path)

# If poses of 4 frames are well reconstructed by COLMAP, connect oldmap and newmap
if (len(image_parsed)==4):
    print("=== COLMAP can well be made ===")

    # Obtain poses of fi in f1 coordinate
    Q1, t1 = np.array([0, 0, 0, 1]), np.array([0, 0, 0])
    Q21, t21 = compute_relative_transformation(image_parsed[1], image_parsed[2])
    Q31, t31 = compute_relative_transformation(image_parsed[1], image_parsed[3])
    Q41, t41 = compute_relative_transformation(image_parsed[1], image_parsed[4])

    # Obtain poses of f1, f2 in oldmap coordinate(O), f3, f4 in newmap coordinate(N)
    firstmap_trajectory = load_keyframe_trajectory(firstmap_trajectory_path)
    nextmap_trajectory = load_keyframe_trajectory(nextmap_trajectory_path)

    timestamp1 = float(image_parsed[1]["timestamp"].replace(".png", ""))
    timestamp2 = float(image_parsed[2]["timestamp"].replace(".png", ""))
    timestamp3 = float(image_parsed[3]["timestamp"].replace(".png", ""))
    timestamp4 = float(image_parsed[4]["timestamp"].replace(".png", ""))

    Q1O_temp, t1O_temp = interpolate_pose(timestamp1, firstmap_trajectory)
    Q2O_temp, t2O_temp = interpolate_pose(timestamp2, firstmap_trajectory)
    Q3N_temp, t3N_temp = interpolate_pose(timestamp3, nextmap_trajectory)
    Q4N_temp, t4N_temp = interpolate_pose(timestamp4, nextmap_trajectory)

    # Scale correction of newmap
    R21_temp = (R.from_quat(Q2O_temp) * R.from_quat(Q1O_temp).inv()).as_matrix() 
    t21_temp = t2O_temp - t1O_temp
    len_t21_temp = np.linalg.norm(t21_temp)
    len_t21 = np.linalg.norm(t21)
    firstmap_scale = len_t21 / len_t21_temp

    R43_temp = (R.from_quat(Q4N_temp) * R.from_quat(Q3N_temp).inv()).as_matrix() 
    t43_temp = t4N_temp - t3N_temp
    R43 = (R.from_quat(Q41) * R.from_quat(Q31).inv()).as_matrix() 
    t43 = t41-t31
    len_t43_temp = np.linalg.norm(t43_temp)
    len_t43 = np.linalg.norm(t43)
    nextmap_scale = len_t43 / len_t43_temp

    scale_factor = nextmap_scale / firstmap_scale

    scale_calibrated_keyframe_trajectory(firstmap_trajectory, 1, firstmap_new_trajectory_path)
    scale_calibrated_keyframe_trajectory(nextmap_trajectory, 1, nextmap_new_trajectory_path)

    # Transform and connect newmap to oldmap using the obtained poses and scale
    firstmap_new_trajectory = load_keyframe_trajectory(firstmap_new_trajectory_path)
    nextmap_new_trajectory = load_keyframe_trajectory(nextmap_new_trajectory_path)

    Q1O, t1O = interpolate_pose(timestamp1, firstmap_new_trajectory)
    Q2O, t2O = interpolate_pose(timestamp2, firstmap_new_trajectory)
    Q3N, t3N = interpolate_pose(timestamp3, nextmap_new_trajectory)
    Q4N, t4N = interpolate_pose(timestamp4, nextmap_new_trajectory)

    QBO, tBO = np.array(firstmap_new_trajectory[-1]["quaternion"]), np.array(firstmap_new_trajectory[-1]["translation"])

    R32 = (R.from_quat(Q31) * R.from_quat(Q21).inv())
    R23 = R32.inv().as_matrix() #1

    R13 = R.from_quat(Q31).inv().as_matrix()
    R3O = R13 @ R.from_quat(Q1O).inv().as_matrix() #2

    transform_and_save_nextmap_trajectory(nextmap_new_trajectory, R3O, tBO, config["firstmap_new_trajectory_path"], config["final_trajectory_path"])
 
elif (2 in image_parsed and 3 in image_parsed):
    print("=== Colmap can well be made for only orb2, orb3 ===")

else:
    print("[ERROR] Insufficient Colmap!")




