import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils_connecting_maps import load_keyframe_trajectory

# ===========================
# 설정 파일 불러오기
# ===========================
with open("config_long.yaml", 'r') as f:
    config = yaml.safe_load(f)

root_dir = config["root_dir"]
date_str = config["date_str"]
time_str = config["time_str"]
sequences = config["sequences"]
connecting_dir = config["connecting_dir"]

traj_base_path = os.path.join(root_dir, f"result_{date_str}_{time_str}", "mono_result")
out_path = "/home/youngsun/vslam/corl/ConnectingMaps/connected_long.txt"

# ===========================
# 첫 시퀀스 저장
# ===========================
first_path = os.path.join(traj_base_path, f"KeyFrameTrajectory0{sequences[0]}.txt")
first_traj = load_keyframe_trajectory(first_path)
merged = first_traj

# 누적 기준이 아닌 직전 pose만 사용
base_pose = merged[-1]
# ===========================
# 나머지 시퀀스 처리 루프
# ===========================
for i in range(1, len(sequences)):
    # prev_idx = sequences[i - 1]
    curr_idx = sequences[i]

    print(f"Transforming manually {curr_idx}")
    new_txt_path = os.path.join(traj_base_path, f"KeyFrameTrajectory0{curr_idx}.txt")
    new_traj = load_keyframe_trajectory(new_txt_path)

    # 기준 포즈로부터 회전/이동 변환 생성
    R_base = R.from_quat(base_pose['quaternion'])
    t_base = np.array(base_pose['translation'])

    transformed = []

    for pose in new_traj:
        t = np.array(pose['translation'])
        q = R.from_quat(pose['quaternion'])

        t_new = R_base.apply(t) + t_base
        q_new = (R_base * q).as_quat()

        transformed.append({
            'timestamp': pose['timestamp'],
            'translation': t_new.tolist(),
            'quaternion': q_new.tolist()
        })

    merged = merged + transformed
    base_pose = merged[-1]

with open(out_path, 'w') as f:
    for pose in merged:
        line = "{:.6f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(
            pose['timestamp'],
            pose['translation'][0],
            pose['translation'][1],
            pose['translation'][2],
            pose['quaternion'][0],
            pose['quaternion'][1],
            pose['quaternion'][2],
            pose['quaternion'][3]
        )
        f.write(line)





