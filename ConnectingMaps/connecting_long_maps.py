import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils_connecting_maps import load_keyframe_trajectory

# ===========================
# 설정 파일 불러오기
# ===========================
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

root_dir = config["root_dir"]
date_str = config["date_str"]
time_str = config["time_str"]
sequences = config["sequences"]
connecting_dir = config["connecting_dir"]

traj_base_path = os.path.join(root_dir, f"result_{date_str}_{time_str}", "mono_result")
# out_path = os.path.join(traj_base_path, "trajectory_merged!!.txt")
out_path = "/home/youngsun/vslam/corl/ConnectingMaps/connected_long.txt"

# ===========================
# Trajectory 이어붙이기 함수
# ===========================
def merge_and_transform(prev_traj, curr_traj, output_path):
    tA = np.array(prev_traj[-1]['translation'])
    qA = R.from_quat(prev_traj[-1]['quaternion'])

    tB = np.array(curr_traj[0]['translation'])
    qB = R.from_quat(curr_traj[0]['quaternion'])

    # 상대 변환
    R_rel = qA * qB.inv()
    t_rel = tA - R_rel.apply(tB)

    # 변환 적용
    transformed = []
    for pose in curr_traj:
        t_orig = np.array(pose['translation'])
        q_orig = R.from_quat(pose['quaternion'])

        t_new = R_rel.apply(t_orig) + t_rel
        q_new = (R_rel * q_orig).as_quat()

        transformed.append({
            'timestamp': pose['timestamp'],
            'translation': t_new.tolist(),
            'quaternion': q_new.tolist()
        })

    # 저장
    with open(output_path, 'a') as f:
        for pose in transformed:
            line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} " \
                   f"{pose['quaternion'][0]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]}\n"
            f.write(line)


# ===========================
# 전체 시퀀스 처리 루프
# ===========================
first_path = os.path.join(traj_base_path, f"KeyFrameTrajectory{sequences[0]}.txt")
first_traj = load_keyframe_trajectory(first_path)

# 초기화
with open(out_path, 'w') as f:
    for pose in first_traj:
        line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} " \
               f"{pose['quaternion'][0]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]}\n"
        f.write(line)

# 나머지 시퀀스 반복
for i in range(1, len(sequences)):
    prev_idx = sequences[i - 1]
    curr_idx = sequences[i]

    # 변환된 Trajectory 존재하는 경우: KeyFrameTrajectory{prev}{curr}_new.txt
    new_txt_path = os.path.join(
        connecting_dir, date_str, time_str, "yolo", f"{curr_idx}th",
        "four_frame_result", "0", "txt", f"KeyFrameTrajectory{prev_idx}{curr_idx}_new.txt"
    )

    if os.path.exists(new_txt_path):
        print(f"Appending precomputed {prev_idx}->{curr_idx} from {new_txt_path}")
        new_traj = load_keyframe_trajectory(new_txt_path)
    else:
        print(f"Transforming manually {prev_idx}->{curr_idx}")
        new_txt_path = os.path.join(traj_base_path, f"KeyFrameTrajectory{curr_idx}.txt")
        new_traj = load_keyframe_trajectory(new_txt_path)
        prev_traj = load_keyframe_trajectory(out_path)  # 현재까지 결과
        merge_and_transform(prev_traj, new_traj, out_path)
        continue

    # 그냥 이어붙이기 (변환 이미 적용된 경우)
    with open(out_path, 'a') as f:
        for pose in new_traj:
            line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} " \
                   f"{pose['quaternion'][0]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]}\n"
            f.write(line)

