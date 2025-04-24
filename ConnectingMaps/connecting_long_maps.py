import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils_connecting_maps import load_keyframe_trajectory, load_image_parsed

# ===========================
# 설정 파일 불러오기
# ===========================
with open("/home/youngsun/vslam/corl/ConnectingMaps/config_long.yaml", 'r') as f:
    config = yaml.safe_load(f)

root_dir = config["root_dir"]
date_str = config["date_str"]
time_str = config["time_str"]
sequences = config["sequences"]
connecting_dir = config["connecting_dir"]

traj_base_path = os.path.join(root_dir, f"result_{date_str}_{time_str}", "mono_result") # mono or orb2
out_path = config["out_path"]

# ===========================
# Trajectory 이어붙이기 함수
# ===========================
def merge_and_transform(prev_traj, curr_traj, output_path):

    tA = np.array(prev_traj[-1]['translation'])
    qA = R.from_quat(prev_traj[-1]['quaternion'])

    # tB = np.array(curr_traj[0]['translation'])
    # qB = R.from_quat(curr_traj[0]['quaternion'])

    # # 상대 변환
    # R_rel = qA * qB.inv()
    # t_rel = tA - qA.apply(tB)

    R_rel = qA 
    t_rel = tA 

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
open(out_path, 'w').close()
with open(out_path, 'w') as f:
    for pose in first_traj:
        line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} " \
               f"{pose['quaternion'][0]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]}\n"
        f.write(line)

# 나머지 시퀀스 반복
for i in range(1, len(sequences)):
    prev_idx = sequences[i - 1]
    curr_idx = sequences[i]
    int_curr_idx = int(curr_idx)

    new_txt_path = os.path.join(
        connecting_dir, date_str, time_str, "yolo", f"{int_curr_idx}th",
        "four_frame_result", "0", "txt", f"KeyFrameTrajectory{prev_idx}{curr_idx}_new.txt"
    )

    firstmap_new_traj_path = os.path.join(
        connecting_dir, date_str, time_str, "yolo", f"{int_curr_idx}th",
        "four_frame_result", "0", "txt", f"KeyFrameTrajectory{prev_idx}_new.txt"
    )


    nextmap_new_traj_path = os.path.join(
        connecting_dir, date_str, time_str, "yolo", f"{int_curr_idx}th",
        "four_frame_result", "0", "txt", f"KeyFrameTrajectory{curr_idx}_new.txt"
    )

    # check if we can generate new_txt_path
    image_parsed_txt = os.path.join(
        connecting_dir, date_str, time_str, "yolo", f"{int_curr_idx}th",
        "four_frame_result", "0", "txt", "images_parsed.txt"
    )

    if os.path.exists(image_parsed_txt):
        image_parsed = load_image_parsed(image_parsed_txt)
        if len(image_parsed) == 4:
            print(f"Generating KeyFrameTrajectory{prev_idx}{curr_idx}_new.txt via connecting_maps.py")

            # config 업데이트
            config['firstmap_trajectory_path'] = os.path.join(traj_base_path, f"KeyFrameTrajectory{prev_idx}.txt")
            config['nextmap_trajectory_path'] = os.path.join(traj_base_path, f"KeyFrameTrajectory{curr_idx}.txt")
            config['final_trajectory_path'] = new_txt_path
            config['firstmap_new_trajectory_path'] = firstmap_new_traj_path
            config['nextmap_new_trajectory_path'] = nextmap_new_traj_path
            config['txt_parsed_path'] = image_parsed_txt

            with open("config_long.yaml", "w") as fw: # config_long
                yaml.dump(config, fw)
            
            os.system("python3 connecting_short_maps.py")

    if os.path.exists(new_txt_path):
        print(f"Appending precomputed {prev_idx}->{curr_idx} from {new_txt_path}")

        new_traj = load_keyframe_trajectory(new_txt_path)
        prev_traj = load_keyframe_trajectory(out_path)

        first_time = new_traj[0]['timestamp']
        prev_traj = [pose for pose in prev_traj if pose['timestamp'] < first_time]

        if (i==1):
                with open(out_path, 'a') as f:
                    for pose in prev_traj:
                        line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} " \
                            f"{pose['quaternion'][0]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]}\n"
                        f.write(line)

        else: 
            merge_and_transform(prev_traj, new_traj, out_path)

    else:
        print(f"Transforming manually {prev_idx}->{curr_idx}")

        new_txt_path = os.path.join(traj_base_path, f"KeyFrameTrajectory{curr_idx}.txt")
        new_traj = load_keyframe_trajectory(new_txt_path)
        prev_traj = load_keyframe_trajectory(out_path)

        merge_and_transform(prev_traj, new_traj, out_path)


