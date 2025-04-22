import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils_connecting_maps import load_keyframe_trajectory, load_image_parsed

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

    # R_rel = qA 
    # t_rel = tA 

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

# def merge_and_transform(prev_traj, curr_traj, output_path):
#     """
#     prev_pose: dict, 기준 포즈
#     curr_traj: list of dict, 변환할 시퀀스
#     """
#     prev_pose = prev_traj[-1]
#     base_q = R.from_quat(prev_pose['quaternion'])
#     base_t = np.array(prev_pose['translation'])

#     transformed = []

#     for pose in curr_traj:
#         q = R.from_quat(pose['quaternion'])
#         t = np.array(pose['translation'])

#         t_new = base_q.apply(t) + base_t
#         q_new = (base_q * q).as_quat()

#         transformed.append({
#             'timestamp': pose['timestamp'],
#             'translation': t_new.tolist(),
#             'quaternion': q_new.tolist()
#         })

#     # 마지막 pose를 누적 기준으로 업데이트하기 위해 반환
#     final_pose = transformed[-1]

#     # 저장
#     with open(output_path, 'a') as f:
#         for pose in transformed:
#             line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} " \
#                    f"{pose['quaternion'][0]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]}\n"
#             f.write(line)

#     return final_pose

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

    new_txt_path = os.path.join(
        connecting_dir, date_str, time_str, "yolo", f"{curr_idx}th",
        "four_frame_result", "0", "txt", f"KeyFrameTrajectory{prev_idx}{curr_idx}_new.txt"
    )

    # check if we can generate new_txt_path
    image_parsed_txt = os.path.join(
        connecting_dir, date_str, time_str, "yolo", f"{curr_idx}th",
        "four_frame_result", "0", "txt", "images_parsed.txt"
    )

    # if os.path.exists(image_parsed_txt):
    #     image_parsed = load_image_parsed(image_parsed_txt)
    #     if len(image_parsed) == 4:
    #         print(f"Generating KeyFrameTrajectory{prev_idx}{curr_idx}_new.txt via connecting_maps.py")

    #         # config 업데이트
    #         config['firstmap_trajectory_path'] = os.path.join(traj_base_path, f"KeyFrameTrajectory{prev_idx}.txt")
    #         config['nextmap_trajectory_path'] = os.path.join(traj_base_path, f"KeyFrameTrajectory{curr_idx}.txt")
    #         config['firstmap_new_trajectory_path'] = config['firstmap_trajectory_path']
    #         config['nextmap_new_trajectory_path'] = config['nextmap_trajectory_path']
    #         config['final_trajectory_path'] = new_txt_path
    #         config['txt_parsed_path'] = image_parsed_txt

    #         with open("config.yaml", "w") as fw:
    #             yaml.dump(config, fw)

    #         os.system("python3 connecting_short_maps.py")

    # if os.path.exists(new_txt_path):
    #     print(f"Appending precomputed {prev_idx}->{curr_idx} from {new_txt_path}")
    #     new_traj = load_keyframe_trajectory(new_txt_path)

    #     new_start_time = new_traj[0]['timestamp']
    #     if os.path.exists(out_path):
    #         with open(out_path, 'r') as f:
    #             lines = f.readlines()
                    
    #         filtered_lines = [
    #             line for line in lines
    #             if float(line.split()[0]) < new_start_time
    #         ]

    #         with open(out_path, 'w') as f:
    #             f.writelines(filtered_lines)

    #     # if new_traj[0]['timestamp'] == prev_traj[-1]['timestamp']:
    #     #     new_traj = new_traj[1:]

    #     # with open(out_path, 'a') as f:
    #     #     for pose in new_traj:
    #     #         line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} " \
    #     #             f"{pose['quaternion'][0]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]}\n"
    #     #         f.write(line)

    #     # 현재까지 이어진 out_path의 마지막 포즈 기준으로 한 번 더 transform
    #     prev_traj = load_keyframe_trajectory(out_path)
    #     merge_and_transform(prev_traj, new_traj, out_path)

    # else:
    print(f"Transforming manually {prev_idx}->{curr_idx}")
    new_txt_path = os.path.join(traj_base_path, f"KeyFrameTrajectory{curr_idx}.txt")
    new_traj = load_keyframe_trajectory(new_txt_path)

    prev_traj = load_keyframe_trajectory(out_path)
    merge_and_transform(prev_traj, new_traj, out_path)


