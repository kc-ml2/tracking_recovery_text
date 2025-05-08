import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils_aligning_maps import load_keyframe_trajectory, load_image_parsed

# Load config file
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load key paths
root_dir = config["root_dir"]
root_result_dir = config["root_result_dir"]
sequences = config["sequences"]
connecting_dir = config["connecting_dir"]
traj_path = os.path.join(root_dir,"orb_result") 

out_path = config["out_path"]

# Allign 2 maps, assuming the last pose of oldmap as the starting pose of newmap
def merge_and_transform(prev_traj, curr_traj, output_path):
    tA = np.array(prev_traj[-1]['translation'])
    qA = R.from_quat(prev_traj[-1]['quaternion'])

    R_rel = qA 
    t_rel = tA 

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

    with open(output_path, 'a') as f:
        for pose in transformed:
            line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} " \
                   f"{pose['quaternion'][0]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]}\n"
            f.write(line)

# Save trajectory of the first oldmap
first_path = os.path.join(traj_path, f"KeyFrameTrajectory{sequences[0]}.txt")
first_traj = load_keyframe_trajectory(first_path)

open(out_path, 'w').close()
with open(out_path, 'w') as f:
    for pose in first_traj:
        line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} " \
               f"{pose['quaternion'][0]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]}\n"
        f.write(line)

# Iterate for all newly made maps
for i in range(1, len(sequences)):
    # Assign oldmap and newmap
    prev_idx = sequences[i - 1]
    curr_idx = sequences[i]
    int_curr_idx = int(curr_idx)

    # Assign paths of connected map, scale corrected newmap, scale corrected oldmap, poses of 4 frames
    new_merged_path = os.path.join(
        root_result_dir, "COLMAP", f"{int_curr_idx}th",
        "four_frame_result", "0", "txt", f"KeyFrameTrajectory{prev_idx}{curr_idx}_new.txt"
    )

    firstmap_new_traj_path = os.path.join(
        root_result_dir, "COLMAP", f"{int_curr_idx}th",
        "four_frame_result", "0", "txt", f"KeyFrameTrajectory{prev_idx}_new.txt"
    )


    nextmap_new_traj_path = os.path.join(
        root_result_dir, "COLMAP", f"{int_curr_idx}th",
        "four_frame_result", "0", "txt", f"KeyFrameTrajectory{curr_idx}_new.txt"
    )

    image_parsed_txt = os.path.join(
        root_result_dir, "COLMAP", f"{int_curr_idx}th",
        "four_frame_result", "0", "txt", "images_parsed.txt"
    )

    if os.path.exists(image_parsed_txt):
        image_parsed = load_image_parsed(image_parsed_txt)
        if len(image_parsed) == 4:
            print(f"Generating KeyFrameTrajectory{prev_idx}{curr_idx}_new.txt via connecting_short_maps.py")

            # Update assigned paths to config file
            config['firstmap_trajectory_path'] = os.path.join(traj_path, f"KeyFrameTrajectory{prev_idx}.txt")
            config['nextmap_trajectory_path'] = os.path.join(traj_path, f"KeyFrameTrajectory{curr_idx}.txt")
            config['final_trajectory_path'] = new_merged_path
            config['firstmap_new_trajectory_path'] = firstmap_new_traj_path
            config['nextmap_new_trajectory_path'] = nextmap_new_traj_path
            config['txt_parsed_path'] = image_parsed_txt

            with open("config.yaml", "w") as fw:
                yaml.dump(config, fw)
            
            # Connect maps for each fail
            os.system("python3 connecting_short_maps.py")

    # If connection succeed, allign with scale/rotation correction
    if os.path.exists(new_merged_path):
        print(f"Appending precomputed {prev_idx}->{curr_idx} from {new_merged_path}")

        new_traj = load_keyframe_trajectory(new_merged_path)
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

    # If connection failed, allign without scale/rotation correction
    else:
        print(f"Transforming manually {prev_idx}->{curr_idx}")

        new_merged_path = os.path.join(traj_path, f"KeyFrameTrajectory{curr_idx}.txt")
        new_traj = load_keyframe_trajectory(new_merged_path)
        prev_traj = load_keyframe_trajectory(out_path)

        merge_and_transform(prev_traj, new_traj, out_path)


