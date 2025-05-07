import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# Load poses of selected 4 frames from COLMAP-made local map TXT file
def load_image_parsed(file_path):
    data = {}
    with open(file_path, 'r') as file:
        next(file) 
        for line in file:
            parts = line.strip().split()
            image_id = int(parts[0])
            qx, qy, qz, qw = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            timestamp = parts[9]
            data[image_id] = {"timestamp": timestamp, 'quaternion': [qx, qy, qz, qw], 'translation': [tx, ty, tz]}

    return data

# Load poses of oldmap/newmap from input keyframe trajectory TXT file
def load_keyframe_trajectory(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            timestamp = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8]) 
            data.append({'timestamp': timestamp, 'translation': [tx, ty, tz], 'quaternion': [qx, qy, qz, qw]})

    return data

# Interpolate pose(rotation and translation) at a given timestamp from a given trajectory
def interpolate_pose(timestamp, trajectory):
    times = np.array([pose['timestamp'] for pose in trajectory])
    if timestamp in times:
        pose = next(p for p in trajectory if p['timestamp'] == timestamp)

        return np.array(pose['quaternion']), np.array(pose['translation'])
    
    else:
        idx = np.searchsorted(times, timestamp)
        if idx == 0 or idx == len(times):
            raise ValueError("Timestamp out of range for interpolation.")
        t1, t2 = times[idx - 1], times[idx]
        pose1, pose2 = trajectory[idx - 1], trajectory[idx]
        alpha = (timestamp - t1) / (t2 - t1)

        trans1, trans2 = np.array(pose1['translation']), np.array(pose2['translation'])
        interp_trans = (1 - alpha) * trans1 + alpha * trans2
        slerp = Slerp([0, 1], R.from_quat([pose1['quaternion'], pose2['quaternion']]))
        interp_quat = slerp([alpha]).as_quat()[0]

        return np.array(interp_quat.tolist()), np.array(interp_trans.tolist())

# Compute relative rotation and translation from pose1 to pose2
def compute_relative_transformation(pose1, pose2):
    trans1, quat1 = np.array(pose1['translation']), R.from_quat(pose1['quaternion'])
    trans2, quat2 = np.array(pose2['translation']), R.from_quat(pose2['quaternion'])
    relative_rotation = quat2 * quat1.inv()
    relative_translation = trans2 - trans1

    return np.array(relative_rotation.as_quat().tolist()), np.array(relative_translation.tolist())

# Save scaled trajectory
def scale_calibrated_keyframe_trajectory(firstmap_trajectory, scale, output_file_path):
    scaled_firstmap_trajectory = []
    for pose in firstmap_trajectory:
        scaled_translation = [coord * scale for coord in pose['translation']]
        scaled_firstmap_trajectory.append({
            'timestamp': pose['timestamp'],
            'translation': scaled_translation,
            'quaternion': pose['quaternion']
        })

    with open(output_file_path, 'w') as file:
        for pose in scaled_firstmap_trajectory:
            line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} {pose['quaternion'][0]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]}\n"
            file.write(line)
    
    return True

# Save transformed trajectory
def transform_and_save_nextmap_trajectory(nextmap_trajectory, R1, t, input_path, output_path):
    with open(input_path, 'r') as file:
        original_lines = file.readlines()

    transformed_trajectory = []
    R1 = R.from_matrix(R1)

    for pose in nextmap_trajectory:
        t_orig = np.array(pose['translation'])
        q_orig = np.array(pose['quaternion'])

        q_rotated = (R1 * R.from_quat(q_orig)).as_quat()
        t_transformed = R1.apply(t_orig) + t

        line = f"{pose['timestamp']} " + " ".join([
            f"{v:.7f}" for v in [
                t_transformed[0], t_transformed[1], t_transformed[2],
                q_rotated[0], q_rotated[1], q_rotated[2], q_rotated[3]
            ]
        ]) + "\n"
        transformed_trajectory.append(line)

    with open(output_path, 'w') as file:
        file.writelines(original_lines)
        file.writelines(transformed_trajectory)

    return True