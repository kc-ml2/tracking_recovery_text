import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def load_image_parsed(file_path):
    data = {}
    with open(file_path, 'r') as file:
        next(file) 
        for line in file:
            parts = line.strip().split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            timestamp = parts[9]
            data[image_id] = {"timestamp": timestamp, 'quaternion': [qw, qx, qy, qz], 'translation': [tx, ty, tz]}
    return data

def load_keyframe_trajectory(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            timestamp = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8]) 
            data.append({'timestamp': timestamp, 'translation': [tx, ty, tz], 'quaternion': [qw, qx, qy, qz]})
    return data

def interpolate_pose(timestamp, trajectory):
    times = np.array([pose['timestamp'] for pose in trajectory])
    if timestamp in times:
        pose = next(p for p in trajectory if p['timestamp'] == timestamp)

        # print(" ") # interpolarization debug
        # print(pose)
        # print(pose['translation'])
        # print(pose['quaternion'])

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

        # print(" ") # interpolarization debug
        # print(pose1)
        # print(pose2)
        # print(interp_quat.tolist())
        # print(interp_trans.tolist())

        return np.array(interp_quat.tolist()), np.array(interp_trans.tolist())
    
def compute_relative_transformation(pose1, pose2):
    trans1, quat1 = np.array(pose1['translation']), R.from_quat(pose1['quaternion'])
    trans2, quat2 = np.array(pose2['translation']), R.from_quat(pose2['quaternion'])
    relative_rotation = quat2 * quat1.inv()
    relative_translation = trans2 - trans1

    # print(" ") # debug
    # print(relative_rotation.as_quat().tolist())
    # print(relative_translation.tolist())

    return np.array(relative_rotation.as_quat().tolist()), np.array(relative_translation.tolist())

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
            line = f"{pose['timestamp']} {pose['translation'][0]} {pose['translation'][1]} {pose['translation'][2]} {pose['quaternion'][1]} {pose['quaternion'][2]} {pose['quaternion'][3]} {pose['quaternion'][0]}\n"
            file.write(line)
    
    return True

def transform_and_save_nextmap_trajectory(nextmap_trajectory, R1, t, input_path, output_path):
    with open(input_path, 'r') as file:
        original_lines = file.readlines()

    transformed_trajectory = []
    R1 = R.from_matrix(R1)

    for pose in nextmap_trajectory:
        t_orig = np.array(pose['translation'])
        q_orig = np.array(pose['quaternion'])

        q_rotated = (R1 * R.from_quat(q_orig)).as_quat()
        # t_transformed = R1.append(t_orig) + t
        t_transformed = t_orig + t

        line = f"{pose['timestamp']} " + " ".join([
            f"{v:.7f}" for v in [
                t_transformed[0], t_transformed[1], t_transformed[2],
                q_rotated[1], q_rotated[2], q_rotated[3], q_rotated[0]
            ]
        ]) + "\n"
        transformed_trajectory.append(line)

    with open(output_path, 'w') as file:
        file.writelines(original_lines)
        file.writelines(transformed_trajectory)

    return True