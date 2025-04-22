import numpy as np

def load_traj_txt(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            data.append([float(x) for x in parts])
    return np.array(data)

a = load_traj_txt("/home/youngsun/vslam/corl/ConnectingMaps/connected_long.txt")
b = load_traj_txt("/mnt/sda/coex_data/long_sequence/result_2025_04_14_092728/mono_result/trajectory_merged.txt")

for i in range(min(len(a), len(b))):
    t1 = a[i][1:4]
    t2 = b[i][1:4]
    err = np.linalg.norm(np.array(t1) - np.array(t2))
    if err > 1e-6:
        print(f"[{i}] t-diff: {err:.6f} | t1: {t1} | t2: {t2}")