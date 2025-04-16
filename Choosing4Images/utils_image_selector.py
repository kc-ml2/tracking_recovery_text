import pandas as pd
import yaml
from numpy import linspace

# config 파일 로드
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

timestamp_path = config["timestamp_path"]

# timestamp.txt 불러오기
def load_tracking_events(timestamp_path):
    print("Loading timestamp from:", timestamp_path)
    events = []
    with open(timestamp_path, "r") as f:
        for line in f:
            # print(f"[DEBUG] Line: [{line.strip()}]")
            parts = line.strip().split()
            # print(f"[DEBUG] Parts: {parts}")   
            if len(parts) == 2:
                events.append((float(parts[0]), float(parts[1])))
            elif len(parts) == 1:
                events.append((float(parts[0]), None))
    # print(f"[DEBUG] Final Events: {events}")   
    return events

# yolo_info_filtered.csv 불러오기
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = df["image_filename"].apply(lambda x: float(x.split(".")[0]))
    return df.sort_values("timestamp")

# 주어진 범위에서 timestamp 초당 per_sec개 추출
def sample_timestamps(df, start, end): 
    per_sec = config["hyperparameters"]["image_selector_frames_per_sec"]

    if start >= end:
        print("시작과 끝이 같거나 잘못됨")
        return []
    
    duration = end - start
    num_samples = int(duration * per_sec)
    if num_samples < 1:
        num_samples = 1

    # 1. 구간 내 timestamp 필터링
    df["timestamp"] = df["image_filename"].apply(lambda x: float(x.rsplit(".", 1)[0]))
    sub_df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    sub_df = sub_df.sort_values("timestamp")
    sub_df = sub_df.drop_duplicates("image_filename").sort_values("timestamp").reset_index(drop=True)

    if sub_df.empty:
        print("해당 구간에 이미지 없음")
        return []

    # 2. 고르게 분포된 index 추출
    if len(sub_df) <= num_samples:
        selected = sub_df["image_filename"].tolist()
    else:
        indices = linspace(0, len(sub_df) - 1, num_samples, dtype=int)
        selected = sub_df.iloc[indices]["image_filename"].tolist()

    return selected

# n번째 old map, n+1번째 new map에서 이미지 선택
def select_images(n, csv_path, wanted_timestamp_path, debug):
    max_interval = config["hyperparameters"]["image_selector_max_interval"]
    df = load_csv(csv_path)
    events = load_tracking_events(wanted_timestamp_path)

    selected_before = []
    selected_after = []

    # 현재 이벤트 존재 확인
    if n >= len(events):
        raise ValueError(f"n={n}은 이벤트 개수 {len(events)}보다 크거나 같음!")

    curr_fail, curr_relocal = events[n]

    # 이전 relocal -> 현재 fail
    if n > 0:
        prev_relocal = events[n - 1][1]
        if prev_relocal is not None:
            selected_before = sample_timestamps(df, max(prev_relocal, curr_fail - 3) , curr_fail)
    else:
        # events[0]이면: relocal이 없으므로 fail 전 3초 구간
        selected_before = sample_timestamps(df, curr_fail - max_interval, curr_fail)

    # 현재 relocal -> 다음 fail
    if curr_relocal is not None:
        if n < len(events) - 1:
            next_fail = events[n + 1][0]
            selected_after = sample_timestamps(df, curr_relocal, min(next_fail, curr_relocal + 3))
        else:
            # 마지막 이벤트면 relocal 이후 3초 구간
            selected_after = sample_timestamps(df, curr_relocal, curr_relocal + max_interval)

    if (debug==True):    
        print(f"\nOLD MAP SELECTED: {selected_before}")
        print(f"NEW MAP SELECTED: {selected_after}")

    return selected_before, selected_after
