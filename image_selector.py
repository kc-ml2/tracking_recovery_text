import pandas as pd
import yaml
from numpy import linspace

# config 파일 로드
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

timestamp_path = config["timestamp_path"]
csv_path = config["filtered_csv_path"]
#csv_path = config["file_paths"]["file2"] + "/yolo/yolo_info.csv"

# timestamp.txt 불러오기
def load_tracking_events(timestamp_path):
    events = []
    with open(timestamp_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                events.append((float(parts[0]), float(parts[1])))
            elif len(parts) == 1:
                events.append((float(parts[0]), None))
    return events

# yolo_info_filtered.csv 불러오기
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = df["image_filename"].apply(lambda x: float(x.split(".")[0]))
    return df.sort_values("timestamp")

# 주어진 범위에서 timestamp 초당 n개 추출
def sample_timestamps(df, start, end, per_sec=10): # 3개 추출
    print(f"\n샘플링 구간: {start:.6f} → {end:.6f}")
    if start >= end:
        print("⚠️ 시작과 끝이 같거나 잘못됨")
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

    print(f"🎯 선택된 {len(selected)}개:", selected)
    return selected

# 메인 함수
def select_timestamps_around_n(n):
    df = load_csv(csv_path)
    events = load_tracking_events(timestamp_path)

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
            selected_before = sample_timestamps(df, prev_relocal, curr_fail)
    else:
        # events[0]이면: relocal이 없으므로 fail 전 2초 구간
        selected_before = sample_timestamps(df, curr_fail - 5, curr_fail)

    # 현재 relocal -> 다음 fail
    if curr_relocal is not None:
        if n < len(events) - 1:
            next_fail = events[n + 1][0]
            selected_after = sample_timestamps(df, curr_relocal, next_fail)
        else:
            # 마지막 이벤트면 relocal 이후 2초 구간
            selected_after = sample_timestamps(df, curr_relocal, curr_relocal + 10)
        
    print(f"\nOLD MAP SELECTED: {selected_before}")
    print(f"NEW MAP SELECTED: {selected_after}")

    return selected_before, selected_after

# 예시 실행
n = 1
select_timestamps_around_n(n)

# 디버깅 코드
events = load_tracking_events(timestamp_path)
print(f"\n전체 이벤트 리스트 (총 {len(events)}개):")
for i, (fail, relocal) in enumerate(events):
    print(f"  {i}: fail = {fail}, relocal = {relocal}")

df = load_csv(csv_path)
print(f"\nCSV timestamp 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

