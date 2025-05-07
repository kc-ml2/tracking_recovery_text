import pandas as pd
import yaml
from numpy import linspace

# Load config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

timestamp_path = config["timestamp_path"]

# Load timestamp.txt as list of (fail, relocalization) tuples
def load_tracking_events(timestamp_path):
    print("\nLoading timestamp from:", timestamp_path)
    events = []
    with open(timestamp_path, "r") as f:
        for line in f:
            parts = line.strip().split()   
            if len(parts) == 2:
                events.append((float(parts[0]), float(parts[1])))
            elif len(parts) == 1:
                events.append((float(parts[0]), None)) 
    return events

# Load filtered YOLO CSV and sort by timestamp
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = df["image_filename"].apply(lambda x: float(x.split(".")[0]))
    return df.sort_values("timestamp")

# Sample frames at per_sec fps between two timestamps
def sample_timestamps(df, start, end): 
    per_sec = config["hyperparameters"]["image_selector_frames_per_sec"]

    if start >= end:
        return []
    
    duration = end - start
    num_samples = int(duration * per_sec)
    if num_samples < 1:
        num_samples = 1

    # Filter timestamps in the given interval
    df["timestamp"] = df["image_filename"].apply(lambda x: float(x.rsplit(".", 1)[0]))
    sub_df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    sub_df = sub_df.sort_values("timestamp")
    sub_df = sub_df.drop_duplicates("image_filename").sort_values("timestamp").reset_index(drop=True)

    if sub_df.empty:
        return []

    # Uniform sampling
    if len(sub_df) <= num_samples:
        selected = sub_df["image_filename"].tolist()
    else:
        indices = linspace(0, len(sub_df) - 1, num_samples, dtype=int)
        selected = sub_df.iloc[indices]["image_filename"].tolist()

    return selected

# Sample frames from old (n) and new (n+1) maps
def select_images(n, csv_path, wanted_timestamp_path, debug):
    max_interval = config["hyperparameters"]["image_selector_max_interval"]
    df = load_csv(csv_path)
    events = load_tracking_events(wanted_timestamp_path)

    selected_before = []
    selected_after = []

    if n >= len(events):
        raise ValueError(f"n={n} exceeds number of events ({len(events)})")

    curr_fail, curr_relocal = events[n]

    # Before current fail
    if n > 0:
        prev_relocal = events[n - 1][1]
        if prev_relocal is not None:
            selected_before = sample_timestamps(df, max(prev_relocal, curr_fail - 3) , curr_fail)
    else:
        selected_before = sample_timestamps(df, curr_fail - max_interval, curr_fail)

    # After current relocalization
    if curr_relocal is not None:
        if n < len(events) - 1:
            next_fail = events[n + 1][0]
            selected_after = sample_timestamps(df, curr_relocal, min(next_fail, curr_relocal + 3))
        else:
            selected_after = sample_timestamps(df, curr_relocal, curr_relocal + max_interval)

    if (debug==True):    
        print(f"\nOLD MAP SELECTED: {selected_before}")
        print(f"NEW MAP SELECTED: {selected_after}")

    return selected_before, selected_after
