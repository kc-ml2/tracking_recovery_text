import pandas as pd
import yaml

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

mode = config["mode"]
csv_path = f'{config["file_path"]}/{mode}/{"yolo_info.csv" if mode == "yolo" else "ocr_info.csv"}'
save_path = config[f"filtered_csv_{mode}_path"]

# load .csv file
def load_csv(path_to_csv):
    data = pd.read_csv(path_to_csv)
    return data

# filter rows of conf >= csv_conf_thresh
def filter_conf(data):
    return data[data["conf"] >= config["hyperparameters"]["csv_conf_thresh"]]

# save to .csv
def save_filtered_csv(filtered_data, path_to_output):
    filtered_data.to_csv(path_to_output, index=False)

df = load_csv(csv_path)
filtered_df = filter_conf(df)
save_filtered_csv(filtered_df, save_path)

print(f"Filtered data saved to {save_path}")
