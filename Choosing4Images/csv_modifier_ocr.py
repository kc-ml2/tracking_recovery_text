import pandas as pd
import yaml

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

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

# 1. load paths
csv_path = config["file_path"] + "/OCR/ocr_info.csv"
save_path = config["filtered_csv_ocr_path"]

# 2. load .csv file
df = load_csv(csv_path)

# 3. filtering
filtered_df = filter_conf(df)

# 4. save to .csv
save_filtered_csv(filtered_df, save_path)
print(f"Filtered data saved to {save_path}")
