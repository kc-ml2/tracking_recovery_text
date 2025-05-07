import pandas as pd
import yaml

# Load config file
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Load key paths
mode = config["mode"]
csv_path = config[f"csv_{mode}_path"]
csv_filtered_path = config[f"filtered_csv_{mode}_path"]

# Load LRTD information (bounding box position, confidence score) of selected keyframes
def load_csv(path_to_csv):
    data = pd.read_csv(path_to_csv)
    return data

# Filter keyframes with sufficient confidence score
def filter_conf(data):
    return data[data["conf"] >= config["hyperparameters"]["csv_conf_thresh"]]

# Save filtered keyframes
def save_filtered_csv(filtered_data, path_to_output):
    filtered_data.to_csv(path_to_output, index=False)

df = load_csv(csv_path)
filtered_df = filter_conf(df)
save_filtered_csv(filtered_df, csv_filtered_path)

print(f"Filtered data saved to {csv_filtered_path}")
