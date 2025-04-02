import pandas as pd
import yaml

# YAML 파일 로드
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# CSV 파일 로드
def load_csv(path_to_csv):
    data = pd.read_csv(path_to_csv)
    return data


# conf 값이 0.5 이상인 행만 필터링
def filter_conf(data):
    return data[data["conf"] >= config["hyperparameters"]["conf_threshold"]]


# 필터링된 데이터를 새로운 파일로 저장
def save_filtered_csv(filtered_data, path_to_output):
    filtered_data.to_csv(path_to_output, index=False)


# 파일 경로 설정
csv_path = config["file_path"] + "/yolo/yolo_info.csv"
save_path = config["filtered_csv_path"]

# CSV 파일 로드
df = load_csv(csv_path)

# conf 값이 0.5 이상인 데이터만 필터링
filtered_df = filter_conf(df)

# 필터링된 데이터를 새로운 파일로 저장
save_filtered_csv(filtered_df, save_path)

print(f"Filtered data saved to {save_path}")
