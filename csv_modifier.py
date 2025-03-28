import pandas as pd
import yaml

# YAML 파일 로드
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# CSV 파일 로드
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df


# conf 값이 0.5 이상인 행만 필터링
def filter_conf(df):
    return df[df["conf"] >= 0.5]


# 필터링된 데이터를 새로운 파일로 저장
def save_filtered_csv(df, output_path):
    df.to_csv(output_path, index=False)


# 파일 경로 설정
file_path = config["imgs_paths"]["img1"] + "/yolo/yolo_info.csv"
output_path = config["filtered_csv_path"]

# CSV 파일 로드
df = load_csv(file_path)

# conf 값이 0.5 이상인 데이터만 필터링
filtered_df = filter_conf(df)

# 필터링된 데이터를 새로운 파일로 저장
save_filtered_csv(filtered_df, output_path)

print(f"Filtered data saved to {output_path}")
