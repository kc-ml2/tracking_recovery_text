import pandas as pd
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

timestamp_path = config["timestamp_path"]


# 첫 new map이 시작되는 시점의 이미지 불러오기
def load_newmap_first_image():
    with open(timestamp_path, "r") as file:  # timestamp.txt 파일 불러오기
        lines = file.readlines()

    newmap_images = []  # new map의 시작 프레임 저장하는 배열
    for line in lines:
        parts = line.split()
        if len(parts) > 1:
            timestamp_2nd_column = parts[1]  # 2열
            image_filename = timestamp_2nd_column + ".png"
            newmap_images.append(image_filename)

    if len(newmap_images) > 0:
        first_newmap_image = newmap_images[0]
        print(first_newmap_image)
        return first_newmap_image
    else:
        raise ValueError("timestamp.txt에서 new map 이미지 정보를 찾을 수 없음")


# 첫 old map이 끝나는 시점의 이미지 불러오기
def load_oldmap_last_image():
    with open(timestamp_path, "r") as file:  # timestamp.txt 파일 불러오기
        lines = file.readlines()

    oldmap_images = []
    for line in lines:
        parts = line.split()
        if len(parts) > 1:
            timestamp_2nd_column = parts[0]  # 1열
            image_filename = timestamp_2nd_column + ".png"
            oldmap_images.append(image_filename)

    if len(oldmap_images) > 0:
        last_oldmap_image = oldmap_images[0]
        print(last_oldmap_image)
        return last_oldmap_image
    else:
        raise ValueError("timestamp.txt에서 new map 이미지 정보를 찾을 수 없음")


# csv 파일 로드
def load_csv(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)  # 첫 번째 행이 컬럼명이 아니면 이 줄 수정
    df.columns = ["image_filename", "x1", "y1", "x2", "y2", "conf"]
    return df


# new map의 시작 시점부터 동일한 초 단위에서 2개의 이미지를 균등하게 선택
def select_newmap_images(df, first_newmap_image):
    # 컬럼에 문자열이 아닌 값이 들어있는지 확인
    df = df[df["image_filename"].str.endswith(".png")]  # 확장자로 필터링

    # timestamp 변환
    df["timestamp"] = df["image_filename"].apply(lambda x: float(x.split(".")[0]))

    # new map 시작 시점을 기준으로 필터링
    new_map_start_timestamp = float(first_newmap_image.rsplit(".", 1)[0])
    df = df[
        (df["timestamp"] >= new_map_start_timestamp) & (df["timestamp"] <= new_map_start_timestamp + 10)
    ]  # 시작시간 이후 10초 이내의 데이터만 필터링

    # timestamp 기준으로 그룹화
    grouped = df.drop_duplicates("image_filename").groupby(df["timestamp"].astype(int))

    selected_newmap_images = []
    for _, group in grouped:
        group = group.sort_values("timestamp")
        num_images = len(group)

        if num_images >= 2:
            step = num_images / 2
            indices = [round(i * step) for i in range(2)]
        else:
            indices = list(range(num_images))  # 2개 이하라면 모두 선택

        selected_newmap_images.extend(group.iloc[indices]["image_filename"].tolist())

    return selected_newmap_images


# old map의 끝나는 시점부터 10초 전까지 동일한 초 단위에서 2개의 이미지를 균등하게 선택
def select_oldmap_images(df, last_oldmap_image):
    # 컬럼에 문자열이 아닌 값이 들어있는지 확인
    df = df[df["image_filename"].str.endswith(".png")]  # 확장자로 필터링

    # timestamp 변환
    df["timestamp"] = df["image_filename"].apply(lambda x: float(x.split(".")[0]))

    # old map 끝나는 시점을 기준으로 필터링
    old_map_end_timestamp = float(last_oldmap_image.rsplit(".", 1)[0])
    df = df[
        (df["timestamp"] <= old_map_end_timestamp) & (df["timestamp"] >= old_map_end_timestamp - 10)
    ]  # old map 끝나는 10초 전까지만 필터링

    # timestamp 기준으로 그룹화
    grouped = df.drop_duplicates("image_filename").groupby(df["timestamp"].astype(int))

    selected_oldmap_images = []
    for _, group in grouped:
        group = group.sort_values("timestamp")
        num_images = len(group)

        if num_images >= 2:
            step = num_images / 2
            indices = [round(i * step) for i in range(2)]
        else:
            indices = list(range(num_images))  # 2개 이하라면 모두 선택

        selected_oldmap_images.extend(group.iloc[indices]["image_filename"].tolist())

    return selected_oldmap_images


# CSV 파일 경로 설정
file_path = config["filtered_csv_path"]
df = load_csv(file_path)

# 이미지 선택
first_newmap_image = load_newmap_first_image()
selected_newmap_images = select_newmap_images(df, first_newmap_image)

last_oldmap_image = load_oldmap_last_image()
selected_oldmap_images = select_oldmap_images(df, last_oldmap_image)

print("Selected Images:", selected_oldmap_images)
print("Selected Images:", selected_newmap_images)
