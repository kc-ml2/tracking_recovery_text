import csv
import os
import time
from paddleocr import PaddleOCR
import utils
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang='en') 

INPUT_DIR = f"{utils.OUTPUT_ROOT}/images"
OUTPUT_DIR = f"{utils.OUTPUT_ROOT}/OCR/computed_images"
CSV_PATH = f"{utils.OUTPUT_ROOT}/OCR/ocr_info.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_to_csv(file_path, filename, x1, y1, x2, y2, conf):
    file_path = os.path.expanduser(file_path)
    file_exist = os.path.exists(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exist:
            header = ['image_filename', 'x1', 'y1', 'x2', 'y2', 'conf']
            writer.writerow(header)
        writer.writerow([filename, x1, y1, x2, y2, conf])

def run_ocr():
    for file_name in os.listdir(INPUT_DIR):
        if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(INPUT_DIR, file_name)
        image = cv2.imread(image_path)
        result = ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            continue

        for line in result[0]:
            box, (_, score) = line
            x1, y1 = box[0]
            x2, y2 = box[2]

            save_to_csv(CSV_PATH, file_name, x1, y1, x2, y2, score)

            # 시각화 저장
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(image, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

        save_path = os.path.join(OUTPUT_DIR, file_name)
        cv2.imwrite(save_path, image)

def main():
    start_time = time.time()
    run_ocr()
    end_time = time.time()
    print(f"총 실행 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    main()
