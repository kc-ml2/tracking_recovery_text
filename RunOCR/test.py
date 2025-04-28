import csv
import os
import time
from paddleocr import PaddleOCR
import utils
import cv2

# Initialize OCR engine (with angle classification, English)
ocr = PaddleOCR(use_angle_cls=True, lang='en') 

# Paths
INPUT_DIR = f"{utils.OUTPUT_ROOT}/images"
OUTPUT_DIR = f"{utils.OUTPUT_ROOT}/ocr/computed_images"
CSV_PATH = f"{utils.OUTPUT_ROOT}/ocr/ocr_info.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Append detection results to CSV
def save_to_csv(file_path, filename, x1, y1, x2, y2, conf):
    file_path = os.path.expanduser(file_path)
    file_exist = os.path.exists(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exist:
            header = ['image_filename', 'x1', 'y1', 'x2', 'y2', 'conf']
            writer.writerow(header)
        writer.writerow([filename, x1, y1, x2, y2, conf])

# Run OCR over all images and save results
def run_ocr():
    for file_name in os.listdir(INPUT_DIR):
        if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(INPUT_DIR, file_name)
        image = cv2.imread(image_path)
        result = ocr.ocr(image_path, cls=True)
    
        if result and result[0]:
            for line in result[0]:
                box, (_, score) = line
                x1, y1 = box[0]
                x2, y2 = box[2]
                save_to_csv(CSV_PATH, file_name, x1, y1, x2, y2, score)

                cv2.rectangle(
                    image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0), 8
                )

        cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), image)

# Entry point with timing
def main():
    start = time.time()
    run_ocr()
    print(f"Execution time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
