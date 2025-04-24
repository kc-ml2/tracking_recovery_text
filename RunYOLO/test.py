import csv
import os
import time
from ultralytics import YOLO
import utils

# Load our YOLO model
our_model = YOLO("./best.pt")

# Run inference
results = our_model(
    source=f"{utils.OUTPUT_ROOT}/images",
    save=True,
    show=False,
    project=f"{utils.OUTPUT_ROOT}/yolo",
    stream=True,
    conf=utils.CONFIDENCE
)

# Append detection results to CSV
def save_to_csv(file_path, filename, x1, y1, x2, y2, conf):
    try:
        file_path = os.path.expanduser(file_path)
        file_exist = os.path.exists(file_path)
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exist:
                header = ['image_filename', 'x1', 'y1', 'x2', 'y2', 'conf']
                writer.writerow(header)
            writer.writerow([filename, x1, y1, x2, y2, conf])
    except Exception as e:
        print(f"CSV write error: {e}")

# Extract YOLO predictions and append to CSV
def run_yolo(results):
    for r in results:
        if not r.boxes:
            continue
        file_name = os.path.basename(r.path)
        for box in r.boxes.data:
            x1, y1, x2, y2, conf = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
            save_to_csv(f'{utils.OUTPUT_ROOT}/yolo/yolo_info.csv', file_name, x1, y1, x2, y2, conf)

# Entry point with timing
def main():
    start = time.time()
    run_yolo(results)
    print(f"Execution time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()