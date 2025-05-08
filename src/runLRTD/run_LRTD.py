import csv
import os
import time
from ultralytics import YOLO

# Load our YOLO model trained for Location-Relevant Text Detection (LRTD)
our_model = YOLO("./LRTD.pt")

# Load key paths and confidence threshold
INPUT_ROOT = os.environ.get("INPUT_ROOT")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT")
CONFIDENCE = 0.5

# Run inference
results = our_model(
    source=f"{INPUT_ROOT}/images",
    save=True,
    show=False,
    project=OUTPUT_ROOT,
    name= "LRTD_images",
    stream=True,
    conf=CONFIDENCE
)

# Save LRTD results to CSV
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

# Run LRTD and save results
def run_yolo(results):
    for r in results:
        if not r.boxes:
            continue
        file_name = os.path.basename(r.path)
        for box in r.boxes.data:
            x1, y1, x2, y2, conf = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
            save_to_csv(f'{OUTPUT_ROOT}/LRTD_info.csv', file_name, x1, y1, x2, y2, conf)

# Perform full LRTD and saving pipeline with timing
def main():
    start = time.time()
    run_yolo(results)
    print(f"Execution time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()