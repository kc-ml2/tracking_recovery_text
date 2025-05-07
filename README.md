# Supplementary Code for CoRL 2025

**"Recovory from Tracking Failure with Location-Relevant Text Detection for Indoor Visual SLAM"**  
This repository contains the supplementary code for our CoRL 2025 submission.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Download Sample Dataset
Due to size limits, sample data is hosted externally.

```bash
cd ./data
wget https://github.com/kc-ml2/tracking_recovery_text/releases/download/v1.0.0/example_sequence.zip
unzip example_sequence.zip -d data/
```

### 3. Run the Full Pipeline
```bash
bash run_all_pipeline.sh
```
This script will sequentially execute:

    runLRTD: Perform LRTD 

    search4frames: Perform text guided frame search & local map generation

    alignmaps: Align two maps with local map

## Input format
All inputs should be stored in:
```bash
data/your_sequence_name
```
Should contain:

    images/: RGB images of the keyframes extracted by ORB-SLAM pipeline

    orb_result/KeyframeTrakectoryXX.txt: 

    orb_result/timestamp.txt: relocalization & tracking fail timestamps 

    Ground_Truth.txt: 

    ORB-SLAM.txt: 

## Output format
All outputs are saved in:
```bash
results/your_sequence_name
```
Should produce:

    COLMAP/:

    LRTD_images/: 

    log_4images.txt

    log_colmap.txt

    log_tracking_fail.txt:

    LRTD_filtered_info.csv:

    LRTD_info.csv:

    ORB-SLAM_with_LRTD.txt: 

## Configuration
Each module has its own config.yaml file:

    src/search4frames/config.yaml

    src/alignmaps/config.yaml

You can change:

    Dataset paths

    Thresholds

    Output filenames
