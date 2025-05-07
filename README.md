# Recovory from Tracking Failure with Location-Relevant Text Detection for Indoor Visual SLAM

This repository contains the supplementary code for our CoRL 2025 submission.

---

## How to run

### Dependencies
```bash
pip install -r requirements.txt
```
### Download Sample Dataset
Due to size limits, sample data is hosted externally.
Make sure to create the 'data/' and 'results/' directory in this step.

```bash
mkdir data && cd data
wget https://github.com/kc-ml2/tracking_recovery_text/releases/download/v1.0.0/example_sequence.zip
unzip example_sequence.zip -d data/
cd ..
mkdir results
```

### Execute our program
The command below runs the full pipeline of our system.
```bash
bash run_all_pipeline.sh
```
This script will sequentially execute:

    runLRTD: Perform LRTD on all keyframes

    search4frames: Text guided frame search & local map generation

    alignmaps: Align two maps with local map

    evo_traj: Trajectory comparision between our method and ORB-SLAM

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
All outputs will be saved in:
```bash
results/your_sequence_name
```
Will contain:

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

## License
