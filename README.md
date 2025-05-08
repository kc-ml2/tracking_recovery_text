# Recovory from Tracking Failure with Location-Relevant Text Detection for Indoor Visual SLAM

This repository contains the supplementary code for our CoRL 2025 submission.

---

## How to run

### 1. Install Dependencies
This project requires Python 3.10+
```bash
pip install -r requirements.txt
```
### 2. Download Sample Dataset
Due to size limits, sample data is hosted externally. 

Make sure to create the 'data/' and 'results/' directory in this step.

```bash
mkdir data && cd data
gdown https://drive.google.com/uc?id=1tZsYiypBhw_9EdzqTGKThjxZBzSjsgU7
unzip example_sequence.zip 
cd .. && mkdir results
```

### 3. Set the Working Directory
In 'env.sh', set the path below to the absolute path of your code directory.
```bash
RUN_DIR="absolute/path/to/your/code"
```

### 4. Run Full Pipeline
The command below runs the full pipeline of our system.
```bash
bash run_all_pipeline.sh
```
Will sequentially execute:

- src/runLRTD - Perform LRTD on all keyframes

- src/search4frames - Text guided frame search & Local map generation

- src/alignmaps - Align two maps with local map

- evo_traj - Visualize trajectory comparision between our method and ORB-SLAM

## Input format
All inputs should be stored in:
```bash
data/your_sequence_name
```
Should contain:

- images/ - RGB images of keyframes

- orb_result/KeyframeTrakectoryXX.txt - Trajectories of built maps

- orb_result/timestamp.txt - Timestamps of relocalization & tracking fail

- Ground_Truth.txt - Ground truth trajectory 

- ORB-SLAM.txt - Aligned trajectory without LRTD
 
## Output format
All outputs will be stored in:
```bash
results/your_sequence_name
```
Should contain:

- COLMAP/

- LRTD_images/ 

- log_4images.txt 

- log_colmap.txt 

- log_tracking_fail.txt 

- LRTD_filtered_info.csv 

- LRTD_info.csv 

- ORB-SLAM_with_LRTD.txt - Aligned trajectory with LRTD

## Results
Visualized results will appear like below:

![Result](https://drive.google.com/uc?id=1boEXYdB3eDExO_tB_qSH1SMYy3A3-bq0)

## Configuration
You can configure:

- Frame search hyperparameters (in 'src/search4frames/config.yaml')

- Data and result paths (in 'env.sh')
