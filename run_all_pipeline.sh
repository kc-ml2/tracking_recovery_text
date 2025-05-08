#!/bin/bash

set -e

source ./env.sh

echo "Step 1: Run LRTD"
cd ./src/runLRTD/
bash run_LRTD_all.sh

echo "Step 2: Running COLMAP for selected frames"
cd ../search4frames/
bash run_colmap_all.sh

echo "Step 3: Connecting short maps"
cd ../alignmaps/
bash run_align_2_maps_all.sh

echo "Step 4: Connecting long maps"
bash run_align_full_traj.sh

echo "Step 5: Evaluating trajectory with evo_traj"
for ID in "${DATA_LIST[@]}"
do
    OURS_PATH="$RESULTS_DIR/$ID/ORB-SLAM_with_LRTD.txt"
    ORB_PATH="$DATA_DIR/$ID/ORB-SLAM.txt"
    GROUND_TRUTH_PATH="$DATA_DIR/$ID/Ground_Truth.txt"

    evo_traj tum "$OURS_PATH" "$ORB_PATH" \
        --ref="$GROUND_TRUTH_PATH" \
        --align --correct_scale -p --plot_mode xy
done

echo "All steps completed."