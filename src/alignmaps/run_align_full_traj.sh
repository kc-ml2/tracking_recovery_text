#!/bin/bash

source ../../env.sh

echo "[INFO] Running connecting_long_maps.py for selected sequences..."

# Iterate over each dataset
for ID in "${DATA_LIST[@]}"
do
    echo "▶ Processing dataset $ID"

    TRAJ_DIR="${DATA_DIR}/${ID}/orb_result"

    # Extract keyframe trajactory files
    SEQ_LIST=$(find $TRAJ_DIR -maxdepth 1 -type f -name 'KeyFrameTrajectory[0-9][0-9].txt' \
                -exec basename {} \; | sed -n 's/KeyFrameTrajectory\([0-9][0-9]\)\.txt/\1/p' | sort)

    # Path for saving connected map
    CONNECTED_PATH="${RESULTS_DIR}/${ID}/ORB-SLAM_with_LRTD.txt"

    # Convert the sequence list of maps to a YAML format
    YAML_SEQ=""
    for SEQ in $SEQ_LIST; do
        YAML_SEQ+="  - '$SEQ'\n"
    done

 # Update the config file with current file paths
cat > "$ALIGN_CONFIG" <<EOF
connecting_dir: ${CONNECTING_DIR}
final_trajectory_path: ''
firstmap_new_trajectory_path: ''
firstmap_trajectory_path: ''
nextmap_new_trajectory_path: ''
nextmap_trajectory_path: ''
txt_input_path: ''
txt_parsed_path: ''

root_dir: "$DATA_DIR/$ID"
root_result_dir: "$RESULTS_DIR/$ID"
out_path: "$CONNECTED_PATH"
sequences:
$(echo -e "$YAML_SEQ")
EOF

    python3 align_full_traj.py
        {
        echo "$DATA - $MODE"
        echo "Total tracking fail: $total"
        echo "Filtered out (low score or insufficient): $filtered"
        echo "Colmap sparse model failures: $colmap_fail"
        echo ""
    } >> "$RESULTS_DIR/$ID/log_tracking_fail.txt"
done
