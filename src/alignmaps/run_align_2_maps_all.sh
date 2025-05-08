#!/bin/bash

source ../../env.sh

echo "[INFO] Running connecting_short_maps.py for selected sequences..."

# Iterate over each dataset
for ID in "${DATA_LIST[@]}"
do
    echo "▶ Processing dataset $ID"

    TRAJ_DIR="${DATA_DIR}/${ID}/orb_result"

    echo ""

    # Iterate over each local map
    for FAIL_DIR in "$RESULTS_DIR/$ID"/COLMAP/*th/; do

        # Procced only if COLMAP reconstruction was successful
        IDX_NAME=$(basename "$FAIL_DIR")      
        IDX=$(echo "$IDX_NAME" | grep -oP '\d+')

        if [[ -z "$IDX" ]]; then
            echo "Processing $IDX_NAME" 
            echo "[ERROR] No Colmap!"
            continue
        fi

        # Set oldmap and newmap
        PREV=$(printf "%02d" $((IDX - 1)))
        CURR=$(printf "%02d" $((IDX)))

        echo "Processing $IDX_NAME"

        TXT_DIR="$FAIL_DIR/four_frame_result/0/txt"
        mkdir -p "$TXT_DIR"

        # Define input and output trajectory file paths
        FIRSTMAP_TRAJ="$TRAJ_DIR/KeyFrameTrajectory${PREV}.txt" 
        NEXTMAP_TRAJ="$TRAJ_DIR/KeyFrameTrajectory${CURR}.txt"

        FIRSTMAP_NEW="$TXT_DIR/KeyFrameTrajectory${PREV}_new.txt"
        NEXTMAP_NEW="$TXT_DIR/KeyFrameTrajectory${CURR}_new.txt"
        FINAL_PATH="$TXT_DIR/KeyFrameTrajectory${PREV}${CURR}_new.txt"

        # Define paths of selected 4 frames poses
        TXT_INPUT="$TXT_DIR/images.txt"
        TXT_PARSED="$TXT_DIR/images_parsed.txt"
        
        if [ ! -f "$TXT_INPUT" ]; then
            echo "[ERROR] Colmap Failed!"
            continue
        fi

        # Update the config file with current file paths
        sed -i "s|^firstmap_trajectory_path:.*|firstmap_trajectory_path: '$FIRSTMAP_TRAJ'|" "$ALIGN_CONFIG"
        sed -i "s|^nextmap_trajectory_path:.*|nextmap_trajectory_path: '$NEXTMAP_TRAJ'|" "$ALIGN_CONFIG"
        sed -i "s|^firstmap_new_trajectory_path:.*|firstmap_new_trajectory_path: '$FIRSTMAP_NEW'|" "$ALIGN_CONFIG"
        sed -i "s|^nextmap_new_trajectory_path:.*|nextmap_new_trajectory_path: '$NEXTMAP_NEW'|" "$ALIGN_CONFIG"
        sed -i "s|^final_trajectory_path:.*|final_trajectory_path: '$FINAL_PATH'|" "$ALIGN_CONFIG"
        sed -i "s|^txt_input_path:.*|txt_input_path: '$TXT_INPUT'|" "$ALIGN_CONFIG"
        sed -i "s|^txt_parsed_path:.*|txt_parsed_path: '$TXT_PARSED'|" "$ALIGN_CONFIG"

        # Extract selected 4 frames & align 2 maps
        python3 txt_parsing.py
        python3 align_2_maps.py
    done
done



