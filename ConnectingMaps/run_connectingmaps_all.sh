#!/bin/bash

BASE_DIR="/home/youngsun/vslam/corl/ConnectingMaps/ORB-SLAM1"
CONFIG_FILE="/home/youngsun/vslam/corl/ConnectingMaps/config.yaml"
DATA_LIST=(
    # "result_2025_04_14_092728"
    # "result_2025_04_14_093729"
    # "result_2025_04_14_094840"
    # "result_2025_04_14_101149"
    # "result_2025_04_14_101618"
    # "result_2025_04_16_082550"
    # "result_2025_04_16_084556"
    # "result_2025_04_16_085517"
    # "result_2025_04_16_085911"
    # "result_2025_04_16_090110"

    "result_2025_04_16_111022"
    "result_2025_04_16_112744"
    "result_2025_04_16_113145"
    "result_2025_04_16_113500"
    "result_2025_04_16_113932"
    "result_2025_04_16_114324"
    "result_2025_04_16_114908"
    "result_2025_04_16_114954"


)

echo "[INFO] Running connecting_maps.py for selected sequences..."
echo " "

for ID in "${DATA_LIST[@]}"
do
    DATE_DIR=$(echo "$ID" | cut -d'_' -f2-4 | tr '_' '_')
    TIME_STR=$(echo "$ID" | cut -d'_' -f5)
    ROOT="/mnt/sda/coex_data/short_sequence/$ID" # long or short

    for FAIL_DIR in "$BASE_DIR/$DATE_DIR/$TIME_STR"/yolo/*th/; do

        IDX_NAME=$(basename "$FAIL_DIR")      
        IDX=$(echo "$IDX_NAME" | grep -oP '\d+')

     
        if [[ -z "$IDX" ]]; then
            echo "Processing $DATE_DIR/$TIME_STR/"
            echo "[ERROR] No Colmap!"
            echo ""
            continue
        fi

        PREV=$((IDX - 1))
        CURR=$((IDX))

        echo "Processing $DATE_DIR/$TIME_STR/$IDX_NAME (Traj $PREV + $CURR)"

        TXT_DIR="$FAIL_DIR/four_frame_result/0/txt"
        mkdir -p "$TXT_DIR"

        FIRSTMAP_TRAJ="$ROOT/mono_result/KeyFrameTrajectory${PREV}.txt"
        NEXTMAP_TRAJ="$ROOT/mono_result/KeyFrameTrajectory${CURR}.txt"

        FIRSTMAP_NEW="$TXT_DIR/KeyFrameTrajectory${PREV}_new.txt"
        NEXTMAP_NEW="$TXT_DIR/KeyFrameTrajectory${CURR}_new.txt"
        FINAL_PATH="$TXT_DIR/KeyFrameTrajectory${PREV}${CURR}_new.txt"

        TXT_INPUT="$TXT_DIR/images.txt"
        TXT_PARSED="$TXT_DIR/images_parsed.txt"

        # images.txt 없으면 건너뜀
        if [ ! -f "$TXT_INPUT" ]; then
            echo "[WARN] $TXT_INPUT not found. skipping..."
            continue
        fi

        # update config.yaml
        sed -i "s|^firstmap_trajectory_path:.*|firstmap_trajectory_path: '$FIRSTMAP_TRAJ'|" "$CONFIG_FILE"
        sed -i "s|^nextmap_trajectory_path:.*|nextmap_trajectory_path: '$NEXTMAP_TRAJ'|" "$CONFIG_FILE"
        sed -i "s|^firstmap_new_trajectory_path:.*|firstmap_new_trajectory_path: '$FIRSTMAP_NEW'|" "$CONFIG_FILE"
        sed -i "s|^nextmap_new_trajectory_path:.*|nextmap_new_trajectory_path: '$NEXTMAP_NEW'|" "$CONFIG_FILE"
        sed -i "s|^final_trajectory_path:.*|final_trajectory_path: '$FINAL_PATH'|" "$CONFIG_FILE"
        sed -i "s|^txt_input_path:.*|txt_input_path: '$TXT_INPUT'|" "$CONFIG_FILE"
        sed -i "s|^txt_parsed_path:.*|txt_parsed_path: '$TXT_PARSED'|" "$CONFIG_FILE"

        # run parsing and connecting
        python3 /home/youngsun/vslam/corl/ConnectingMaps/txt_parsing.py
        python3 /home/youngsun/vslam/corl/ConnectingMaps/connecting_maps.py
    done
done



