#!/bin/bash

# Initialize log files
> ./log_colmap.txt
> ./log_tracking_fail_summary.txt
> ./log_model_used_frames.txt
> ./log_time_cost.txt

# List of dataset IDs (folder names under REAL_DATA_ROOT)
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
    # "result_2025_04_16_113500"
    "result_2025_04_16_114324"
    "result_2025_04_16_114954"
    "result_2025_04_17_093724"
    "result_2025_04_17_094054"
    "result_2025_04_17_095035"
    "result_2025_04_17_095758"
    "result_2025_04_17_100354"
    "result_2025_04_17_101340"
    # "result_2025_04_17_103941"
    # "result_2025_04_17_104116"
)

# Path configuration
CHOOSING_DIR="/home/youngsun/vslam/corl/Choosing4Images"
CONNECTING_DIR="/home/youngsun/vslam/corl/ConnectingMaps"
REAL_DATA_ROOT="/mnt/sda/coex_data/short_sequence" # long or short
CONFIG_FILE="$CHOOSING_DIR/config.yaml"

# Iterate over each dataset
for ID in "${DATA_LIST[@]}"
do
    start_time=$(date +%s.%N)

    ROOT="$REAL_DATA_ROOT/$ID"
    IMAGES_DIR="$ROOT/images"

    echo "▶ Processing dataset $ID"

    # Update file path in config.yaml
    sed -i "s|^file_path:.*|file_path: \"$ROOT\"|" "$CONFIG_FILE"

    # Update timestamp path in config.yaml
    DATE_DIR=$(echo "$ID" | cut -d'_' -f2-4 | tr '_' '_')
    TIME_STR=$(echo "$ID" | cut -d'_' -f5)
    TIMESTAMP_FILE="/home/youngsun/vslam/corl/AllTimestamp/ORB-SLAM1/$DATE_DIR/$TIME_STR.txt" # orb1 or orb2
    sed -i "s|^timestamp_saved_path:.*|timestamp_saved_path: \"$TIMESTAMP_FILE\"|" "$CONFIG_FILE"

    # Loop over both 'yolo' and 'ocr' modes
    for MODE in "yolo" "ocr"
    do
        echo "▶ [$MODE] Mode for dataset $ID"

        total=0
        filtered=0
        colmap_fail=0

        cd "$CHOOSING_DIR"

        # Preprocessing and matching
        sed -i "s|^mode:.*|mode: \"$MODE\"|" "$CONFIG_FILE"

        python3 csv_modifier.py
        python3 select_images.py

        TXT_FILE="$CHOOSING_DIR/4images_${MODE}.txt"

        mkdir -p "$CONNECTING_DIR/ORB-SLAM1/$DATE_DIR/$TIME_STR/$MODE" # orb1 or orb2

        # Read selected 4 image groups
        while IFS= read -r line
        do
            if [[ $line == number\ of\ tracking\ fail* ]]; then
                total=$(echo "$line" | grep -oP '\d+$')
            elif [[ $line == Avg* || $line == Cannot* ]]; then
                ((filtered++))
            elif [[ $line == For*fail* ]]; then
                current_idx=$(echo "$line" | grep -oP '\d+')
                imgs=()
            elif [[ $line == *.png* ]]; then
                img_file=$(echo "$line" | cut -d' ' -f1)
                imgs+=("$img_file")
            fi

            if [[ ${#imgs[@]} -eq 4 ]]; then
                idx="$current_idx"
                FAIL_DIR="$CONNECTING_DIR/ORB-SLAM1/$DATE_DIR/$TIME_STR/$MODE/${idx}th" # orb1 or orb2
                IMG_OUT="$FAIL_DIR/four_frame_images"
                RESULT_OUT="$FAIL_DIR/four_frame_result"

                mkdir -p "$IMG_OUT" "$RESULT_OUT"

                for img in "${imgs[@]}"; do
                    cp "$IMAGES_DIR/$img" "$IMG_OUT/"
                done

                cd "$FAIL_DIR"

                # Remove previous database if exists
                DB_PATH="./four_frame_result/database.db"
                [ -f "$DB_PATH" ] && rm "$DB_PATH"

                # If 4 images are collected, proceed with COLMAP
                colmap feature_extractor \
                    --image_path ./four_frame_images/ \
                    --database_path ./four_frame_result/database.db \
                    --ImageReader.single_camera 1 \
                    --SiftExtraction.gpu_index 0 \
                    --ImageReader.camera_model=SIMPLE_RADIAL \
                    --ImageReader.camera_params 640.744,656.977,365.402,-0.056 \
                    --SiftExtraction.edge_threshold 20

                colmap exhaustive_matcher \
                    --database_path ./four_frame_result/database.db \
                    --TwoViewGeometry.min_num_inliers 8

                colmap mapper \
                    --database_path ./four_frame_result/database.db \
                    --image_path ./four_frame_images/ \
                    --output_path ./four_frame_result/ \
                    --Mapper.min_num_matches 8 \
                    --Mapper.abs_pose_min_num_inliers 8 \
                    --Mapper.init_min_tri_angle 8

                # Convert COLMAP result to TXT if reconstruction is sufficient
                if [ -f "./four_frame_result/0/cameras.bin" ]; then
                    mkdir -p ./four_frame_result/0/txt
                    colmap model_converter \
                        --input_path ./four_frame_result/0 \
                        --output_path ./four_frame_result/0/txt \
                        --output_type TXT
                    echo "$ID - $MODE - ${idx}th fail - sparse model successed" >> "$CHOOSING_DIR/log_colmap.txt"
                else
                    echo "$ID - $MODE - ${idx}th fail - sparse model failed" >> "$CHOOSING_DIR/log_colmap.txt"
                    ((colmap_fail++))
                fi

                imgs=()
            fi
        done < "$TXT_FILE"

        end_time=$(date +%s.%N)
        total_time=$(echo "$end_time - $start_time" | bc)
        total_time_fmt=$(printf "%.3f" "$total_time")

        # Save results
        {
            echo "$ID - $MODE"
            echo "Total tracking fail: $total"
            echo "Filtered out (low score or insufficient): $filtered"
            echo "Colmap sparse model failures: $colmap_fail"
            echo ""
        } >> "$CHOOSING_DIR/log_tracking_fail_summary.txt"

        {
            echo "$ID - $MODE"
            echo "Time cost for choosing 4 images & running colmap: ${total_time_fmt} sec"
            echo ""
        } >> "$CHOOSING_DIR/log_time_cost.txt"

        echo ""
    done
done