#!/bin/bash

# Initialize log files
> /home/youngsun/vslam/corl/Choosing4Images/log_colmap.txt
> /home/youngsun/vslam/corl/Choosing4Images/log_tracking_fail_summary.txt

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

    # "result_2025_04_16_111022"
    # "result_2025_04_16_112744"
    # "result_2025_04_16_113145"
    # "result_2025_04_16_113500"
    # "result_2025_04_16_113932"
    # "result_2025_04_16_114324"
    # "result_2025_04_16_114908"
    # "result_2025_04_16_114954"
    # "result_2025_04_17_093724"
    # "result_2025_04_17_094054"
    # "result_2025_04_17_095035"
    # "result_2025_04_17_095758"
    # "result_2025_04_17_100354"
    # "result_2025_04_17_101340"
    # "result_2025_04_17_103828"
    # "result_2025_04_17_103941"
    # "result_2025_04_17_104116"
)

# Path configuration
CHOOSING_DIR="/home/youngsun/vslam/corl/Choosing4Images"
CONNECTING_DIR="/home/youngsun/vslam/corl/ConnectingMaps"
REAL_DATA_ROOT="/mnt/sda/coex_data/long_sequence" # long or short
CONFIG_FILE="$CHOOSING_DIR/config.yaml"

# Iterate over each dataset
for ID in "${DATA_LIST[@]}"
do
    ROOT="$REAL_DATA_ROOT/$ID"
    IMAGES_DIR="$ROOT/images"

    echo "▶ Processing dataset $ID"

    # Update file_path in config.yaml
    sed -i "s|^file_path:.*|file_path: \"$ROOT\"|" "$CONFIG_FILE"

    # Run ORB-SLAM to generate timestamp.txt
    cd ~/vslam/ORB_SLAM3_release/Examples/Monocular
    ./mono_tum ./../../Vocabulary/ORBvoc.txt ./RealSense_D455.yaml "$ROOT"

    for MODE in "yolo" "ocr"
    do
        echo "▶ [$MODE] Mode for dataset $ID"

        # Initialize counters
        total=0
        filtered=0
        colmap_fail=0

        # Run preprocessing 
        cd "$CHOOSING_DIR"
        if [ "$MODE" = "yolo" ]; then
            python3 csv_modifier_yolo.py
        else
            python3 csv_modifier_ocr.py
        fi

        # Run feature matching and select 4 images
        if [ "$MODE" = "yolo" ]; then
            python3 feature_matching_all_threshold_yolo_save_timestamp.py
        else
            python3 feature_matching_all_threshold_ocr.py
        fi

        TXT_FILE="$CHOOSING_DIR/4images_${MODE}.txt"

        DATE_DIR=$(echo "$ID" | cut -d'_' -f2-4 | tr '_' '_')
        SHORT_ID=$(echo "$ID" | cut -d'_' -f5)
        mkdir -p "$CONNECTING_DIR/ORB-SLAM1/$DATE_DIR/$SHORT_ID/$MODE"

        # Read selected 4-image groups
        while IFS= read -r line
        do
            if [[ $line == number\ of\ tracking\ fail* ]]; then
                total=$(echo "$line" | grep -oP '\d+$')
            elif [[ $line == Avg* || $line == Cannot* ]]; then
                ((filtered++))
            elif [[ $line == For*fail* ]]; then
                idx=$(echo "$line" | grep -oP '\d+')
                imgs=()
            elif [[ $line == *.png* ]]; then
                img_file=$(echo "$line" | cut -d' ' -f1)
                imgs+=("$img_file")
            fi

            if [[ ${#imgs[@]} -eq 4 ]]; then
                FAIL_DIR="$CONNECTING_DIR/ORB-SLAM1/$DATE_DIR/$SHORT_ID/$MODE/${idx}th" # orb1 or orb2
                IMG_OUT="$FAIL_DIR/four_frame_images"
                RESULT_OUT="$FAIL_DIR/four_frame_result"

                mkdir -p "$IMG_OUT"
                mkdir -p "$RESULT_OUT"

                for img in "${imgs[@]}"; do
                    cp "$IMAGES_DIR/$img" "$IMG_OUT/"
                done

                cd "$FAIL_DIR"

                # If 4 images are collected, proceed with COLMAP
                colmap feature_extractor \
                    --image_path ./four_frame_images/ \
                    --database_path ./four_frame_result/database.db \
                    --ImageReader.single_camera 1 \
                    --SiftExtraction.gpu_index 0 \
                    --ImageReader.camera_model=SIMPLE_RADIAL \
                    --ImageReader.camera_params 640.744,656.977,365.402,-0.056

                colmap exhaustive_matcher \
                    --database_path ./four_frame_result/database.db

                colmap mapper \
                    --database_path ./four_frame_result/database.db \
                    --image_path ./four_frame_images/ \
                    --output_path ./four_frame_result/

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

        # Save results
        {
            echo "$ID - $MODE"
            echo "Total tracking fail: $total"
            echo "Filtered out (low score or insufficient): $filtered"
            echo "Colmap sparse model failures: $colmap_fail"
            echo ""
        } >> "$CHOOSING_DIR/log_tracking_fail_summary.txt"

        echo ""
    done

done
