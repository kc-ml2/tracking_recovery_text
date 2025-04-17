#!/bin/bash

> ./Choosing4Images/log_colmap.txt
> ./Choosing4Images/log_tracking_fail_summary.txt

DATA_LIST=(
    "result_2025_04_14_092728"
    "result_2025_04_14_093729"
    "result_2025_04_14_094411"
    "result_2025_04_14_094840"
    "result_2025_04_14_095300"
    "result_2025_04_14_095731"
    "result_2025_04_14_100628"
    "result_2025_04_14_101149"
    "result_2025_04_14_101618"
    "result_2025_04_14_102307"

    # "result_2025_04_16_082550/"
    # "result_2025_04_16_083443/"
    # "result_2025_04_16_084556/"
    # "result_2025_04_16_085517/"
    # "result_2025_04_16_085911/"
    # "result_2025_04_16_090110/"
    # "result_2025_04_16_090537/"
)

CHOOSING_DIR="/home/youngsun/vslam/corl/Choosing4Images"
CONNECTING_DIR="/home/youngsun/vslam/corl/ConnectingMaps"
REAL_DATA_ROOT="/mnt/sda/coex_data/long_sequence"
CONFIG_FILE="$CHOOSING_DIR/config.yaml"

for ID in "${DATA_LIST[@]}"
do
    ROOT="$REAL_DATA_ROOT/$ID"
    IMAGES_DIR="$ROOT/images"

    echo "▶ Processing dataset $ID"

    # 1. Update config.yaml with new file_path
    sed -i "s|^file_path:.*|file_path: \"$ROOT\"|" "$CONFIG_FILE"

    # 2. Run ORB-SLAM (generate timestamp.txt)
    cd ~/vslam/ORB_SLAM3_release/Examples/Monocular
    ./mono_tum ./../../Vocabulary/ORBvoc.txt ./RealSense_D455.yaml "$ROOT"

    for MODE in "yolo" "ocr"
    do
        echo "▶ [$MODE] Mode for dataset $ID"

        # Init counters
        total=0
        filtered=0
        colmap_fail=0

        # 1.5 Run csv_modifier
        cd "$CHOOSING_DIR"
        if [ "$MODE" = "yolo" ]; then
            python3 csv_modifier_yolo.py
        else
            python3 csv_modifier_ocr.py
        fi

        # 3. Run feature matching
        if [ "$MODE" = "yolo" ]; then
            python3 feature_matching_all_threshold_yolo_save_timestamp.py
        else
            python3 feature_matching_all_threshold_ocr.py
        fi

        TXT_FILE="$CHOOSING_DIR/4images_${MODE}.txt"

        DATE_DIR=$(echo "$ID" | cut -d'_' -f2-4 | tr '_' '_')
        SHORT_ID=$(echo "$ID" | cut -d'_' -f5)
        mkdir -p "$CONNECTING_DIR/ORB-SLAM1/$DATE_DIR/$SHORT_ID/$MODE"

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
                FAIL_DIR="$CONNECTING_DIR/ORB-SLAM1/$DATE_DIR/$SHORT_ID/$MODE/${idx}th"
                IMG_OUT="$FAIL_DIR/four_frame_images"
                RESULT_OUT="$FAIL_DIR/four_frame_result"

                mkdir -p "$IMG_OUT"
                mkdir -p "$RESULT_OUT"

                for img in "${imgs[@]}"; do
                    cp "$IMAGES_DIR/$img" "$IMG_OUT/"
                done

                cd "$FAIL_DIR"

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

        # 즉시 tracking summary 기록
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
