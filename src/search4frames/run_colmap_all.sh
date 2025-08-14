#!/bin/bash

source ../../env.sh

# Iterate over each dataset
for ID in "${DATA_LIST[@]}"
do
    ROOT="$DATA_DIR/$ID"
    IMAGES_DIR="$ROOT/images"

    echo "▶ Processing dataset $ID"

    # Update data/timestamp file paths in config fiile
    sed -i "s|^file_path:.*|file_path: \"$ROOT\"|" "$SEARCH_CONFIG"

    TIMESTAMP_FILE="$DATA_DIR/$ID/orb_result/timestamp.txt" # orb or orb2
    sed -i "s|^timestamp_saved_path:.*|timestamp_saved_path: \"$TIMESTAMP_FILE\"|" "$SEARCH_CONFIG"

    # Loop over LRTD mode
    for MODE in "LRTD"
    do
        # Initialize log files
        > $RESULTS_DIR/$ID/log_4images.txt
        > $RESULTS_DIR/$ID/log_colmap.txt
        > $RESULTS_DIR/$ID/log_tracking_fail.txt

        echo "▶ [$MODE] Mode for dataset $ID"
        
        total=0 # Number of total tracking failures
        filtered=0 # Number of filtered out(insufficient) tracking failures
        colmap_fail=0 # Number of filtered tracking failures that failed COLMAP reconstruction

        cd "$SELECTING_DIR"
        sed -i "s|^mode:.*|mode: \"$MODE\"|" "$SEARCH_CONFIG"
        sed -i "s|^mode:.*|mode: \"$MODE\"|" "$SEARCH_CONFIG"
        sed -i "s|^csv_.*_path:.*|csv_${MODE}_path: \"${RESULTS_DIR}/${ID}/${MODE}_info.csv\"|" "$SEARCH_CONFIG"
        sed -i "s|^filtered_csv_.*_path:.*|filtered_csv_${MODE}_path: \"${RESULTS_DIR}/${ID}/${MODE}_filtered_info.csv\"|" "$SEARCH_CONFIG"
        sed -i "s|^4images_.*_path:.*|4images_${MODE}_path: \"${RESULTS_DIR}/${ID}/log_4images.txt\"|" "$SEARCH_CONFIG"        

        # Load filtered LRTD information and select 4 frames
        python3 csv_modifier.py
        python3 select_images.py

        # Log timestamps of the 4 selected frames for sach failure
        TXT_FILE="$RESULTS_DIR/$ID/log_4images.txt"

        mkdir -p "$RESULTS_DIR/$ID/COLMAP" # orb1 or orb2

        # Save images and COLMAP result of 4 selected frame sets
        while IFS= read -r line
        do
            # Read the timestamp log of the sets
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

            # If 4 images are selected, save images and COLMAP results
            if [[ ${#imgs[@]} -eq 4 ]]; then
                idx="$current_idx"
                FAIL_DIR="$RESULTS_DIR/$ID/COLMAP/${idx}th" # orb1 or orb2
                IMG_OUT="$FAIL_DIR/four_frame_images"
                RESULT_OUT="$FAIL_DIR/four_frame_result"

                mkdir -p "$IMG_OUT" "$RESULT_OUT"
                
                # Save images
                for img in "${imgs[@]}"; do
                    cp "$IMAGES_DIR/$img" "$IMG_OUT/"
                done

                cd "$FAIL_DIR"

                DB_PATH="./four_frame_result/database.db"
                [ -f "$DB_PATH" ] && rm "$DB_PATH"

                # Run COLMAP
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

                # Save COLMAP results if reconstruction is sufficient
                if [ -f "./four_frame_result/0/cameras.bin" ]; then
                    mkdir -p ./four_frame_result/0/txt
                    colmap model_converter \
                        --input_path ./four_frame_result/0 \
                        --output_path ./four_frame_result/0/txt \
                        --output_type TXT
                    echo "$ID - $MODE - ${idx}th fail - sparse model successed" >> "$RESULTS_DIR/$ID/log_colmap.txt"
                else
                    echo "$ID - $MODE - ${idx}th fail - sparse model failed" >> "$RESULTS_DIR/$ID/log_colmap.txt"
                    ((colmap_fail++))
                fi

                imgs=()
            fi
        done < "$TXT_FILE"

        # Log the results
        {
            echo "$ID - $MODE"
            echo "Total tracking failures: $total"
            echo "Failed selecting 4 images: $filtered"
            echo "Failed COLMAP reconstruction: $colmap_fail"
            echo ""
        } >> "$RESULTS_DIR/$ID/log_tracking_fail.txt"

        echo ""
    done
done