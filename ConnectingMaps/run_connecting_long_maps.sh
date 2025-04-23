#!/bin/bash

DATA_LIST=(
    # "result_2025_04_14_092728"
    "result_2025_04_16_084556"
    # "result_2025_04_16_085911"
)

CONFIG_FILE="/home/youngsun/vslam/corl/ConnectingMaps/config_long.yaml"
CONNECTING_DIR="/home/youngsun/vslam/corl/ConnectingMaps/ORB-SLAM2" # orb1 or orb2
REAL_DATA_ROOT="/mnt/sda/coex_data/long_sequence"

for ID in "${DATA_LIST[@]}"
do
    echo "▶ Processing $ID"

    ROOT="$REAL_DATA_ROOT/$ID"
    TRAJ_DIR="$ROOT/orb2_result" # mono or orb2
    DATE_STR=$(echo "$ID" | cut -d'_' -f2-4)
    TIME_STR=$(echo "$ID" | cut -d'_' -f5)
    OUT_PATH="$CONNECTING_DIR/$DATE_STR/$TIME_STR/connected_long.txt"

    # sequences 자동 수집
    mapfile -t SEQUENCES < <(
        find "$TRAJ_DIR" -name "KeyFrameTrajectory*.txt" |
        sed -E 's/.*KeyFrameTrajectory([0-9]+)\.txt/\1/' |
        sort -n |
        awk '{printf "%02d\n", $0}'
        )

    # YAML 포맷 문자열 생성
    SEQUENCE_YAML=$(printf -- "- '%s'\n" "${SEQUENCES[@]}")

    # config.yaml 업데이트 (sed로)
    sed -i "s|^root_dir:.*|root_dir: $REAL_DATA_ROOT|" "$CONFIG_FILE"
    sed -i "s|^date_str:.*|date_str: '$DATE_STR'|" "$CONFIG_FILE"
    sed -i "s|^time_str:.*|time_str: $TIME_STR|" "$CONFIG_FILE"
    sed -i "s|^out_path:.*|out_path: '$OUT_PATH'|" "$CONFIG_FILE"

    # sequences 블록만 정확하게 삭제
    start_line=$(grep -n "^sequences:" "$CONFIG_FILE" | cut -d: -f1)
    if [[ -n "$start_line" ]]; then
        end_line=$(tail -n +"$start_line" "$CONFIG_FILE" | grep -n -m 1 '^[^[:space:]]' | cut -d: -f1)
        if [[ -n "$end_line" ]]; then
            end_line=$((start_line + end_line - 2))
        else
            end_line=$(wc -l < "$CONFIG_FILE")
        fi
        sed -i "${start_line},${end_line}d" "$CONFIG_FILE"
    fi

    # sequences 블록 다시 붙이기
    # sequences 블록 추가 (key 포함해서!)
    {
        echo ""
        echo "sequences:"
        for SEQ in "${SEQUENCES[@]}"; do
            echo "  - '$SEQ'"
        done
    } >> "$CONFIG_FILE"

    # 파이썬 실행
    python3 /home/youngsun/vslam/corl/ConnectingMaps/connecting_long_maps.py
done
