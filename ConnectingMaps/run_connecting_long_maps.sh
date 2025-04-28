#!/bin/bash

DATA_LIST=(
    # "result_2025_04_14_092728"
    # "result_2025_04_16_084556"
    # "result_2025_04_16_085911"

    "result_2025_04_16_111022"
    "result_2025_04_16_113500"
    "result_2025_04_16_114324"
    "result_2025_04_16_114954"
    "result_2025_04_17_093724"
    "result_2025_04_17_094054"
    "result_2025_04_17_095035"
    "result_2025_04_17_095758"
    "result_2025_04_17_100354"
    "result_2025_04_17_101340"
    "result_2025_04_17_103941"
    "result_2025_04_17_104116"
)

CONNECTING_DIR="/home/youngsun/vslam/corl/ConnectingMaps/ORB-SLAM1" # orb1 or orb2
ROOT_DIR="/mnt/sda/coex_data/long_sequence" # long or short 

for DATA in "${DATA_LIST[@]}"
do
    SECONDS=0
    echo "▶ Processing dataset $DATA"
    
    DATE_STR=$(echo $DATA | cut -d'_' -f2-4)
    TIME_STR=$(echo $DATA | cut -d'_' -f5)
    TRAJ_DIR="${ROOT_DIR}/${DATA}/mono_result" # mono or orb2

    # 시퀀스 자동 추출 (숫자 2자리만 있는 파일명만 필터링)
    SEQ_LIST=$(find $TRAJ_DIR -maxdepth 1 -type f -name 'KeyFrameTrajectory[0-9][0-9].txt' \
                -exec basename {} \; | sed -n 's/KeyFrameTrajectory\([0-9][0-9]\)\.txt/\1/p' | sort)

    # 연결 결과 저장 경로 구성
    OUT_PATH="${CONNECTING_DIR}/${DATE_STR}/${TIME_STR}/connected_long.txt"
    CONFIG_PATH="/home/youngsun/vslam/corl/ConnectingMaps/config.yaml"

    # sequences를 YAML 배열 형식으로 변환
    YAML_SEQ=""
    for SEQ in $SEQ_LIST; do
        YAML_SEQ+="  - '$SEQ'\n"
    done

cat > "$CONFIG_PATH" <<EOF
connecting_dir: ${CONNECTING_DIR}
final_trajectory_path: ''
firstmap_new_trajectory_path: ''
firstmap_trajectory_path: ''
nextmap_new_trajectory_path: ''
nextmap_trajectory_path: ''
txt_input_path: ''
txt_parsed_path: ''

root_dir: ${ROOT_DIR}
date_str: '$DATE_STR'
time_str: '$TIME_STR'
out_path: '$OUT_PATH'
sequences:
$(echo -e "$YAML_SEQ")
EOF

#     # 파이썬 스크립트 실행
#     python3 connecting_long_maps.py
#         {
#         echo "$DATA - $MODE"
#         echo "Total tracking fail: $total"
#         echo "Filtered out (low score or insufficient): $filtered"
#         echo "Colmap sparse model failures: $colmap_fail"
#         echo ""
#     } >> "$CHOOSING_DIR/log_tracking_fail_summary.txt"
done
