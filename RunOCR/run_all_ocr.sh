#!/bin/bash

ROOTS=(
"/mnt/sda/coex_data/long_sequence/result_2025_04_14_093729/"
"/mnt/sda/coex_data/long_sequence/result_2025_04_14_092728/"
"/mnt/sda/coex_data/long_sequence/result_2025_04_14_094411/"
"/mnt/sda/coex_data/long_sequence/result_2025_04_14_094840/"
"/mnt/sda/coex_data/long_sequence/result_2025_04_14_095300/"
"/mnt/sda/coex_data/long_sequence/result_2025_04_14_095731/"
"/mnt/sda/coex_data/long_sequence/result_2025_04_14_100628/"
"/mnt/sda/coex_data/long_sequence/result_2025_04_14_101149/"
"/mnt/sda/coex_data/long_sequence/result_2025_04_14_101618/"
"/mnt/sda/coex_data/long_sequence/result_2025_04_14_102307/"
)

for ROOT in "${ROOTS[@]}"
do
    echo "▶ Running OCR for $ROOT"
    OUTPUT_ROOT="$ROOT" python3 test.py
done