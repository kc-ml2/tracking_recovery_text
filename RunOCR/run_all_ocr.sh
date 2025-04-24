#!/bin/bash

ROOTS=(
# "/mnt/sda/coex_data/long_sequence/result_2025_04_14_092728/"
# "/mnt/sda/coex_data/long_sequence/result_2025_04_14_093729/"
# "/mnt/sda/coex_data/long_sequence/result_2025_04_14_094840/"
# "/mnt/sda/coex_data/long_sequence/result_2025_04_14_101149/"
# "/mnt/sda/coex_data/long_sequence/result_2025_04_14_101618/"
# "/mnt/sda/coex_data/long_sequence/result_2025_04_16_082550/"
# "/mnt/sda/coex_data/long_sequence/result_2025_04_16_084556/"
# "/mnt/sda/coex_data/long_sequence/result_2025_04_16_085517/"
# "/mnt/sda/coex_data/long_sequence/result_2025_04_16_085911/"
# "/mnt/sda/coex_data/long_sequence/result_2025_04_16_090110/"
)

for ROOT in "${ROOTS[@]}"
do
    echo "▶ Running OCR for $ROOT"
    OUTPUT_ROOT="$ROOT" python3 test.py
done