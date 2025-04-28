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

# "/mnt/sda/coex_data/short_sequence/result_2025_04_16_111022"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_16_113500"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_16_114324"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_16_114054"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_17_093724"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_17_094054"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_17_095035"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_17_095758"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_17_100354"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_17_101340"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_17_103941"
# "/mnt/sda/coex_data/short_sequence/result_2025_04_17_104116"

"/mnt/sda/coex_data/short_sequence/result_2025_04_28"
)

for ROOT in "${ROOTS[@]}"
do
    echo "▶ Running OCR for $ROOT"
    OUTPUT_ROOT="$ROOT" python3 test.py
done