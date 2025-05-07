#!/bin/bash

source ../../env.sh

# Iterate over each dataset
for ID in "${DATA_LIST[@]}"
do
    echo "▶ Running LRTD for $ID"
    INPUT_ROOT="$DATA_DIR/$ID" OUTPUT_ROOT="$RESULTS_DIR/$ID" python3 run_our_model.py
done