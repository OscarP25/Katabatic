#!/usr/bin/env bash

set -e

models=("arf")
datasets=("adult" "car" "magic" "nursery" "shuttle")

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "==============================="
        echo " Running $model on $dataset "
        echo "==============================="

        LOG_FILE="${LOG_DIR}/${model}_${dataset}.log"

        python run_model.py "$model" "$dataset" | tee "$LOG_FILE"

        echo "Finished $model on $dataset. Log saved to $LOG_FILE"
        echo
    done
done

echo "All runs completed."
