#! /bin/bash

set -e

cd ../code

for i in $(seq 1 10);
do
    echo "Running Seed ${i}"
    python main.py preprocess configs/iclr_2023.jsonnet \
        --config_args "{\"att\":${i}}"

    python main.py train configs/iclr_2023.jsonnet \
        --config_args "{\"att\":${i}}"

    python main.py eval configs/iclr_2023.jsonnet \
        --config_args "{\"att\":${i}}"

done