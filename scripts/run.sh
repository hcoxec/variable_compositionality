#! /bin/bash

set -e

cd ../code

python main.py preprocess ../configs/iclr_2023.jsonnet

python main.py train ../configs/iclr_2023.jsonnet

python main.py eval ../configs/iclr_2023.jsonnet