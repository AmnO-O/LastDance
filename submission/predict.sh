#!/bin/bash
set -e
python3 /code/preprocessing.py
python3 /code/predict.py "$@"
