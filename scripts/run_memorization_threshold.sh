#!/bin/bash

source "$(dirname "$0")/config.sh"

echo "Running Memorization Threshold Experiment..."

# Generate fine-tuning data first
python3 -m src.memorization_threshold.fine_tune_data_generator

# Run main memorization threshold analysis
python3 -m src.memorization_threshold.memorization_threshold

echo "Memorization Threshold Experiment completed."
