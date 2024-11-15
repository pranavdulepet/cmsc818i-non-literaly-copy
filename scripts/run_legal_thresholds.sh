#!/bin/bash

source "$(dirname "$0")/config.sh"

echo "Running Legal Thresholds Experiment..."

# Run legal thresholds analysis
python3 -m src.legal_thresholds.legal_similarity_threshold

echo "Legal Thresholds Experiment completed." 