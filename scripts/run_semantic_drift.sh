#!/bin/bash

source "$(dirname "$0")/config.sh"

echo "Running Semantic Drift Experiment..."

# Run semantic drift analysis
python3 -m src.semantic_drift.semantic_drift

echo "Semantic Drift Experiment completed."
