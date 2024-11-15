#!/bin/bash

source "$(dirname "$0")/config.sh"

echo "Running Stylometric Analysis Experiment..."

# Run stylometric analysis
python3 -m src.stylometric_analysis

echo "Stylometric Analysis Experiment completed." 