#!/bin/bash

source "$(dirname "$0")/config.sh"

echo "Running Privacy Sensitivity Experiment..."

# Run privacy sensitivity analysis
python3 -m src.privacy_sensitivity.privacy_sensitivity

echo "Privacy Sensitivity Experiment completed." 