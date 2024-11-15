#!/bin/bash

source "$(dirname "$0")/config.sh"

echo "Starting all experiments..."

# Make all scripts executable
chmod +x "$(dirname "$0")/run_trigger_analysis.sh"
chmod +x "$(dirname "$0")/run_memorization_threshold.sh"
chmod +x "$(dirname "$0")/run_semantic_drift.sh"
chmod +x "$(dirname "$0")/run_privacy_sensitivity.sh"
chmod +x "$(dirname "$0")/run_stylometric_analysis.sh"
chmod +x "$(dirname "$0")/run_legal_thresholds.sh"

# Run each experiment
"$(dirname "$0")/run_trigger_analysis.sh"
"$(dirname "$0")/run_memorization_threshold.sh"
"$(dirname "$0")/run_semantic_drift.sh"
"$(dirname "$0")/run_privacy_sensitivity.sh"
"$(dirname "$0")/run_stylometric_analysis.sh"
"$(dirname "$0")/run_legal_thresholds.sh"

echo "All experiments completed." 