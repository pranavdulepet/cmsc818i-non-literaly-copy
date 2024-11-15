#!/bin/bash

source "$(dirname "$0")/config.sh"

echo "Running Trigger Analysis Experiment..."

# Run main trigger analysis
python3 -m src.trigger_analysis.trigger_analysis

# Run analysis for unique phrases
# python3 -m src.trigger_analysis.trigger_analysis_unique_phrases

# Generate visualizations
# python3 -m src.trigger_analysis.trigger_analysis_visualization

echo "Trigger Analysis Experiment completed."
