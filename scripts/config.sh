#!/bin/bash

# Base project directory
export PROJECT_ROOT="/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Change to project root directory
cd "$PROJECT_ROOT"

# Data directories
export DATA_DIR="$PROJECT_ROOT/data"
export RESULTS_DIR="$PROJECT_ROOT/experiments"

# Create necessary directories
mkdir -p "$RESULTS_DIR/experiment_1_trigger_analysis/results"
mkdir -p "$RESULTS_DIR/experiment_2_memorization_threshold/results"
mkdir -p "$RESULTS_DIR/experiment_3_semantic_drift/results"
mkdir -p "$RESULTS_DIR/experiment_4_privacy_sensitivity/results"
mkdir -p "$RESULTS_DIR/experiment_5_stylometry/results"
mkdir -p "$RESULTS_DIR/experiment_6_legal_thresholds/results" 