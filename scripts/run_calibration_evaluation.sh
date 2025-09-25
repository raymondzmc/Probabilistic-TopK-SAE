#!/bin/bash

# Example script for running stratified calibration evaluation
# This analyzes how well activation strength correlates with interpretability

# Set your project and filter
WANDB_PROJECT="raymondl/tinystories-1m"
FILTER_NAME="gated"  # Filter to specific run types if needed

# Run evaluation with stratified calibration
python evaluation.py \
    --wandb_project raymondl/gpt2-small \
    --filter_runs_by_name "$FILTER_NAME" \
    --n_eval_samples 10000 \
    --num_neurons 50 \
    --num_features_to_explain 10 \
    --window_size 64 \
    --generate_explanations \
    --save_activation_data \
    --stratified_calibration \
    --calibration_buckets 5 \
    --examples_per_bucket 20 \
    --explanation_model "gpt-4o" \
    --scoring_model "llama-3.3-70b" \
    --output_path "./artifacts/calibration_analysis"

echo "Calibration evaluation complete!"
echo "Check ./artifacts/calibration_analysis/calibration_plots/ for visualization" 