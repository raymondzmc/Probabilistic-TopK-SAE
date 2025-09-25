# Stratified Calibration Analysis

This feature allows you to measure how well SAE neuron activation strength correlates with interpretability scores.

## Overview

The stratified calibration evaluation:
1. Divides neuron activations into percentile buckets (e.g., 0-20%, 20-40%, ..., 80-100%)
2. Generates explanations using only the highest activation examples (top bucket)
3. Evaluates these explanations on examples from each bucket separately
4. Measures how interpretability scores vary across activation strengths

## Why Calibration Matters

A well-calibrated SAE neuron should show higher interpretability scores for higher activation values. This indicates that:
- The neuron's activation strength is meaningful
- High activations genuinely represent stronger presence of the detected feature
- The explanations derived from high activations generalize to lower activations

## Usage

To run evaluation with stratified calibration:

```bash
python evaluation.py \
    --generate_explanations \
    --stratified_calibration \
    --calibration_buckets 5 \
    --examples_per_bucket 20 \
    --num_features_to_explain 10 \
    # ... other arguments
```

Key parameters:
- `--stratified_calibration`: Enable calibration analysis
- `--calibration_buckets`: Number of percentile buckets (default: 5 for quintiles)
- `--examples_per_bucket`: Examples to evaluate per bucket (default: 20)

## Outputs

The calibration analysis produces:

1. **Calibration Plots** (`calibration_plots/`):
   - Line plots showing mean score vs activation percentile
   - Error bars indicate standard deviation across neurons
   - Ideal calibration line for reference

2. **Heatmaps** (for runs with >5 neurons):
   - Per-neuron scores across all buckets
   - Visualizes consistency of calibration across neurons

3. **Calibration Metrics**:
   - Correlation coefficient between activation percentile and score
   - Printed in the summary statistics

## Interpreting Results

### Good Calibration
- Positive correlation between activation strength and interpretability scores
- Monotonically increasing scores across buckets
- Low variance within buckets

### Poor Calibration
- Flat or negative correlation
- High scores only in top bucket, low elsewhere (overfitting to high activations)
- High variance suggesting inconsistent behavior

## Example Script

See `scripts/run_calibration_evaluation.sh` for a complete example.

## Implementation Details

The calibration analysis modifies the standard evaluation pipeline:
- **Explanation Generation**: Always uses top bucket examples
- **Evaluation**: Tests on stratified samples from each bucket
- **Scoring**: Computes scores separately per bucket, then aggregates

This approach reveals whether high-activation patterns that generate good explanations actually generalize across the full activation spectrum. 