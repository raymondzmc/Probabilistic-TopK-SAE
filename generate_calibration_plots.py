#!/usr/bin/env python3
"""
Standalone script to generate calibration plots from saved explanation scores.
This allows regenerating plots without re-running the entire evaluation.
"""

import json
import argparse
from pathlib import Path
from evaluation import create_calibration_plots


def main():
    parser = argparse.ArgumentParser(description="Generate calibration plots from saved explanation scores")
    parser.add_argument("--scores_file", type=str, required=True,
                       help="Path to the explanation_scores JSON file")
    parser.add_argument("--run_id", type=str, required=True,
                       help="Run ID for naming the plots")
    parser.add_argument("--output_path", type=str, default="./artifacts",
                       help="Base output path for plots (default: ./artifacts)")
    
    args = parser.parse_args()
    
    # Load the saved scores
    print(f"Loading scores from: {args.scores_file}")
    with open(args.scores_file, 'r') as f:
        all_explanation_scores = json.load(f)
    
    # Check if there's calibration data
    has_calibration = False
    for sae_pos, scores_list in all_explanation_scores.items():
        if any('bucket_detection_scores' in s for s in scores_list):
            has_calibration = True
            break
    
    if not has_calibration:
        print("No calibration data found in the scores file!")
        return
    
    # Generate the plots
    print(f"Generating calibration plots...")
    create_calibration_plots(all_explanation_scores, args.output_path, args.run_id)
    print("Done!")


if __name__ == "__main__":
    main() 