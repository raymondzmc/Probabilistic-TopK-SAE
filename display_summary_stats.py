#!/usr/bin/env python3
"""
Display summary statistics from saved explanation scores.
"""

import json
import numpy as np
import argparse
from pathlib import Path


def display_summary(scores_file: str):
    """Display summary statistics from saved scores."""
    print(f"Loading scores from: {scores_file}")
    with open(scores_file, 'r') as f:
        all_explanation_scores = json.load(f)
    
    print("\n" + "=" * 60)
    print("EXPLANATION SCORES SUMMARY")
    print("=" * 60)
    
    # Display summary statistics by layer
    print("\nScore Statistics by Layer:")
    all_scores = []  # For overall top explanations
    
    for sae_pos, scores_list in all_explanation_scores.items():
        if not scores_list:
            continue
            
        # Get scores for this layer (handle both regular and multi-bucket entries)
        layer_detection_scores = []
        for s in scores_list:
            if 'detection_score' in s and s['detection_score'] is not None:
                # Regular entry
                layer_detection_scores.append(s['detection_score'])
            elif 'bucket_detection_scores' in s:
                # Multi-bucket entry - compute average across buckets
                bucket_scores = [float(score) for score in s['bucket_detection_scores'].values()]
                if bucket_scores:
                    layer_detection_scores.append(np.mean(bucket_scores))
        
        print(f"\n  {sae_pos}:")
        print(f"    Total neurons: {len(scores_list)}")
        if layer_detection_scores:
            print(f"    Detection: mean={np.mean(layer_detection_scores):.3f}, std={np.std(layer_detection_scores):.3f}, n={len(layer_detection_scores)}")
        
        # Check if multi-bucket explanations
        multi_bucket_count = sum(1 for s in scores_list if s.get('multi_bucket', False))
        if multi_bucket_count > 0:
            print(f"    Multi-bucket explanations: {multi_bucket_count}/{len(scores_list)}")
        
        # Add to overall list with sae_pos for top explanations display
        for score_dict in scores_list:
            score_dict_with_pos = score_dict.copy()
            score_dict_with_pos['sae_pos'] = sae_pos
            all_scores.append(score_dict_with_pos)

    # Display overall statistics (handle both regular and multi-bucket entries)
    overall_detection_scores = []
    for s in all_scores:
        if 'detection_score' in s and s['detection_score'] is not None:
            overall_detection_scores.append(s['detection_score'])
        elif 'bucket_detection_scores' in s:
            bucket_scores = [float(score) for score in s['bucket_detection_scores'].values()]
            if bucket_scores:
                overall_detection_scores.append(np.mean(bucket_scores))
    
    print(f"\n  Overall (All Layers):")
    print(f"    Total neurons: {len(all_scores)}")
    if overall_detection_scores:
        print(f"    Detection: mean={np.mean(overall_detection_scores):.3f}, std={np.std(overall_detection_scores):.3f}, n={len(overall_detection_scores)}")

    # Display top scoring explanations across all layers
    print("\nTop 5 explanations by Detection score (across all layers):")
    # Filter and compute scores for sorting
    scores_for_sorting = []
    for s in all_scores:
        if 'detection_score' in s and s['detection_score'] is not None:
            score_val = s['detection_score']
            scores_for_sorting.append((s, score_val))
        elif 'bucket_detection_scores' in s:
            bucket_scores = [float(score) for score in s['bucket_detection_scores'].values()]
            if bucket_scores:
                score_val = np.mean(bucket_scores)
                scores_for_sorting.append((s, score_val))
    
    sorted_by_detection = sorted(scores_for_sorting, key=lambda x: x[1], reverse=True)[:5]
    for i, (score_entry, score_val) in enumerate(sorted_by_detection, 1):
        print(f"  {i}. Neuron {score_entry['neuron_idx']} ({score_entry['sae_pos']}): {score_val:.3f}")
        if 'explanation' in score_entry:
            print(f"     {score_entry['explanation'][:100]}...")
        elif 'bucket_explanations' in score_entry:
            # For multi-bucket, show the highest bucket's explanation
            bucket_scores_dict = score_entry.get('bucket_detection_scores', {})
            if bucket_scores_dict:
                # Convert string keys to handle JSON loading
                bucket_scores_items = [(int(k) if isinstance(k, str) else k, v) 
                                      for k, v in bucket_scores_dict.items()]
                best_bucket_idx, _ = max(bucket_scores_items, key=lambda x: x[1])
                best_bucket = str(best_bucket_idx)  # Convert back to string for dict access
                if best_bucket in score_entry['bucket_explanations']:
                    print(f"     [Multi-bucket, best bucket {best_bucket_idx}]: {score_entry['bucket_explanations'][best_bucket][:100]}...")
    
    # Display calibration metrics if available
    calibration_metrics_path = Path(scores_file).parent / "calibration_metrics.json"
    if calibration_metrics_path.exists():
        print("\n" + "=" * 60)
        print("CALIBRATION METRICS")
        print("=" * 60)
        with open(calibration_metrics_path, 'r') as f:
            calibration_metrics = json.load(f)
        
        for sae_pos, metrics in calibration_metrics.items():
            print(f"\n{sae_pos}:")
            print(f"  Expected Calibration Error (ECE): {metrics['ece']:.4f}")
            print(f"  Correlation (activation vs detection): {metrics['correlation']:.4f}")
            print(f"  Maximum Calibration Error (MCE): {metrics['mce']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Display summary statistics from saved explanation scores")
    parser.add_argument("--scores_file", type=str, required=True,
                       help="Path to the explanation_scores JSON file")
    
    args = parser.parse_args()
    display_summary(args.scores_file)


if __name__ == "__main__":
    main() 