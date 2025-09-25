#!/usr/bin/env python3
"""
Compute calibration metrics including Expected Calibration Error (ECE) from saved explanation scores.
"""

import json
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def compute_calibration_metrics(all_explanation_scores: dict):
    """
    Compute calibration metrics for each SAE position.
    
    Returns:
        dict: Calibration metrics for each SAE position
    """
    calibration_results = {}
    
    for sae_pos, scores_list in all_explanation_scores.items():
        # Filter to entries with bucket scores
        calibration_entries = [s for s in scores_list if 'bucket_detection_scores' in s]
        
        if not calibration_entries:
            continue
            
        print(f"\nComputing calibration metrics for {sae_pos}")
        print(f"Found {len(calibration_entries)} neurons with calibration data")
        
        # Collect data for ECE computation
        all_activations = []
        all_detection_scores = []
        bucket_data = {}
        
        # Get bucket info from first entry
        bucket_boundaries = calibration_entries[0]['bucket_boundaries']
        bucket_percentiles = calibration_entries[0]['bucket_percentiles']
        n_buckets = len(bucket_percentiles) - 1
        
        # Initialize bucket data
        for i in range(n_buckets):
            bucket_data[i] = {
                'activations': [],
                'detection_scores': [],
                'bucket_min': bucket_boundaries[i],
                'bucket_max': bucket_boundaries[i+1],
                'percentile_min': bucket_percentiles[i],
                'percentile_max': bucket_percentiles[i+1]
            }
        
        # Process each neuron
        for entry in calibration_entries:
            # For multi-bucket entries, we need to handle differently
            if entry.get('multi_bucket', False):
                # Multi-bucket: each bucket has its own explanation and score
                for bucket_idx, det_score in entry['bucket_detection_scores'].items():
                    bucket_idx = int(bucket_idx) if isinstance(bucket_idx, str) else bucket_idx
                    
                    # Use bucket midpoint as representative activation
                    bucket_min = bucket_data[bucket_idx]['bucket_min']
                    bucket_max = bucket_data[bucket_idx]['bucket_max']
                    activation_midpoint = (bucket_min + bucket_max) / 2
                    
                    bucket_data[bucket_idx]['activations'].append(activation_midpoint)
                    bucket_data[bucket_idx]['detection_scores'].append(det_score)
                    
                    all_activations.append(activation_midpoint)
                    all_detection_scores.append(det_score)
            else:
                # Single explanation evaluated on multiple buckets
                for bucket_idx, det_score in entry['bucket_detection_scores'].items():
                    bucket_idx = int(bucket_idx) if isinstance(bucket_idx, str) else bucket_idx
                    
                    # Use bucket midpoint as representative activation
                    bucket_min = bucket_data[bucket_idx]['bucket_min']
                    bucket_max = bucket_data[bucket_idx]['bucket_max']
                    activation_midpoint = (bucket_min + bucket_max) / 2
                    
                    bucket_data[bucket_idx]['activations'].append(activation_midpoint)
                    bucket_data[bucket_idx]['detection_scores'].append(det_score)
                    
                    all_activations.append(activation_midpoint)
                    all_detection_scores.append(det_score)
        
        # Normalize activations to [0, 1] range
        all_activations = np.array(all_activations)
        all_detection_scores = np.array(all_detection_scores)
        
        # Min-max normalization
        if all_activations.max() > all_activations.min():
            normalized_activations = (all_activations - all_activations.min()) / (all_activations.max() - all_activations.min())
        else:
            normalized_activations = np.zeros_like(all_activations)
        
        # Compute ECE (Expected Calibration Error)
        ece = 0.0
        bucket_metrics = []
        
        for bucket_idx in range(n_buckets):
            if len(bucket_data[bucket_idx]['activations']) == 0:
                continue
                
            bucket_acts = np.array(bucket_data[bucket_idx]['activations'])
            bucket_scores = np.array(bucket_data[bucket_idx]['detection_scores'])
            
            # Normalize bucket activations
            if all_activations.max() > all_activations.min():
                normalized_bucket_acts = (bucket_acts - all_activations.min()) / (all_activations.max() - all_activations.min())
            else:
                normalized_bucket_acts = np.zeros_like(bucket_acts)
            
            # Compute average normalized activation and detection score for this bucket
            avg_normalized_act = normalized_bucket_acts.mean()
            avg_detection = bucket_scores.mean()
            
            # Bucket weight (proportion of samples)
            bucket_weight = len(bucket_scores) / len(all_detection_scores)
            
            # Calibration error for this bucket
            bucket_error = abs(avg_normalized_act - avg_detection)
            
            # Add to ECE
            ece += bucket_weight * bucket_error
            
            bucket_metrics.append({
                'bucket_idx': bucket_idx,
                'percentile_range': f"{bucket_data[bucket_idx]['percentile_min']:.0f}-{bucket_data[bucket_idx]['percentile_max']:.0f}%",
                'n_samples': len(bucket_scores),
                'avg_normalized_activation': avg_normalized_act,
                'avg_detection_score': avg_detection,
                'calibration_error': bucket_error,
                'weight': bucket_weight
            })
        
        # Compute additional metrics
        # 1. Correlation between normalized activations and detection scores
        if len(normalized_activations) > 1:
            correlation = np.corrcoef(normalized_activations, all_detection_scores)[0, 1]
        else:
            correlation = np.nan
            
        # 2. Root Mean Square Calibration Error (RMSCE)
        rmsce = 0.0
        for bm in bucket_metrics:
            rmsce += bm['weight'] * (bm['calibration_error'] ** 2)
        rmsce = np.sqrt(rmsce)
        
        # 3. Maximum Calibration Error (MCE)
        mce = max([bm['calibration_error'] for bm in bucket_metrics]) if bucket_metrics else 0.0
        
        calibration_results[sae_pos] = {
            'ece': ece,
            'rmsce': rmsce,
            'mce': mce,
            'correlation': correlation,
            'n_neurons': len(calibration_entries),
            'bucket_metrics': bucket_metrics
        }
        
        # Print summary
        print(f"\nCalibration Metrics for {sae_pos}:")
        print(f"  Expected Calibration Error (ECE): {ece:.4f}")
        print(f"  Root Mean Square Calibration Error (RMSCE): {rmsce:.4f}")
        print(f"  Maximum Calibration Error (MCE): {mce:.4f}")
        print(f"  Correlation (activation vs detection): {correlation:.4f}")
        print(f"\n  Per-bucket breakdown:")
        for bm in bucket_metrics:
            print(f"    Bucket {bm['bucket_idx']} ({bm['percentile_range']}): "
                  f"n={bm['n_samples']:3d}, "
                  f"act={bm['avg_normalized_activation']:.3f}, "
                  f"det={bm['avg_detection_score']:.3f}, "
                  f"err={bm['calibration_error']:.3f}")
    
    return calibration_results


def plot_calibration_diagram(calibration_results: dict, output_path: str):
    """Create a calibration diagram showing actual vs expected scores."""
    output_dir = Path(output_path) / "calibration_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for sae_pos, results in calibration_results.items():
        bucket_metrics = results['bucket_metrics']
        if not bucket_metrics:
            continue
            
        # Create calibration diagram
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Extract data
        expected = [bm['avg_normalized_activation'] for bm in bucket_metrics]
        actual = [bm['avg_detection_score'] for bm in bucket_metrics]
        weights = [bm['n_samples'] for bm in bucket_metrics]
        
        # Plot calibration line
        ax.scatter(expected, actual, s=[w*5 for w in weights], alpha=0.6, label='Buckets')
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Add bucket labels
        for i, bm in enumerate(bucket_metrics):
            ax.annotate(f"B{bm['bucket_idx']}", (expected[i], actual[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Expected Score (Normalized Activation)', fontsize=12)
        ax.set_ylabel('Actual Score (Detection)', fontsize=12)
        ax.set_title(f'Calibration Diagram - {sae_pos}\nECE={results["ece"]:.4f}, Corr={results["correlation"]:.4f}', fontsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'calibration_diagram_{sae_pos.replace(".", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"\nCalibration diagrams saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compute calibration metrics from saved explanation scores")
    parser.add_argument("--scores_file", type=str, required=True,
                       help="Path to the explanation_scores JSON file")
    parser.add_argument("--output_path", type=str, default="./artifacts",
                       help="Base output path for plots (default: ./artifacts)")
    
    args = parser.parse_args()
    
    # Load the saved scores
    print(f"Loading scores from: {args.scores_file}")
    with open(args.scores_file, 'r') as f:
        all_explanation_scores = json.load(f)
    
    # Compute calibration metrics
    calibration_results = compute_calibration_metrics(all_explanation_scores)
    
    # Save results
    results_path = Path(args.output_path) / "calibration_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    print(f"\nCalibration metrics saved to: {results_path}")
    
    # Create calibration diagrams
    plot_calibration_diagram(calibration_results, args.output_path)


if __name__ == "__main__":
    main() 