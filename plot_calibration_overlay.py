#!/usr/bin/env python3
"""
Overlay calibration curves from multiple SAE models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_calibration_data(base_path):
    """Load calibration metrics from a model directory."""
    metrics_path = Path(base_path) / "calibration_metrics.json"
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Calibration metrics not found at {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Extract bucket data for the first (and only) SAE position
    sae_pos = list(metrics.keys())[0]
    bucket_metrics = metrics[sae_pos]['bucket_metrics']
    
    # Extract data for plotting
    bucket_midpoints = []
    detection_means = []
    
    for bm in bucket_metrics:
        # Calculate bucket midpoint from percentile range
        percentile_range = bm['percentile_range']
        start, end = map(lambda x: int(x.strip('%')), percentile_range.split('-'))
        midpoint = (start + end) / 2
        bucket_midpoints.append(midpoint)
        detection_means.append(bm['avg_detection_score'])
    
    return {
        'midpoints': bucket_midpoints,
        'detection_scores': detection_means,
        'ece': metrics[sae_pos]['ece'],
        'correlation': metrics[sae_pos]['correlation']
    }


def main():
    # Model configurations
    models = {
        'TopK': {
            'path': 'artifacts/calibration_gpt2_topk_k8_multi',
            'color': 'blue',
            'marker': 'o',
            'label': 'TopK (k=8)'
        },
        'Probabilistic': {
            'path': 'artifacts/calibration_gpt2_probabilistic_k8_multi',
            'color': 'red',
            'marker': 's',
            'label': 'Probabilistic HC-TopK'
        },
        'Gated': {
            'path': 'artifacts/calibration_gpt2_gated_9e-02_multi',
            'color': 'green',
            'marker': '^',
            'label': 'Gated (λ=0.09)'
        },
        'ReLU': {
            'path': 'artifacts/calibration_gpt2_relu_30_multi',
            'color': 'orange',
            'marker': 'D',
            'label': 'ReLU (λ=30)'
        }
    }
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Load and plot data for each model
    for model_name, config in models.items():
        try:
            data = load_calibration_data(config['path'])
            
            # Plot the calibration curve
            ax.plot(data['midpoints'], data['detection_scores'], 
                   color=config['color'], 
                   marker=config['marker'], 
                   markersize=10, 
                   linewidth=2.5,
                   label=f"{config['label']} (ECE={data['ece']:.3f})")
            
            # Add error bars if we had them (we don't currently store std devs)
            # ax.errorbar(data['midpoints'], data['detection_scores'], yerr=data['stds'], ...)
            
        except FileNotFoundError as e:
            print(f"Warning: Could not load data for {model_name}: {e}")
            continue
    
    # Add perfect calibration line
    ax.plot([0, 100], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect calibration')
    
    # Customize plot
    ax.set_xlabel('Activation Percentile', fontsize=14)
    ax.set_ylabel('Detection Score', fontsize=14)
    ax.set_title('Automatic Interpretability by Activation Percentile', fontsize=16, fontweight='bold')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Add text box with key insights
    textstr = 'Lower ECE = Better Calibration\nHigher Correlation = Stronger Relationship'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Save the plot
    output_dir = Path('artifacts')
    output_path = output_dir / 'calibration_overlay_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'calibration_overlay_comparison.svg', bbox_inches='tight')
    
    print(f"Overlay plot saved to: {output_path}")
    
    # Print summary statistics
    print("\nModel Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'ECE':<10} {'Correlation':<15}")
    print("-" * 60)
    
    for model_name, config in models.items():
        try:
            data = load_calibration_data(config['path'])
            print(f"{model_name:<20} {data['ece']:<10.4f} {data['correlation']:<15.4f}")
        except:
            print(f"{model_name:<20} {'N/A':<10} {'N/A':<15}")
    
    print("\nRegarding ECE vs Correlation:")
    print("- ECE (Expected Calibration Error) is more relevant for calibration assessment")
    print("- ECE measures how well activation strength predicts interpretability")
    print("- Lower ECE means you can better trust activation magnitudes as indicators")
    print("- Correlation shows relationship strength but not calibration quality")


if __name__ == "__main__":
    main() 