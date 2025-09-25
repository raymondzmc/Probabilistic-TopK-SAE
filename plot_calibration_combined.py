#!/usr/bin/env python3
"""
Combined calibration curves figure with two subplots side-by-side.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


def load_calibration_data_with_confidence(base_path):
    """Load calibration data including raw scores for confidence intervals."""
    metrics_path = Path(base_path) / "calibration_metrics.json"
    
    # Find the explanation scores file
    scores_files = list(Path(base_path).glob("explanation_scores_*.json"))
    if not scores_files:
        raise FileNotFoundError(f"No explanation scores found in {base_path}")
    
    scores_path = scores_files[0]
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load raw scores for confidence intervals
    with open(scores_path, 'r') as f:
        all_scores = json.load(f)
    
    sae_pos = list(metrics.keys())[0]
    scores_list = all_scores[sae_pos]
    
    # Collect scores by bucket
    n_buckets = 5
    bucket_scores = {i: [] for i in range(n_buckets)}
    
    for entry in scores_list:
        if 'bucket_detection_scores' in entry:
            for bucket_idx, score in entry['bucket_detection_scores'].items():
                bucket_idx = int(bucket_idx) if isinstance(bucket_idx, str) else bucket_idx
                bucket_scores[bucket_idx].append(score)
    
    # Calculate statistics for each bucket
    bucket_midpoints = []
    detection_means = []
    detection_stds = []
    detection_sems = []
    detection_cis = []
    
    for i in range(n_buckets):
        # Calculate bucket midpoint
        start = i * 20
        end = (i + 1) * 20
        midpoint = (start + end) / 2
        bucket_midpoints.append(midpoint)
        
        # Calculate statistics
        scores = bucket_scores[i]
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            n = len(scores)
            sem = std / np.sqrt(n)
            # 95% confidence interval
            ci = stats.t.ppf(0.975, n-1) * sem
            
            detection_means.append(mean)
            detection_stds.append(std)
            detection_sems.append(sem)
            detection_cis.append(ci)
        else:
            detection_means.append(np.nan)
            detection_stds.append(np.nan)
            detection_sems.append(np.nan)
            detection_cis.append(np.nan)
    
    return {
        'midpoints': bucket_midpoints,
        'detection_scores': detection_means,
        'detection_cis': detection_cis,
        'correlation': metrics[sae_pos]['correlation'],
        'bucket_raw_scores': bucket_scores
    }


def plot_subplot(ax, models, colors, markers, labels, title_suffix=""):
    """Plot calibration curves on a given subplot."""
    
    # Define horizontal offsets for staggering (dodge) - increased for clarity
    offsets = {
        'relu': -5.0,
        'gated': -1.7,
        'topk': 1.7,
        'probabilistic': 5.0,
    }
    
    # Track plotted models for ordering legend
    plotted_models = []
    
    # Load and plot data for each model
    for model_key, model_path in models.items():
        try:
            data = load_calibration_data_with_confidence(model_path)
            
            # Apply horizontal offset for better readability
            x_positions = np.array(data['midpoints']) + offsets[model_key]
            
            # First, plot the connecting line (dotted, behind points)
            ax.plot(x_positions, 
                   data['detection_scores'],
                   color=colors[model_key],
                   linestyle=':',
                   linewidth=3,
                   alpha=0.7,
                   zorder=1)  # Behind points
            
            # Then plot points with error bars (on top)
            ax.errorbar(x_positions, 
                       data['detection_scores'],
                       yerr=data['detection_cis'],
                       color=colors[model_key],
                       marker=markers[model_key],
                       markersize=18,
                       linestyle='none',  # No line from errorbar
                       capsize=8,
                       capthick=3,
                       label=labels[model_key](data['correlation']),
                       alpha=0.9,
                       zorder=2)  # On top
            
            plotted_models.append(model_key)
            
        except Exception as e:
            print(f"Warning: Could not load data for {model_key}: {e}")
            continue
    
    # Add perfect calibration line (behind data)
    ax.plot([0, 100], [0, 1], 'k--', alpha=0.3, linewidth=3, 
            label='Perfect calibration', zorder=0)
    
    # Customize subplot
    ax.set_xlabel('Activation Percentile', fontsize=24, labelpad=14)
    ax.set_ylabel('Interpretability Score', fontsize=24, labelpad=14)
    
    # Set axis limits with minimal padding
    ax.set_xlim(-1, 101)
    ax.set_ylim(-0.02, 1.02)
    
    # Improve grid appearance
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Add minor gridlines for better readability
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.1, linestyle=':', linewidth=0.6)
    
    # Add subtle background shading for bucket regions
    for i in range(5):
        ax.axvspan(i*20, (i+1)*20, alpha=0.02, color='gray', zorder=0)
    
    # Improve tick formatting
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Add subplot label
    ax.text(0.02, 0.98, title_suffix, transform=ax.transAxes, 
            fontsize=22, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    return ax


def main():
    # Define colors and markers according to specification
    colors = {
        'relu': '#ff7f0e',        # Orange
        'gated': '#2ca02c',       # Green
        'topk': '#1f77b4',        # Blue
        'probabilistic': '#d62728', # Red
    }
    
    markers = {
        'relu': 's',              # Square
        'gated': '^',             # Triangle up
        'topk': 'o',              # Circle
        'probabilistic': 'D',     # Diamond
    }
    
    # Label functions that return formatted strings with LaTeX
    def relu_label(r, lam):
        return f'ReLU ($r={r:.3f}$)'
    
    def gated_label(r, lam):
        return f'Gated ($r={r:.3f}$)'
    
    def topk_label(r, k):
        return f'TopK ($r={r:.3f}$)'
    
    def prob_label(r, k):
        return f'Probabilistic TopK ($r={r:.3f}$)'
    
    # Model configurations for first set
    models_set1 = {
        'topk': 'artifacts/calibration_gpt2_topk_k8_multi',
        'probabilistic': 'artifacts/calibration_gpt2_probabilistic_k8_multi',
        'gated': 'artifacts/calibration_gpt2_gated_9e-02_multi',
        'relu': 'artifacts/calibration_gpt2_relu_30_multi',
    }
    
    labels_set1 = {
        'relu': lambda r: relu_label(r, 30),
        'gated': lambda r: gated_label(r, 0.09),
        'topk': lambda r: topk_label(r, 8),
        'probabilistic': lambda r: prob_label(r, 8),
    }
    
    # Model configurations for second set
    models_set2 = {
        'relu': 'artifacts/calibration_gpt2_relu_20_multi',
        'topk': 'artifacts/calibration_gpt2_topk_k_16_multi',
        'gated': 'artifacts/calibration_gpt2_gated_6e-02_multi',
        'probabilistic': 'artifacts/calibration_gpt2_probabilistic_k_16_multi',
    }
    
    labels_set2 = {
        'relu': lambda r: relu_label(r, 20),
        'gated': lambda r: gated_label(r, 0.06),
        'topk': lambda r: topk_label(r, 16),
        'probabilistic': lambda r: prob_label(r, 16),
    }
    
    # Create figure with two subplots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 11))
    
    # Plot first set
    plot_subplot(ax1, models_set1, colors, markers, labels_set1, "(a)")
    
    # Plot second set
    plot_subplot(ax2, models_set2, colors, markers, labels_set2, "(b)")
    
    # Add overall title
    fig.suptitle('Automatic Interpretability by Activation Percentile ($K=8, 16$)', 
                 fontsize=30, fontweight='bold', y=0.98)
    
    # Customize legends for both subplots
    for ax in [ax1, ax2]:
        handles, labels_list = ax.get_legend_handles_labels()
        
        # Separate perfect calibration line
        perfect_cal_idx = labels_list.index('Perfect calibration')
        perfect_cal_handle = handles.pop(perfect_cal_idx)
        perfect_cal_label = labels_list.pop(perfect_cal_idx)
        
        # Sort remaining by correlation value (extract from label)
        # Extract r value from labels like "ReLU ($\lambda_{\rm{sparsity}}=30$, $r=0.394$)"
        def extract_r_value(label):
            try:
                r_part = label.split('$r=')[1].split('$')[0]
                return float(r_part)
            except:
                return 0.0
        
        sorted_indices = sorted(range(len(labels_list)), 
                              key=lambda i: extract_r_value(labels_list[i]), 
                              reverse=True)
        
        sorted_handles = [handles[i] for i in sorted_indices] + [perfect_cal_handle]
        sorted_labels = [labels_list[i] for i in sorted_indices] + [perfect_cal_label]
        
        ax.legend(sorted_handles, sorted_labels, 
                 loc='upper left', 
                 fontsize=20, 
                 framealpha=0.95,
                 frameon=True,
                 fancybox=True,
                 shadow=True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot with high quality
    output_dir = Path('artifacts')
    output_path = output_dir / 'calibration_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'calibration_combined.svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'calibration_combined.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"Combined calibration plot saved to: {output_path}")
    
    # Print statistical summary for both sets
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    for set_num, (models, set_name) in enumerate([(models_set1, "Set 1 (k=8, λ=30/0.09)"), 
                                                   (models_set2, "Set 2 (k=16, λ=20/0.06)")], 1):
        print(f"\n{set_name}:")
        print("-" * 60)
        
        # Collect top bucket scores
        top_scores = {}
        for model_key, model_path in models.items():
            try:
                data = load_calibration_data_with_confidence(model_path)
                top_scores[model_key] = data['bucket_raw_scores'][4]
            except:
                pass
        
        print("Top 20% Activation Scores:")
        for model_key in ['probabilistic', 'topk', 'relu', 'gated']:
            if model_key in top_scores:
                scores = top_scores[model_key]
                mean = np.mean(scores)
                print(f"  {model_key.title():<20} {mean:.3f}")


if __name__ == "__main__":
    main() 