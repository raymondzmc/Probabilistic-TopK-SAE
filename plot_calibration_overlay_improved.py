#!/usr/bin/env python3
"""
Improved overlay calibration curves from multiple SAE models with confidence intervals.
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
        'bucket_raw_scores': bucket_scores  # Return raw scores for statistical tests
    }


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
    
    labels = {
        'relu': 'ReLU',
        'gated': 'Gated',
        'topk': 'TopK',
        'probabilistic': 'Probabilistic TopK',
    }
    
    # Model configurations with paths
    models = {
        'topk': 'artifacts/calibration_gpt2_topk_k8_multi',
        'probabilistic': 'artifacts/calibration_gpt2_probabilistic_k8_multi',
        'gated': 'artifacts/calibration_gpt2_gated_9e-02_multi',
        'relu': 'artifacts/calibration_gpt2_relu_30_multi',
    }
    
    # Set style for academic presentation
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Define horizontal offsets for staggering (dodge) - increased for clarity
    offsets = {
        'relu': -3.0,
        'gated': -1.0,
        'topk': 1.0,
        'probabilistic': 3.0,
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
                   linewidth=2,
                   alpha=0.7,
                   zorder=1)  # Behind points
            
            # Then plot points with error bars (on top)
            ax.errorbar(x_positions, 
                       data['detection_scores'],
                       yerr=data['detection_cis'],
                       color=colors[model_key],
                       marker=markers[model_key],
                       markersize=12,
                       linestyle='none',  # No line from errorbar
                       capsize=5,
                       capthick=2,
                       label=f"{labels[model_key]} (r={data['correlation']:.3f})",
                       alpha=0.9,
                       zorder=2)  # On top
            
            plotted_models.append(model_key)
            
        except Exception as e:
            print(f"Warning: Could not load data for {model_key}: {e}")
            continue
    
    # Add perfect calibration line (behind data)
    ax.plot([0, 100], [0, 1], 'k--', alpha=0.3, linewidth=2, 
            label='Perfect calibration', zorder=0)
    
    # Customize plot with improved formatting
    ax.set_xlabel('Activation Percentile', fontsize=16, labelpad=10)
    ax.set_ylabel('Interpretability Score', fontsize=16, labelpad=10)
    ax.set_title('Automatic Interpretability by Activation Percentile', 
                fontsize=18, fontweight='bold', pad=15)
    
    # Set axis limits with some padding (adjusted for increased staggering)
    ax.set_xlim(-10, 110)
    ax.set_ylim(-0.05, 1.05)
    
    # Improve grid appearance
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Customize legend - order by correlation
    handles, labels_list = ax.get_legend_handles_labels()
    # Separate perfect calibration line
    perfect_cal_idx = labels_list.index('Perfect calibration')
    perfect_cal_handle = handles.pop(perfect_cal_idx)
    perfect_cal_label = labels_list.pop(perfect_cal_idx)
    
    # Sort remaining by correlation value (extract from label)
    sorted_indices = sorted(range(len(labels_list)), 
                          key=lambda i: float(labels_list[i].split('r=')[1].rstrip(')')), 
                          reverse=True)
    
    sorted_handles = [handles[i] for i in sorted_indices] + [perfect_cal_handle]
    sorted_labels = [labels_list[i] for i in sorted_indices] + [perfect_cal_label]
    
    ax.legend(sorted_handles, sorted_labels, 
             loc='upper left', 
             fontsize=13, 
             framealpha=0.95,
             frameon=True,
             fancybox=True,
             shadow=True)
    
    # Add subtle background shading for bucket regions
    for i in range(5):
        ax.axvspan(i*20, (i+1)*20, alpha=0.02, color='gray', zorder=0)
    
    # Improve tick formatting
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add minor gridlines for better readability
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
    
    # Save the plot with high quality
    output_dir = Path('artifacts')
    output_path = output_dir / 'calibration_overlay_improved.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'calibration_overlay_improved.svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'calibration_overlay_improved.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"Improved overlay plot saved to: {output_path}")
    
    # Print summary statistics
    print("\nModel Comparison (sorted by correlation):")
    print("-" * 60)
    print(f"{'Model':<25} {'Correlation':<15} {'Mean Score':<15}")
    print("-" * 60)
    
    # Collect and sort models by correlation
    model_stats = []
    for model_key, model_path in models.items():
        try:
            data = load_calibration_data_with_confidence(model_path)
            mean_score = np.nanmean(data['detection_scores'])
            model_stats.append((labels[model_key], data['correlation'], mean_score))
        except:
            pass
    
    # Sort by correlation
    model_stats.sort(key=lambda x: x[1], reverse=True)
    
    for label, corr, mean_score in model_stats:
        print(f"{label:<25} {corr:<15.4f} {mean_score:<15.3f}")
    
    print("\nNote: Error bars show 95% confidence intervals")
    print("Points are horizontally staggered for clarity")
    
    # Statistical analysis of top 20% bucket (bucket 4)
    print("\n" + "="*60)
    print("TOP 20% ACTIVATION ANALYSIS (80-100 percentile)")
    print("="*60)
    
    # Collect top bucket scores for each model
    top_bucket_scores = {}
    for model_key, model_path in models.items():
        try:
            data = load_calibration_data_with_confidence(model_path)
            # Bucket 4 is the top 20% (80-100 percentile)
            top_bucket_scores[model_key] = data['bucket_raw_scores'][4]
        except:
            pass
    
    # Print mean scores for top bucket
    print("\nInterpretability scores for top 20% activations:")
    print("-" * 60)
    
    model_means = []
    for model_key in ['probabilistic', 'topk', 'relu', 'gated']:  # Order by expected performance
        if model_key in top_bucket_scores:
            scores = top_bucket_scores[model_key]
            mean = np.mean(scores)
            std = np.std(scores)
            sem = std / np.sqrt(len(scores))
            ci = stats.t.ppf(0.975, len(scores)-1) * sem
            model_means.append((model_key, mean, ci, scores))
            print(f"{labels[model_key]:<20} {mean:.3f} ± {ci:.3f} (95% CI)")
    
    # Perform pairwise statistical tests
    print("\nStatistical Significance (pairwise t-tests):")
    print("-" * 60)
    
    # Compare all pairs
    for i in range(len(model_means)):
        for j in range(i+1, len(model_means)):
            model1_key, mean1, _, scores1 = model_means[i]
            model2_key, mean2, _, scores2 = model_means[j]
            
            # Perform independent samples t-test
            t_stat, p_value = stats.ttest_ind(scores1, scores2, equal_var=False)
            
            # Cohen's d effect size
            pooled_std = np.sqrt((np.std(scores1)**2 + np.std(scores2)**2) / 2)
            cohens_d = abs(mean1 - mean2) / pooled_std
            
            # Determine significance level
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
            else:
                sig = "ns"
            
            print(f"{labels[model1_key]} vs {labels[model2_key]}: "
                  f"Δ={abs(mean1-mean2):.3f}, p={p_value:.4f} {sig}, d={cohens_d:.2f}")
    
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Effect size (Cohen's d): 0.2=small, 0.5=medium, 0.8=large")


if __name__ == "__main__":
    main() 