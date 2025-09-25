#!/usr/bin/env python3
"""
Script to summarize and visualize explanation results from multiple SAE runs.
Creates dot + errorbar plots comparing detection and fuzz scores across runs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from settings import settings
import wandb
from scipy import stats

# Import the utility function for loading explanations
from utils.io import load_explanations_from_wandb


def extract_scores_from_explanations(explanations: Dict) -> Dict[str, List[Dict]]:
    """
    Extract scores from explanation data grouped by layer.
    
    Args:
        explanations: Dictionary with layer keys mapping to lists of explanations
        
    Returns:
        Dictionary mapping layers to list of score dictionaries
    """
    scores_by_layer = {}
    
    for layer, layer_explanations in explanations.items():
        scores_by_layer[layer] = []
        
        for exp in layer_explanations:
            # Extract scores, handling None values
            detection_score = exp.get('detection_score')
            fuzz_score = exp.get('fuzz_score')
            
            # Only include if we have at least one score
            if detection_score is not None or fuzz_score is not None:
                scores_by_layer[layer].append({
                    'neuron_idx': exp.get('neuron_idx'),
                    'detection_score': detection_score,
                    'fuzz_score': fuzz_score
                })
    
    return scores_by_layer


def compute_statistics(scores_list: List[Dict], metric: str) -> Tuple[float, float, int]:
    """
    Compute mean, std, and count for a specific metric.
    
    Args:
        scores_list: List of score dictionaries
        metric: 'detection_score' or 'fuzz_score'
        
    Returns:
        Tuple of (mean, std, count)
    """
    values = [s[metric] for s in scores_list if s[metric] is not None]
    
    if not values:
        return np.nan, np.nan, 0
    
    return np.mean(values), np.std(values), len(values)


def calculate_99_ci(values: List[float]) -> Tuple[float, float]:
    """
    Calculate 99% confidence interval for a list of values.
    
    Args:
        values: List of values
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not values or len(values) < 2:
        return 0, 0
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Use sample standard deviation
    n = len(values)
    
    # 99% CI critical value (two-tailed)
    z_critical = 2.576
    margin_of_error = z_critical * (std / np.sqrt(n))
    
    return margin_of_error, margin_of_error  # Return symmetric error margins


def resample(data: List[float], n_samples: int = None) -> List[float]:
    """Simple bootstrap resampling function."""
    if n_samples is None:
        n_samples = len(data)
    indices = np.random.choice(len(data), size=n_samples, replace=True)
    return [data[i] for i in indices]


def bootstrap_test(group1: List[float], group2: List[float], n_bootstrap: int = 10000, alpha: float = 0.01) -> Dict:
    """
    Perform bootstrap hypothesis test for difference in means.
    
    Args:
        group1: First group of values
        group2: Second group of values
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    observed_diff = np.mean(group1) - np.mean(group2)
    
    # Bootstrap distribution of differences
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample from each group
        sample1 = resample(group1, n_samples=len(group1))
        sample2 = resample(group2, n_samples=len(group2))
        bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
    
    # Calculate p-value (two-tailed)
    # Shift distribution to be centered at 0 (null hypothesis)
    centered_diffs = np.array(bootstrap_diffs) - np.mean(bootstrap_diffs)
    p_value = np.sum(np.abs(centered_diffs) >= np.abs(observed_diff)) / n_bootstrap
    
    # Bootstrap confidence interval
    ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
    
    return {
        'observed_difference': observed_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha
    }


def permutation_test(group1: List[float], group2: List[float], n_permutations: int = 10000, alpha: float = 0.01) -> Dict:
    """
    Perform permutation test for difference in means.
    
    Args:
        group1: First group of values
        group2: Second group of values
        n_permutations: Number of permutations
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    # Generate permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        perm_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))
    
    # Calculate p-value (two-tailed)
    p_value = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / n_permutations
    
    return {
        'observed_difference': observed_diff,
        'p_value': p_value,
        'significant': p_value < alpha
    }


def bayesian_comparison(group1: List[float], group2: List[float], n_samples: int = 10000) -> Dict:
    """
    Perform Bayesian comparison using analytical posterior for normal distributions.
    
    Args:
        group1: First group of values
        group2: Second group of values
        n_samples: Number of posterior samples
        
    Returns:
        Dictionary with Bayesian analysis results
    """
    # Use normal-inverse-gamma conjugate prior
    # Weakly informative priors
    
    # Group 1 statistics
    n1 = len(group1)
    mean1 = np.mean(group1)
    var1 = np.var(group1, ddof=1)
    
    # Group 2 statistics
    n2 = len(group2)
    mean2 = np.mean(group2)
    var2 = np.var(group2, ddof=1)
    
    # Sample from posterior distributions
    # Using t-distribution as posterior for difference in means
    df = n1 + n2 - 2
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
    se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))
    
    # Sample from posterior
    posterior_samples = stats.t.rvs(df, loc=mean1-mean2, scale=se_diff, size=n_samples)
    
    # Calculate probability that group1 > group2
    prob_greater = np.mean(posterior_samples > 0)
    
    # 99% credible interval
    credible_lower = np.percentile(posterior_samples, 0.5)
    credible_upper = np.percentile(posterior_samples, 99.5)
    
    # Bayes factor approximation (using Savage-Dickey ratio)
    # BF01 = Posterior at 0 / Prior at 0 (for null hypothesis)
    prior_at_zero = stats.norm.pdf(0, 0, 10)  # Wide prior
    posterior_at_zero = stats.t.pdf(0, df, loc=mean1-mean2, scale=se_diff)
    # BF10 = 1 / BF01 (for alternative hypothesis)
    bayes_factor_10 = prior_at_zero / posterior_at_zero
    
    return {
        'mean_difference': mean1 - mean2,
        'probability_greater': prob_greater,
        'credible_interval': (credible_lower, credible_upper),
        'bayes_factor': bayes_factor_10,
        'evidence_strength': interpret_bayes_factor_10(bayes_factor_10)
    }


def interpret_bayes_factor_10(bf10: float) -> str:
    """Interpret Bayes factor (BF10) for alternative hypothesis according to Jeffreys' scale."""
    if bf10 > 100:
        return "Decisive evidence against null"
    elif bf10 > 30:
        return "Very strong evidence against null"
    elif bf10 > 10:
        return "Strong evidence against null"
    elif bf10 > 3:
        return "Moderate evidence against null"
    elif bf10 > 1:
        return "Weak evidence against null"
    elif bf10 > 1/3:
        return "No evidence either way"
    elif bf10 > 1/10:
        return "Weak evidence for null"
    elif bf10 > 1/30:
        return "Moderate evidence for null"
    elif bf10 > 1/100:
        return "Strong evidence for null"
    else:
        return "Decisive evidence for null"


def compute_cliffs_delta(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cliff's Delta, a non-parametric effect size measure.
    
    Args:
        group1: First group of values
        group2: Second group of values
        
    Returns:
        Cliff's Delta value between -1 and 1
    """
    n1 = len(group1)
    n2 = len(group2)
    
    # Count dominances
    dominance_count = 0
    for val1 in group1:
        for val2 in group2:
            if val1 > val2:
                dominance_count += 1
            elif val1 < val2:
                dominance_count -= 1
    
    # Cliff's Delta
    delta = dominance_count / (n1 * n2)
    return delta


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's Delta effect size."""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def apply_multiple_comparisons_correction(p_values: List[float], method: str = 'bonferroni') -> Dict:
    """
    Apply multiple comparisons correction.
    
    Args:
        p_values: List of p-values to correct
        method: 'bonferroni' or 'fdr_bh' (Benjamini-Hochberg)
        
    Returns:
        Dictionary with corrected p-values and significance results
    """
    n = len(p_values)
    
    if method == 'bonferroni':
        # Bonferroni correction
        corrected = [min(p * n, 1.0) for p in p_values]
        significant = [p < 0.01 for p in corrected]
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR correction
        # Sort p-values and their indices
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        # Apply BH correction
        corrected = np.zeros_like(sorted_p)
        for i in range(n):
            corrected[i] = sorted_p[i] * n / (i + 1)
        
        # Ensure monotonicity
        for i in range(n-2, -1, -1):
            corrected[i] = min(corrected[i], corrected[i+1])
        
        # Cap at 1
        corrected = np.minimum(corrected, 1.0)
        
        # Restore original order
        final_corrected = np.zeros(n)
        final_corrected[sorted_indices] = corrected
        corrected = final_corrected.tolist()
        
        # Determine significance
        significant = [p < 0.01 for p in corrected]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'method': method,
        'corrected_p_values': corrected,
        'significant_at_0.01': significant
    }


def perform_statistical_tests(run_data: Dict[str, Dict], baseline_run: str = "probabilistic_k_8") -> Dict[str, Dict]:
    """
    Perform statistical significance tests comparing baseline run to others.
    
    Args:
        run_data: Dictionary mapping run names to their score data
        baseline_run: Name of the baseline run to compare against
        
    Returns:
        Dictionary of test results
    """
    results = {}
    
    if baseline_run not in run_data:
        print(f"Warning: Baseline run '{baseline_run}' not found in data")
        return results
    
    # Get baseline scores
    baseline_scores = {}
    for layer, scores_list in run_data[baseline_run].items():
        baseline_scores[layer] = {
            'detection': [s['detection_score'] for s in scores_list if s['detection_score'] is not None]
        }
    
    # Compare baseline to other runs
    for run_name, scores_by_layer in run_data.items():
        if run_name == baseline_run:
            continue
            
        results[run_name] = {}
        
        for layer in scores_by_layer:
            if layer not in baseline_scores:
                continue
                
            # Get comparison scores
            comparison_scores = {
                'detection': [s['detection_score'] for s in scores_by_layer[layer] if s['detection_score'] is not None]
            }
            
            layer_results = {}
            
            # Perform t-tests for detection metric only
            metric = 'detection'
            baseline_vals = baseline_scores[layer][metric]
            comparison_vals = comparison_scores[metric]
            
            if len(baseline_vals) > 1 and len(comparison_vals) > 1:
                # 1. Two-sample t-test (assuming unequal variances)
                t_stat, t_pvalue = stats.ttest_ind(baseline_vals, comparison_vals, equal_var=False)
                
                # 2. Mann-Whitney U test (non-parametric)
                u_stat, u_pvalue = stats.mannwhitneyu(baseline_vals, comparison_vals, alternative='two-sided')
                
                # 3. Bootstrap test
                bootstrap_results = bootstrap_test(baseline_vals, comparison_vals, n_bootstrap=5000)
                
                # 4. Permutation test
                permutation_results = permutation_test(baseline_vals, comparison_vals, n_permutations=5000)
                
                # 5. Bayesian comparison
                bayesian_results = bayesian_comparison(baseline_vals, comparison_vals)
                
                # Calculate effect sizes
                pooled_std = np.sqrt((np.var(baseline_vals, ddof=1) + np.var(comparison_vals, ddof=1)) / 2)
                cohens_d = (np.mean(baseline_vals) - np.mean(comparison_vals)) / pooled_std if pooled_std > 0 else 0
                
                # Cliff's Delta (non-parametric effect size)
                cliffs_delta = compute_cliffs_delta(baseline_vals, comparison_vals)
                
                layer_results[metric] = {
                    'baseline_mean': float(np.mean(baseline_vals)),
                    'comparison_mean': float(np.mean(comparison_vals)),
                    'baseline_better': bool(np.mean(baseline_vals) > np.mean(comparison_vals)),
                    
                    # T-test results
                    't_test': {
                        't_statistic': float(t_stat),
                        'p_value': float(t_pvalue),
                        'significant_at_0.01': bool(t_pvalue < 0.01)
                    },
                    
                    # Mann-Whitney U test results
                    'mann_whitney': {
                        'u_statistic': float(u_stat),
                        'p_value': float(u_pvalue),
                        'significant_at_0.01': bool(u_pvalue < 0.01)
                    },
                    
                    # Bootstrap results
                    'bootstrap': {
                        'p_value': float(bootstrap_results['p_value']),
                        'ci_lower': float(bootstrap_results['ci_lower']),
                        'ci_upper': float(bootstrap_results['ci_upper']),
                        'significant_at_0.01': bool(bootstrap_results['significant'])
                    },
                    
                    # Permutation test results
                    'permutation': {
                        'p_value': float(permutation_results['p_value']),
                        'significant_at_0.01': bool(permutation_results['significant'])
                    },
                    
                    # Bayesian results
                    'bayesian': {
                        'probability_better': float(bayesian_results['probability_greater']),
                        'credible_interval': [float(x) for x in bayesian_results['credible_interval']],
                        'bayes_factor': float(bayesian_results['bayes_factor']),
                        'evidence': bayesian_results['evidence_strength']
                    },
                    
                    # Effect sizes
                    'effect_sizes': {
                        'cohens_d': float(cohens_d),
                        'cliffs_delta': float(cliffs_delta)
                    }
                }
            else:
                layer_results[metric] = {
                    'error': 'Insufficient data for statistical test'
                }
            
            results[run_name][layer] = layer_results
    
    return results


def create_comparison_plot(run_data: Dict[str, Dict], output_path: str = "./plots"):
    """
    Create grouped bar plots with 99% confidence intervals comparing detection and fuzz scores.
    
    Args:
        run_data: Dictionary mapping run names to their score data
        output_path: Directory to save plots
    """
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Since all data is from one layer, we'll create a single grouped bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors and metrics
    colors = {'detection_score': '#1f77b4'}  # Blue for detection
    metrics = ['detection_score']
    metric_labels = {'detection_score': 'Detection Score'}
    
    # Run names for x-axis
    run_names = list(run_data.keys())
    run_labels = {
        'probabilistic_k_8': 'Probabilistic\n(k=8)',
        'topk_k_8_interpret': 'Top-k\n(k=8)',
        'gated_sparsity_coeff_0.09': 'Gated\n(λ=0.09)'
    }
    
    # Prepare data
    x = np.arange(len(run_names))
    width = 0.6  # Wider bars since we only have one metric
    
    # Collect means and confidence intervals
    means = {metric: [] for metric in metrics}
    ci_errors = {metric: [] for metric in metrics}
    
    for run_name in run_names:
        # Assuming single layer data
        layer_data = list(run_data[run_name].values())[0]
        
        for metric in metrics:
            values = [s[metric] for s in layer_data if s[metric] is not None]
            
            if values:
                mean_val = np.mean(values)
                ci_lower, ci_upper = calculate_99_ci(values)
                
                # Cap at 1.0
                means[metric].append(min(mean_val, 1.0))
                # Ensure error bar doesn't exceed 1.0
                ci_errors[metric].append(min(ci_upper, 1.0 - mean_val))
            else:
                means[metric].append(0)
                ci_errors[metric].append(0)
    
    # Create bars (single metric, no grouping needed)
    metric = metrics[0]  # Only detection_score
    bars = ax.bar(x, means[metric], width, 
                  yerr=ci_errors[metric], 
                  label=metric_labels[metric],
                  color=colors[metric], 
                  alpha=0.8,
                  capsize=5,
                  error_kw={'linewidth': 2, 'capthick': 2})
    
    # Add value labels on bars
    for j, (bar, mean) in enumerate(zip(bars, means[metric])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Customize plot
    ax.set_xlabel('SAE Architecture', fontsize=14, fontweight='bold')
    ax.set_ylabel('Detection Score', fontsize=14, fontweight='bold')
    ax.set_title('SAE Explanation Quality: Detection Scores\n(99% Confidence Intervals)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([run_labels.get(name, name) for name in run_names], fontsize=12)
    # Remove legend since we only have one metric
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Set y-axis limits starting at 0.5
    ax.set_ylim(0.5, 1.05)
    
    # Add horizontal line at y=1.0 for reference
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Add note about statistical significance
    ax.text(0.02, 0.98, '99% CI shown', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / 'grouped_bar_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved grouped bar plot to {output_file}")
    
    # Perform statistical tests
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*80)
    print("Comparing Probabilistic (baseline) vs other methods")
    print("Using multiple statistical approaches (α = 0.01)")
    
    test_results = perform_statistical_tests(run_data, baseline_run="probabilistic_k_8")
    
    # Collect p-values for multiple comparisons correction
    all_p_values_t = []
    all_p_values_mw = []
    test_labels = []
    
    for run_name, layer_results in test_results.items():
        print(f"\n{'='*60}")
        print(f"{run_name}")
        print(f"{'='*60}")
        
        for layer, metrics_results in layer_results.items():
            for metric_name, results in metrics_results.items():
                if 'error' in results:
                    print(f"  {metric_name}: {results['error']}")
                else:
                    test_labels.append(f"{run_name}_{metric_name}")
                    all_p_values_t.append(results['t_test']['p_value'])
                    all_p_values_mw.append(results['mann_whitney']['p_value'])
                    
                    print(f"\n{metric_name.upper()}:")
                    print(f"  Means: Probabilistic={results['baseline_mean']:.3f}, {run_name}={results['comparison_mean']:.3f}")
                    print(f"  Difference: {results['baseline_mean'] - results['comparison_mean']:.3f}")
                    
                    # Traditional tests
                    print(f"\n  Frequentist Tests:")
                    print(f"    T-test: t={results['t_test']['t_statistic']:.3f}, p={results['t_test']['p_value']:.4f} {'*' if results['t_test']['significant_at_0.01'] else ''}")
                    print(f"    Mann-Whitney U: U={results['mann_whitney']['u_statistic']:.1f}, p={results['mann_whitney']['p_value']:.4f} {'*' if results['mann_whitney']['significant_at_0.01'] else ''}")
                    
                    # Resampling methods
                    print(f"\n  Resampling Methods:")
                    print(f"    Bootstrap: p={results['bootstrap']['p_value']:.4f} {'*' if results['bootstrap']['significant_at_0.01'] else ''}")
                    print(f"      99% CI of difference: [{results['bootstrap']['ci_lower']:.3f}, {results['bootstrap']['ci_upper']:.3f}]")
                    print(f"    Permutation: p={results['permutation']['p_value']:.4f} {'*' if results['permutation']['significant_at_0.01'] else ''}")
                    
                    # Bayesian analysis
                    print(f"\n  Bayesian Analysis:")
                    print(f"    P(Probabilistic > {run_name}): {results['bayesian']['probability_better']:.3f}")
                    print(f"    99% Credible Interval: [{results['bayesian']['credible_interval'][0]:.3f}, {results['bayesian']['credible_interval'][1]:.3f}]")
                    print(f"    Bayes Factor (BF10): {results['bayesian']['bayes_factor']:.2f} ({results['bayesian']['evidence']})")
                    
                    # Effect sizes
                    print(f"\n  Effect Sizes:")
                    print(f"    Cohen's d: {results['effect_sizes']['cohens_d']:.3f} ({interpret_cohens_d(results['effect_sizes']['cohens_d'])})")
                    print(f"    Cliff's Delta: {results['effect_sizes']['cliffs_delta']:.3f} ({interpret_cliffs_delta(results['effect_sizes']['cliffs_delta'])})")
    
    # Multiple comparisons correction
    if all_p_values_t:
        print(f"\n{'='*80}")
        print("MULTIPLE COMPARISONS CORRECTION")
        print(f"{'='*80}")
        
        # Apply corrections
        bonferroni_t = apply_multiple_comparisons_correction(all_p_values_t, 'bonferroni')
        fdr_t = apply_multiple_comparisons_correction(all_p_values_t, 'fdr_bh')
        
        print("\nT-test p-values after correction:")
        print(f"{'Test':<30} {'Original':<10} {'Bonferroni':<15} {'FDR (BH)':<15}")
        print("-" * 70)
        
        for i, label in enumerate(test_labels):
            print(f"{label:<30} {all_p_values_t[i]:<10.4f} "
                  f"{bonferroni_t['corrected_p_values'][i]:<15.4f} "
                  f"{'*' if bonferroni_t['significant_at_0.01'][i] else ' '} "
                  f"{fdr_t['corrected_p_values'][i]:<15.4f} "
                  f"{'*' if fdr_t['significant_at_0.01'][i] else ' '}")
    
    print("\n* = significant at α=0.01 after correction")
    
    # Save statistical test results
    stats_file = output_dir / 'statistical_test_results.json'
    with open(stats_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nStatistical test results saved to {stats_file}")


def print_summary_statistics(run_data: Dict[str, Dict]):
    """Print detailed summary statistics for each run."""
    
    print("\n" + "="*80)
    print("EXPLANATION SCORES SUMMARY")
    print("="*80)
    
    for run_name, scores_by_layer in run_data.items():
        print(f"\n{run_name}")
        print("-" * len(run_name))
        
        # Overall statistics
        all_detection_scores = []
        all_fuzz_scores = []
        
        for layer, scores_list in scores_by_layer.items():
            detection_scores = [s['detection_score'] for s in scores_list if s['detection_score'] is not None]
            fuzz_scores = [s['fuzz_score'] for s in scores_list if s['fuzz_score'] is not None]
            
            all_detection_scores.extend(detection_scores)
            all_fuzz_scores.extend(fuzz_scores)
        
        print(f"\nOverall Statistics:")
        print(f"  Total neurons explained: {sum(len(scores) for scores in scores_by_layer.values())}")
        if all_detection_scores:
            print(f"  Detection Score: mean={np.mean(all_detection_scores):.3f}, "
                  f"std={np.std(all_detection_scores):.3f}, n={len(all_detection_scores)}")
        if all_fuzz_scores:
            print(f"  Fuzz Score: mean={np.mean(all_fuzz_scores):.3f}, "
                  f"std={np.std(all_fuzz_scores):.3f}, n={len(all_fuzz_scores)}")
        
        # Per-layer statistics
        print(f"\nPer-Layer Statistics:")
        for layer in sorted(scores_by_layer.keys()):
            scores_list = scores_by_layer[layer]
            detection_mean, detection_std, detection_count = compute_statistics(scores_list, 'detection_score')
            fuzz_mean, fuzz_std, fuzz_count = compute_statistics(scores_list, 'fuzz_score')
            
            print(f"  {layer}:")
            print(f"    Neurons: {len(scores_list)}")
            if detection_count > 0:
                print(f"    Detection: mean={detection_mean:.3f}, std={detection_std:.3f}, n={detection_count}")
            if fuzz_count > 0:
                print(f"    Fuzz: mean={fuzz_mean:.3f}, std={fuzz_std:.3f}, n={fuzz_count}")


def main():
    """Main function to load and visualize explanation results."""
    
    # Define the runs to analyze
    # Format: (run_name, run_id, version)
    # Version can be None (latest), int (e.g., 2), or string (e.g., "v2" or "latest")
    runs = [
        ("probabilistic_k_8", "ahss4wx6", 'v3'),  # Use latest version
        ("topk_k_8_interpret", "5t2docn3", 'v3'),  # Use latest version
        ("gated_sparsity_coeff_0.09", "4mfj7rqk", 'v4')  # Use latest version
        # Examples with specific versions:
        # ("run_name", "run_id", 2),  # Use version 2
        # ("run_name", "run_id", "v1"),  # Use version 1
        # ("run_name", "run_id", "latest"),  # Explicitly use latest
    ]
    
    # Wandb project
    wandb_project = "raymondl/gpt2-small"
    
    # Initialize wandb
    wandb.login(key=settings.wandb_api_key)
    
    # Load explanations for each run
    run_data = {}
    
    for run_info in runs:
        if len(run_info) == 2:
            # Backward compatibility: (run_name, run_id)
            run_name, run_id = run_info
            version = None
        else:
            # New format: (run_name, run_id, version)
            run_name, run_id, version = run_info
        
        version_str = f" (version: {version})" if version is not None else ""
        print(f"\nLoading explanations for {run_name} (run_id: {run_id}){version_str}...")
        
        explanations = load_explanations_from_wandb(
            run_id=run_id,
            project=wandb_project,
            output_path="./artifacts",
            version=version
        )
        
        if explanations is None:
            print(f"Failed to load explanations for {run_name}")
            continue
        
        # Extract scores
        scores_by_layer = extract_scores_from_explanations(explanations)
        run_data[run_name] = scores_by_layer
        
        print(f"Successfully loaded explanations for {len(scores_by_layer)} layers")
    
    if not run_data:
        print("No explanation data could be loaded. Exiting.")
        return
    
    # Print summary statistics
    print_summary_statistics(run_data)
    
    # Create grouped bar comparison plot with statistical tests
    create_comparison_plot(run_data)
    
    # Save raw data for further analysis
    output_path = Path("./artifacts")
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_data = {}
    for run_name, scores_by_layer in run_data.items():
        summary_data[run_name] = {}
        for layer, scores_list in scores_by_layer.items():
            detection_mean, detection_std, detection_count = compute_statistics(scores_list, 'detection_score')
            fuzz_mean, fuzz_std, fuzz_count = compute_statistics(scores_list, 'fuzz_score')
            
            summary_data[run_name][layer] = {
                'num_neurons': len(scores_list),
                'detection': {
                    'mean': detection_mean if not np.isnan(detection_mean) else None,
                    'std': detection_std if not np.isnan(detection_std) else None,
                    'count': detection_count
                },
                'fuzz': {
                    'mean': fuzz_mean if not np.isnan(fuzz_mean) else None,
                    'std': fuzz_std if not np.isnan(fuzz_std) else None,
                    'count': fuzz_count
                }
            }
    
    summary_file = output_path / 'explanation_scores_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved summary data to {summary_file}")


if __name__ == "__main__":
    main() 