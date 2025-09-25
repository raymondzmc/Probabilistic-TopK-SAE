#!/usr/bin/env python3
"""
Plot Pareto curves for temperature (beta) experiments across different K values.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple

from settings import settings
from utils.io import load_metrics_from_wandb
from models import SAETransformer

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9


def collect_temperature_metrics_data(project: str = "raymondl/gpt2-small", k_values: List[int] = [8, 16, 32]) -> Dict[int, Dict[float, List[Dict]]]:
    """
    Collect metrics data for temperature experiments across different K values.
    
    Args:
        project: Wandb project name
        k_values: List of K values to analyze
    
    Returns:
        Dictionary with K values as keys, each containing beta -> run data
    """
    print(f"Collecting temperature metrics data from {project}...")
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get all runs from the project
    print(f"\nFetching runs from {project}...")
    all_runs = list(api.runs(project))
    print(f"  Found {len(all_runs)} runs")
    
    # Define experiment patterns for each K and beta
    experiment_patterns = {
        8: {
            'probabilistic_topk_k_8_initial_beta_0.5': 0.5,
            'probabilistic_topk_k_8_initial_beta_1.0': 1.0,
            'probabilistic_k_8': 5.0,  # Default beta
            'probabilistic_topk_k_8_initial_beta_10.0': 10.0,
        },
        16: {
            'probabilistic_topk_k_16_initial_beta_0.5': 0.5,
            'probabilistic_topk_k_16_initial_beta_1.0': 1.0,
            'probabilistic_k_16': 5.0,  # Default beta
            'probabilistic_topk_k_16_initial_beta_10.0': 10.0,
        },
        32: {
            'probabilistic_topk_k_32_initial_beta_0.5': 0.5,
            'probabilistic_topk_k_32_initial_beta_1.0': 1.0,
            'probabilistic_k_32': 5.0,  # Default beta
            'probabilistic_topk_k_32_initial_beta_10.0': 10.0,
        }
    }
    
    # Collect data by K value and beta
    data = {k: {} for k in k_values}
    
    # Track layer names
    all_layers = set()
    
    for run in all_runs:
        run_name = run.name
        
        # Check each K value
        for k in k_values:
            if run_name in experiment_patterns[k]:
                beta = experiment_patterns[k][run_name]
                print(f"  Loading metrics for {run_name} ({run.id}) - K={k}, β={beta}")
                metrics = load_metrics_from_wandb(run.id, project)
                
                if metrics:
                    # Initialize beta dict if needed
                    if beta not in data[k]:
                        data[k][beta] = []
                    
                    # Store per-layer metrics
                    run_data = {
                        'run_name': run_name,
                        'run_id': run.id,
                        'k': k,
                        'beta': beta,
                        'layers': {}
                    }
                    
                    # Process each layer
                    for layer_name, layer_metrics in metrics.items():
                        all_layers.add(layer_name)
                        run_data['layers'][layer_name] = {
                            'l0': layer_metrics['sparsity_l0'],
                            'mse': layer_metrics['mse'],
                            'explained_variance': layer_metrics['explained_variance'],
                            'alive_dict_components': layer_metrics.get('alive_dict_components', 0),
                            'alive_dict_proportion': layer_metrics.get('alive_dict_components_proportion', 0)
                        }
                    
                    data[k][beta].append(run_data)
    
    print(f"\nCollected data summary:")
    for k in k_values:
        print(f"  K={k}:")
        for beta, runs in sorted(data[k].items()):
            print(f"    β={beta}: {len(runs)} runs")
    print(f"Found {len(all_layers)} layers: {sorted(all_layers)}")
    
    return data, sorted(all_layers)


def plot_temperature_pareto_curves(data: Dict[int, Dict[float, List[Dict]]], layers: List[str], 
                                  output_dir: Path = Path("plots/temperature"),
                                  max_mse: float = float('inf'), max_l0: float = float('inf'),
                                  min_mse: float = 0.0, min_l0: float = 0.0):
    """
    Create Pareto curve plots where each curve represents a temperature (beta) value across K values.
    
    Args:
        data: Dictionary with K values and beta -> run data
        layers: List of layer names
        output_dir: Output directory for plots
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Color scheme for different beta values - using different colors than main plots
    colors = {
        0.5: '#8c564b',     # Brown (hot - low beta)
        1.0: '#9467bd',     # Purple
        5.0: '#d62728',     # Red (default beta)
        10.0: '#17becf',    # Cyan (cold - high beta)
    }
    
    # Marker styles for different K values
    k_markers = {
        8: 'o',     # Circle
        16: 's',    # Square
        32: 'D',    # Diamond
    }
    
    k_values = sorted(data.keys())
    beta_values = sorted(set(beta for k_data in data.values() for beta in k_data.keys()))
    
    # Create a figure for each layer
    for layer_idx, layer_name in enumerate(layers):
        print(f"\nProcessing temperature Pareto curves for layer: {layer_name}")
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        # Plot 1: MSE vs L0 (each curve is a beta value)
        ax1 = axes[0]
        
        for beta in beta_values:
            # Collect data across K values for this beta
            l0_values = []
            mse_values = []
            k_list = []
            
            for k in k_values:
                if beta in data[k] and data[k][beta]:
                    for run_data in data[k][beta]:
                        if layer_name in run_data['layers']:
                            l0 = run_data['layers'][layer_name]['l0']
                            mse = run_data['layers'][layer_name]['mse']
                            
                            # Apply filters
                            if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                                l0_values.append(l0)
                                mse_values.append(mse)
                                k_list.append(k)
            
            if l0_values and len(l0_values) > 1:
                # Convert to arrays
                l0_values = np.array(l0_values)
                mse_values = np.array(mse_values)
                
                # Sort by K (which equals L0)
                sort_idx = np.argsort(k_list)
                l0_sorted = l0_values[sort_idx]
                mse_sorted = mse_values[sort_idx]
                k_sorted = [k_list[i] for i in sort_idx]
                
                # Plot the line
                ax1.plot(l0_sorted, mse_sorted, 
                        color=colors[beta], linewidth=3, alpha=0.8,
                        label=f'β = {beta}')
                
                # Plot individual points with K-specific markers
                for l0, mse, k in zip(l0_sorted, mse_sorted, k_sorted):
                    ax1.scatter(l0, mse, 
                               color=colors[beta], marker=k_markers[k],
                               s=300, edgecolors='black', linewidth=2, zorder=5)
                    # Add K labels
                    ax1.annotate(f'K={k}', (l0, mse), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('L0 Sparsity', fontsize=20)
        ax1.set_ylabel('MSE ← (better)', fontsize=20)
        ax1.set_title('MSE vs L0 Pareto Curves (Temperature Comparison)', fontsize=24, pad=20)
        ax1.set_yscale('log')
        ax1.legend(loc='upper right', fontsize=16)
        ax1.tick_params(axis='both', labelsize=18)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Plot 2: Explained Variance vs L0
        ax2 = axes[1]
        
        for beta in beta_values:
            # Collect data across K values for this beta
            l0_values = []
            ev_values = []
            k_list = []
            
            for k in k_values:
                if beta in data[k] and data[k][beta]:
                    for run_data in data[k][beta]:
                        if layer_name in run_data['layers']:
                            l0 = run_data['layers'][layer_name]['l0']
                            mse = run_data['layers'][layer_name]['mse']
                            ev = run_data['layers'][layer_name]['explained_variance']
                            
                            # Apply filters
                            if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                                l0_values.append(l0)
                                ev_values.append(ev)
                                k_list.append(k)
            
            if l0_values and len(l0_values) > 1:
                # Convert to arrays
                l0_values = np.array(l0_values)
                ev_values = np.array(ev_values)
                
                # Sort by K
                sort_idx = np.argsort(k_list)
                l0_sorted = l0_values[sort_idx]
                ev_sorted = ev_values[sort_idx]
                k_sorted = [k_list[i] for i in sort_idx]
                
                # Plot the line
                ax2.plot(l0_sorted, ev_sorted,
                        color=colors[beta], linewidth=3, alpha=0.8,
                        label=f'β = {beta}')
                
                # Plot individual points with K-specific markers
                for l0, ev, k in zip(l0_sorted, ev_sorted, k_sorted):
                    ax2.scatter(l0, ev,
                               color=colors[beta], marker=k_markers[k],
                               s=300, edgecolors='black', linewidth=2, zorder=5)
                    # Add K labels
                    ax2.annotate(f'K={k}', (l0, ev), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('L0 Sparsity', fontsize=20)
        ax2.set_ylabel('Explained Variance → (better)', fontsize=20)
        ax2.set_title('Explained Variance vs L0 Pareto Curves (Temperature Comparison)', fontsize=24, pad=20)
        ax2.legend(loc='lower right', fontsize=16)
        ax2.tick_params(axis='both', labelsize=18)
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout and save
        layer_display_name = layer_name.replace('.', '_')
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"temperature_pareto_curves_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved temperature Pareto curves to: {output_path}")
        
        # Also save as SVG
        output_path_svg = output_dir / f"temperature_pareto_curves_{layer_display_name}.svg"
        plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
        
        plt.close()


def print_temperature_summary(data: Dict[int, Dict[float, List[Dict]]], layers: List[str]):
    """Print a summary of the temperature experiment results."""
    
    print("\n" + "=" * 80)
    print("TEMPERATURE EXPERIMENT SUMMARY")
    print("=" * 80)
    
    for layer_name in layers:
        print(f"\n{'='*80}")
        print(f"LAYER: {layer_name}")
        print(f"{'='*80}")
        
        for k in sorted(data.keys()):
            print(f"\nK = {k}:")
            print("-" * 40)
            print(f"{'Beta':>6} {'MSE':>12} {'EV':>8} {'Alive':>8}")
            print("-" * 40)
            
            results = []
            for beta in sorted(data[k].keys()):
                if data[k][beta]:
                    for run_data in data[k][beta]:
                        if layer_name in run_data['layers']:
                            metrics = run_data['layers'][layer_name]
                            results.append({
                                'beta': beta,
                                'mse': metrics['mse'],
                                'ev': metrics['explained_variance'],
                                'alive': metrics['alive_dict_components']
                            })
            
            # Sort by beta
            results.sort(key=lambda x: x['beta'])
            
            for r in results:
                print(f"{r['beta']:>6.1f} {r['mse']:>12.6f} {r['ev']:>8.4f} {r['alive']:>8.0f}")
    
    print("\n" + "=" * 80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot temperature Pareto curves across K values"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="raymondl/gpt2-small",
        help="Wandb project to collect data from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/temperature",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs='+',
        default=[8, 16, 32],
        help="K values to analyze (default: 8 16 32)"
    )
    
    args = parser.parse_args()
    
    # Collect data
    print("=" * 80)
    print("Collecting temperature experiment data...")
    print("=" * 80)
    data, layers = collect_temperature_metrics_data(args.project, args.k_values)
    
    # Create temperature Pareto plots
    print("\n" + "=" * 80)
    print("Creating temperature Pareto curve plots...")
    print("=" * 80)
    plot_temperature_pareto_curves(data, layers, Path(args.output_dir))
    
    # Print summary
    print_temperature_summary(data, layers)
    
    print("\n" + "=" * 80)
    print("Temperature Pareto analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main() 