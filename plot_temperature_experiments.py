#!/usr/bin/env python3
"""
Plot Pareto curves for temperature (beta) experiments with K = 16.
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


def collect_temperature_metrics_data(project: str = "raymondl/gpt2-small") -> Dict[str, List[Dict]]:
    """
    Collect metrics data for temperature experiments.
    
    Args:
        project: Wandb project name
    
    Returns:
        Dictionary with beta values as keys, each containing run data
    """
    print(f"Collecting temperature metrics data from {project}...")
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get all runs from the project
    print(f"\nFetching runs from {project}...")
    all_runs = list(api.runs(project))
    print(f"  Found {len(all_runs)} runs")
    
    # Define experiment runs and their beta values
    experiment_runs = {
        'probabilistic_topk_k_16_initial_beta_0.5': 0.5,
        'probabilistic_topk_k_16_initial_beta_1.0': 1.0,
        'probabilistic_k_16': 5.0,  # Default beta
        'probabilistic_topk_k_16_initial_beta_10.0': 10.0,
    }
    
    # Collect data by beta value
    data = {beta: [] for beta in experiment_runs.values()}
    
    # Track layer names
    all_layers = set()
    
    for run in all_runs:
        run_name = run.name
        
        # Check if this is one of our temperature experiment runs
        if run_name in experiment_runs:
            beta = experiment_runs[run_name]
            print(f"  Loading metrics for {run_name} ({run.id}) - β={beta}")
            metrics = load_metrics_from_wandb(run.id, project)
            
            if metrics:
                # Store per-layer metrics
                run_data = {
                    'run_name': run_name,
                    'run_id': run.id,
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
                
                data[beta].append(run_data)
    
    print(f"\nCollected data summary:")
    for beta, runs in sorted(data.items()):
        print(f"  β={beta}: {len(runs)} runs")
    print(f"Found {len(all_layers)} layers: {sorted(all_layers)}")
    
    return data, sorted(all_layers)


def plot_temperature_curves(data: Dict[float, List[Dict]], layers: List[str], 
                           output_dir: Path = Path("plots/temperature"),
                           max_mse: float = float('inf'), max_l0: float = float('inf'),
                           min_mse: float = 0.0, min_l0: float = 0.0):
    """
    Create plots for temperature experiments showing different beta values.
    
    Args:
        data: Dictionary with beta values and run data
        layers: List of layer names
        output_dir: Output directory for plots
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Color scheme - gradient from hot to cold for different betas
    colors = {
        0.5: '#d62728',     # Red (hot - low beta)
        1.0: '#ff7f0e',     # Orange
        5.0: '#2ca02c',     # Green
        10.0: '#1f77b4',    # Blue (cold - high beta)
    }
    
    # Create a figure for each layer
    for layer_idx, layer_name in enumerate(layers):
        print(f"\nProcessing temperature curves for layer: {layer_name}")
        
        # Create figure with 2 subplots - larger size for better visibility
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        # Plot 1: MSE vs Beta
        ax1 = axes[0]
        
        # Collect data for plotting
        beta_values = []
        mse_values = []
        ev_values = []
        alive_values = []
        
        for beta in sorted(data.keys()):
            if data[beta]:
                for run_data in data[beta]:
                    if layer_name in run_data['layers']:
                        layer_data = run_data['layers'][layer_name]
                        beta_values.append(beta)
                        mse_values.append(layer_data['mse'])
                        ev_values.append(layer_data['explained_variance'])
                        alive_values.append(layer_data['alive_dict_components'])
        
        if beta_values:
            # Convert to arrays
            beta_values = np.array(beta_values)
            mse_values = np.array(mse_values)
            ev_values = np.array(ev_values)
            alive_values = np.array(alive_values)
            
            # Plot MSE vs Beta
            for i, (beta, mse) in enumerate(zip(beta_values, mse_values)):
                ax1.scatter(beta, mse, 
                           color=colors[beta], s=300, 
                           edgecolors='black', linewidth=2, zorder=5,
                           label=f'β = {beta}')
            
            # Connect points with a line
            sort_idx = np.argsort(beta_values)
            ax1.plot(beta_values[sort_idx], mse_values[sort_idx], 
                    'k--', linewidth=2, alpha=0.5)
            
            ax1.set_xlabel('Temperature (β)', fontsize=20)
            ax1.set_ylabel('MSE', fontsize=20)
            ax1.set_title('MSE vs Temperature', fontsize=24, pad=20)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.legend(loc='best', fontsize=16)
            ax1.tick_params(axis='both', labelsize=18)
            ax1.grid(True, alpha=0.3, which='both')
            
            # Plot 2: Explained Variance vs Beta
            ax2 = axes[1]
            
            for i, (beta, ev) in enumerate(zip(beta_values, ev_values)):
                ax2.scatter(beta, ev,
                           color=colors[beta], s=300,
                           edgecolors='black', linewidth=2, zorder=5,
                           label=f'β = {beta}')
            
            # Connect points with a line
            ax2.plot(beta_values[sort_idx], ev_values[sort_idx],
                    'k--', linewidth=2, alpha=0.5)
            
            ax2.set_xlabel('Temperature (β)', fontsize=20)
            ax2.set_ylabel('Explained Variance', fontsize=20)
            ax2.set_title('Explained Variance vs Temperature', fontsize=24, pad=20)
            ax2.set_xscale('log')
            ax2.legend(loc='best', fontsize=16)
            ax2.tick_params(axis='both', labelsize=18)
            ax2.grid(True, alpha=0.3, which='both')
        
        # Adjust layout and save
        layer_display_name = layer_name.replace('.', '_')
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"temperature_curves_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved temperature curves to: {output_path}")
        
        # Also save as SVG
        output_path_svg = output_dir / f"temperature_curves_{layer_display_name}.svg"
        plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
        
        plt.close()
        
        # Create a second plot showing Alive Features vs Beta
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if len(beta_values) > 0:
            for i, (beta, alive) in enumerate(zip(beta_values, alive_values)):
                ax.scatter(beta, alive,
                          color=colors[beta], s=300,
                          edgecolors='black', linewidth=2, zorder=5,
                          label=f'β = {beta}')
            
            # Connect points with a line
            ax.plot(beta_values[sort_idx], alive_values[sort_idx],
                   'k--', linewidth=2, alpha=0.5)
            
            ax.set_xlabel('Temperature (β)', fontsize=20)
            ax.set_ylabel('Alive Dictionary Components', fontsize=20)
            ax.set_title('Feature Utilization vs Temperature', fontsize=24, pad=20)
            ax.set_xscale('log')
            ax.legend(loc='best', fontsize=16)
            ax.tick_params(axis='both', labelsize=18)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / f"alive_features_vs_temperature_{layer_display_name}.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"  Saved alive features plot to: {output_path}")
            
            # Also save as SVG
            output_path_svg = output_dir / f"alive_features_vs_temperature_{layer_display_name}.svg"
            plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
            
            plt.close()


def print_temperature_summary(data: Dict[float, List[Dict]], layers: List[str]):
    """Print a summary of the temperature experiment results."""
    
    print("\n" + "=" * 80)
    print("TEMPERATURE EXPERIMENT SUMMARY (K=16)")
    print("=" * 80)
    
    for layer_name in layers:
        print(f"\n{'='*80}")
        print(f"LAYER: {layer_name}")
        print(f"{'='*80}")
        
        print(f"\n{'Beta':>6} {'MSE':>12} {'EV':>8} {'Alive':>8}")
        print("-" * 40)
        
        results = []
        for beta in sorted(data.keys()):
            if data[beta]:
                for run_data in data[beta]:
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
        description="Plot temperature (beta) experiments for K=16"
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
    
    args = parser.parse_args()
    
    # Collect data
    print("=" * 80)
    print("Collecting temperature experiment data...")
    print("=" * 80)
    data, layers = collect_temperature_metrics_data(args.project)
    
    # Create temperature plots
    print("\n" + "=" * 80)
    print("Creating temperature curve plots...")
    print("=" * 80)
    plot_temperature_curves(data, layers, Path(args.output_dir))
    
    # Print summary
    print_temperature_summary(data, layers)
    
    print("\n" + "=" * 80)
    print("Temperature analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main() 