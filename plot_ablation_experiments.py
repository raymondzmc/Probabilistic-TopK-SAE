#!/usr/bin/env python3
"""
Plot Pareto curves for ablation experiments with K = 8, 16, 32.
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


def collect_ablation_metrics_data(project: str = "raymondl/gpt2-small", k_values: List[int] = [8, 16, 32]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Collect metrics data for ablation experiments.
    
    Args:
        project: Wandb project name
        k_values: List of K values to collect data for
    
    Returns:
        Dictionary with K values as keys, each containing experiment type -> run data
    """
    print(f"Collecting ablation metrics data from {project}...")
    
    # Login to wandb
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    
    # Get all runs from the project
    print(f"\nFetching runs from {project}...")
    all_runs = list(api.runs(project))
    print(f"  Found {len(all_runs)} runs")
    
    # Define experiment types and their run name patterns
    experiment_patterns = {
        'topk': 'topk_k_{k}_interpret',
        'probabilistic': 'probabilistic_k_{k}',
        'no_magnitude': 'probabilistic_topk_k_{k}_use_magnitude_false',
        'no_layernorm': 'probabilistic_topk_k_{k}_use_layer_norm_false',
        'no_z': 'probabilistic_topk_k_{k}_z_scale_None',
        'sigmoid_gate': 'probabilistic_topk_k_{k}_use_hard_concrete_false',
    }
    
    # Collect data by K value
    data = {}
    for k in k_values:
        data[k] = {exp_type: [] for exp_type in experiment_patterns.keys()}
    
    # Track layer names
    all_layers = set()
    
    for run in all_runs:
        run_name = run.name
        
        # Check each K value and experiment type
        for k in k_values:
            for exp_type, pattern in experiment_patterns.items():
                expected_name = pattern.format(k=k)
                if run_name == expected_name:
                    print(f"  Loading metrics for {run_name} ({run.id}) - K={k}, Type: {exp_type}")
                    metrics = load_metrics_from_wandb(run.id, project)
                    
                    if metrics:
                        # Store per-layer metrics
                        run_data = {
                            'run_name': run_name,
                            'run_id': run.id,
                            'k': k,
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
                        
                        data[k][exp_type].append(run_data)
                    break
    
    print(f"\nCollected data summary:")
    for k in k_values:
        print(f"  K={k}:")
        for exp_type, runs in data[k].items():
            print(f"    {exp_type}: {len(runs)} runs")
    print(f"Found {len(all_layers)} layers: {sorted(all_layers)}")
    
    return data, sorted(all_layers)


def find_pareto_frontier(x_values: np.ndarray, y_values: np.ndarray, 
                        minimize_x: bool = True, minimize_y: bool = True) -> np.ndarray:
    """
    Find the Pareto frontier for 2D data.
    
    Args:
        x_values: X-axis values
        y_values: Y-axis values
        minimize_x: If True, prefer smaller x values
        minimize_y: If True, prefer smaller y values
    
    Returns:
        Boolean array indicating which points are on the Pareto frontier
    """
    n_points = len(x_values)
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if minimize_x and minimize_y:
                    # Both objectives should be minimized
                    if x_values[j] <= x_values[i] and y_values[j] <= y_values[i]:
                        if x_values[j] < x_values[i] or y_values[j] < y_values[i]:
                            is_pareto[i] = False
                            break
                elif minimize_x and not minimize_y:
                    # Minimize x, maximize y
                    if x_values[j] <= x_values[i] and y_values[j] >= y_values[i]:
                        if x_values[j] < x_values[i] or y_values[j] > y_values[i]:
                            is_pareto[i] = False
                            break
    
    return is_pareto


def plot_ablation_pareto_curves(data: Dict[int, Dict[str, List[Dict]]], layers: List[str], 
                               output_dir: Path = Path("plots/ablation"),
                               max_mse: float = float('inf'), max_l0: float = float('inf'),
                               min_mse: float = 0.0, min_l0: float = 0.0):
    """
    Create Pareto curve plots for ablation experiments.
    
    Args:
        data: Dictionary with K values and experiment type data
        layers: List of layer names
        output_dir: Output directory for plots
        max_mse: Maximum MSE threshold (default: inf - no filtering)
        max_l0: Maximum L0 threshold (default: inf - no filtering)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Color scheme - consistent with main plot (Blue for TopK, Red for Probabilistic)
    colors = {
        'topk': '#1f77b4',                    # Blue
        'probabilistic': '#d62728',           # Red
        'no_magnitude': '#ff7f0e',            # Orange
        'no_layernorm': '#2ca02c',            # Green
        'no_z': '#9467bd',                    # Purple
        'sigmoid_gate': '#8c564b',            # Brown
    }
    
    # Marker styles
    markers = {
        'topk': 'o',                          # Circle
        'probabilistic': 'D',                 # Diamond
        'no_magnitude': 's',                  # Square
        'no_layernorm': '^',                  # Triangle up
        'no_z': 'v',                          # Triangle down
        'sigmoid_gate': 'p',                  # Pentagon
    }
    
    # Labels for legend
    labels = {
        'topk': 'TopK',
        'probabilistic': 'Probabilistic TopK',
        'no_magnitude': 'w/o Magnitude',
        'no_layernorm': 'w/o LayerNorm',
        'no_z': 'w/o z (reconstruction)',
        'sigmoid_gate': 'Sigmoid Gate',
    }
    
    # Create plots for each K value
    for k in sorted(data.keys()):
        print(f"\nProcessing K={k}...")
        
        # Create a figure for each layer
        for layer_idx, layer_name in enumerate(layers):
            print(f"  Processing layer: {layer_name}")
            
            # Create figure with 2 subplots - larger size for better visibility
            fig, axes = plt.subplots(1, 2, figsize=(24, 12))
            
            # Plot 1: MSE vs L0 (minimize both)
            ax1 = axes[0]
            
            # Collect all data points for Pareto analysis
            all_l0 = []
            all_mse = []
            all_exp_types = []
            all_colors = []
            all_markers = []
            all_labels = []
            
            for exp_type in ['topk', 'probabilistic', 'no_magnitude', 'no_layernorm', 'no_z', 'sigmoid_gate']:
                if data[k].get(exp_type) and data[k][exp_type]:
                    # Extract layer-specific data
                    for run_data in data[k][exp_type]:
                        if layer_name in run_data['layers']:
                            l0 = run_data['layers'][layer_name]['l0']
                            mse = run_data['layers'][layer_name]['mse']
                            
                            # Apply filters
                            if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                                all_l0.append(l0)
                                all_mse.append(mse)
                                all_exp_types.append(exp_type)
                                all_colors.append(colors[exp_type])
                                all_markers.append(markers[exp_type])
                                all_labels.append(labels[exp_type])
            
            if all_l0:
                all_l0 = np.array(all_l0)
                all_mse = np.array(all_mse)
                
                # Find Pareto frontier
                is_pareto = find_pareto_frontier(all_l0, all_mse, minimize_x=True, minimize_y=True)
                
                # Plot all points
                for i, (l0, mse, exp_type, color, marker, label) in enumerate(
                    zip(all_l0, all_mse, all_exp_types, all_colors, all_markers, all_labels)):
                    # Only add label once per experiment type
                    label_to_use = label if exp_type not in [et for et, l0_, mse_ in 
                                                             zip(all_exp_types[:i], all_l0[:i], all_mse[:i])] else None
                    ax1.scatter(l0, mse, 
                               color=color, marker=marker,
                               alpha=0.8, s=200, label=label_to_use,
                               edgecolors='black' if is_pareto[i] else 'none',
                               linewidth=2 if is_pareto[i] else 0,
                               zorder=5 if is_pareto[i] else 3)
                
                # Connect Pareto points
                if np.any(is_pareto):
                    pareto_l0 = all_l0[is_pareto]
                    pareto_mse = all_mse[is_pareto]
                    pareto_types = [all_exp_types[i] for i in range(len(all_exp_types)) if is_pareto[i]]
                    
                    # Sort for line plotting
                    sort_idx = np.argsort(pareto_l0)
                    pareto_l0 = pareto_l0[sort_idx]
                    pareto_mse = pareto_mse[sort_idx]
                    pareto_types_sorted = [pareto_types[i] for i in sort_idx]
                    
                    # Plot Pareto frontier line
                    ax1.plot(pareto_l0, pareto_mse, 'k--', linewidth=2, alpha=0.5, label='Pareto Frontier')
                    
                    # Annotate Pareto points
                    for l0, mse, exp_type in zip(pareto_l0, pareto_mse, pareto_types_sorted):
                        ax1.annotate(labels[exp_type], (l0, mse), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=10, fontweight='bold')
            
            ax1.set_xlabel('L0 Sparsity', fontsize=20)
            ax1.set_ylabel('MSE ← (better)', fontsize=20)
            ax1.set_title(f'MSE vs L0 (K={k})', fontsize=24, pad=20)
            ax1.legend(loc='upper right', fontsize=16)
            ax1.tick_params(axis='both', labelsize=18)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Explained Variance vs L0
            ax2 = axes[1]
            
            # Collect all data points for Pareto analysis
            all_l0_ev = []
            all_ev = []
            all_exp_types_ev = []
            all_colors_ev = []
            all_markers_ev = []
            all_labels_ev = []
            
            for exp_type in ['topk', 'probabilistic', 'no_magnitude', 'no_layernorm', 'no_z', 'sigmoid_gate']:
                if data[k].get(exp_type) and data[k][exp_type]:
                    # Extract layer-specific data
                    for run_data in data[k][exp_type]:
                        if layer_name in run_data['layers']:
                            l0 = run_data['layers'][layer_name]['l0']
                            mse = run_data['layers'][layer_name]['mse']
                            ev = run_data['layers'][layer_name]['explained_variance']
                            
                            # Apply filters
                            if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                                all_l0_ev.append(l0)
                                all_ev.append(ev)
                                all_exp_types_ev.append(exp_type)
                                all_colors_ev.append(colors[exp_type])
                                all_markers_ev.append(markers[exp_type])
                                all_labels_ev.append(labels[exp_type])
            
            if all_l0_ev:
                all_l0_ev = np.array(all_l0_ev)
                all_ev = np.array(all_ev)
                
                # Find Pareto frontier (minimize L0, maximize EV)
                is_pareto_ev = find_pareto_frontier(all_l0_ev, -all_ev, minimize_x=True, minimize_y=True)
                
                # Plot all points
                for i, (l0, ev, exp_type, color, marker, label) in enumerate(
                    zip(all_l0_ev, all_ev, all_exp_types_ev, all_colors_ev, all_markers_ev, all_labels_ev)):
                    # Only add label once per experiment type
                    label_to_use = label if exp_type not in [et for et, l0_, ev_ in 
                                                             zip(all_exp_types_ev[:i], all_l0_ev[:i], all_ev[:i])] else None
                    ax2.scatter(l0, ev,
                               color=color, marker=marker,
                               alpha=0.8, s=200, label=label_to_use,
                               edgecolors='black' if is_pareto_ev[i] else 'none',
                               linewidth=2 if is_pareto_ev[i] else 0,
                               zorder=5 if is_pareto_ev[i] else 3)
                
                # Connect Pareto points
                if np.any(is_pareto_ev):
                    pareto_l0 = all_l0_ev[is_pareto_ev]
                    pareto_ev = all_ev[is_pareto_ev]
                    pareto_types = [all_exp_types_ev[i] for i in range(len(all_exp_types_ev)) if is_pareto_ev[i]]
                    
                    # Sort for line plotting
                    sort_idx = np.argsort(pareto_l0)
                    pareto_l0 = pareto_l0[sort_idx]
                    pareto_ev = pareto_ev[sort_idx]
                    pareto_types_sorted = [pareto_types[i] for i in sort_idx]
                    
                    # Plot Pareto frontier line
                    ax2.plot(pareto_l0, pareto_ev, 'k--', linewidth=2, alpha=0.5, label='Pareto Frontier')
                    
                    # Annotate Pareto points
                    for l0, ev, exp_type in zip(pareto_l0, pareto_ev, pareto_types_sorted):
                        ax2.annotate(labels[exp_type], (l0, ev), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=10, fontweight='bold')
            
            ax2.set_xlabel('L0 Sparsity', fontsize=20)
            ax2.set_ylabel('Explained Variance → (better)', fontsize=20)
            ax2.set_title(f'Explained Variance vs L0 (K={k})', fontsize=24, pad=20)
            ax2.legend(loc='lower right', fontsize=16)
            ax2.tick_params(axis='both', labelsize=18)
            ax2.grid(True, alpha=0.3)
            
            # Adjust layout and save
            layer_display_name = layer_name.replace('.', '_')
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / f"ablation_k{k}_{layer_display_name}.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"    Saved plot to: {output_path}")
            
            # Also save as SVG
            output_path_svg = output_dir / f"ablation_k{k}_{layer_display_name}.svg"
            plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
            
            plt.close()


def plot_combined_k_comparison(data: Dict[int, Dict[str, List[Dict]]], layers: List[str], 
                              output_dir: Path = Path("plots/ablation"),
                              max_mse: float = float('inf'), max_l0: float = float('inf'),
                              min_mse: float = 0.0, min_l0: float = 0.0):
    """
    Create combined plots comparing all K values for each experiment type.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Color scheme by K value
    k_colors = {
        8: '#1f77b4',    # Blue
        16: '#ff7f0e',   # Orange  
        32: '#2ca02c',   # Green
    }
    
    # Marker styles by experiment type
    markers = {
        'topk': 'o',                          # Circle
        'probabilistic': 'D',                 # Diamond
        'no_magnitude': 's',                  # Square
        'no_layernorm': '^',                  # Triangle up
        'no_z': 'v',                          # Triangle down
        'sigmoid_gate': 'p',                  # Pentagon
    }
    
    # Labels for legend
    exp_labels = {
        'topk': 'TopK',
        'probabilistic': 'Probabilistic TopK',
        'no_magnitude': 'w/o Magnitude',
        'no_layernorm': 'w/o LayerNorm',
        'no_z': 'w/o z (reconstruction)',
        'sigmoid_gate': 'Sigmoid Gate',
    }
    
    # Create a figure for each layer
    for layer_idx, layer_name in enumerate(layers):
        print(f"\nProcessing combined K comparison for layer: {layer_name}")
        
        # Create figure with subplots for each experiment type
        fig, axes = plt.subplots(2, 3, figsize=(30, 20))
        axes = axes.flatten()
        
        for idx, exp_type in enumerate(['topk', 'probabilistic', 'no_magnitude', 'no_layernorm', 'no_z', 'sigmoid_gate']):
            ax = axes[idx]
            
            # Plot data for each K value
            for k in sorted(data.keys()):
                if data[k].get(exp_type) and data[k][exp_type]:
                    # Extract layer-specific data
                    for run_data in data[k][exp_type]:
                        if layer_name in run_data['layers']:
                            l0 = run_data['layers'][layer_name]['l0']
                            mse = run_data['layers'][layer_name]['mse']
                            ev = run_data['layers'][layer_name]['explained_variance']
                            
                            # Apply filters
                            if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                                # Plot MSE vs EV (using L0 as size)
                                ax.scatter(ev, mse, 
                                          color=k_colors[k], marker=markers[exp_type],
                                          alpha=0.8, s=200 + l0*2, label=f'K={k}',
                                          edgecolors='black', linewidth=1)
            
            ax.set_xlabel('Explained Variance →', fontsize=16)
            ax.set_ylabel('MSE ←', fontsize=16)
            ax.set_title(exp_labels[exp_type], fontsize=18, pad=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=14)
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(loc='upper right', fontsize=14)
        
        # Adjust layout
        layer_display_name = layer_name.replace('.', '_')
        plt.suptitle(f'Ablation Study Comparison - {layer_name}', fontsize=24)
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"ablation_combined_comparison_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved combined comparison to: {output_path}")
        
        # Also save as SVG
        output_path_svg = output_dir / f"ablation_combined_comparison_{layer_display_name}.svg"
        plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
        
        plt.close()


def plot_k_pareto_curves(data: Dict[int, Dict[str, List[Dict]]], layers: List[str], 
                        output_dir: Path = Path("plots/ablation"),
                        max_mse: float = float('inf'), max_l0: float = float('inf'),
                        min_mse: float = 0.0, min_l0: float = 0.0):
    """
    Create Pareto curve plots where each curve represents an experiment type across K values.
    Each curve has 3 points for K=8, 16, 32.
    
    Args:
        data: Dictionary with K values and experiment type data
        layers: List of layer names
        output_dir: Output directory for plots
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Color scheme - consistent with main plot
    colors = {
        'topk': '#1f77b4',                    # Blue
        'probabilistic': '#d62728',           # Red
        'no_magnitude': '#ff7f0e',            # Orange
        'no_layernorm': '#2ca02c',            # Green
        'no_z': '#9467bd',                    # Purple
        'sigmoid_gate': '#8c564b',            # Brown
    }
    
    # Marker styles - different marker for each K
    k_markers = {
        8: 'o',     # Circle
        16: 's',    # Square
        32: 'D',    # Diamond
    }
    
    # Labels for legend
    labels = {
        'topk': 'TopK',
        'probabilistic': 'Probabilistic TopK',
        'no_magnitude': 'w/o Magnitude',
        'no_layernorm': 'w/o LayerNorm',
        'no_z': 'w/o z (reconstruction)',
        'sigmoid_gate': 'Sigmoid Gate',
    }
    
    k_values = sorted(data.keys())
    
    # Create a figure for each layer
    for layer_idx, layer_name in enumerate(layers):
        print(f"\nProcessing K-Pareto curves for layer: {layer_name}")
        
        # Create figure with 2 subplots - larger size for better visibility
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        # Plot 1: MSE vs L0 (each curve is an experiment type)
        ax1 = axes[0]
        
        for exp_type in ['topk', 'probabilistic', 'no_magnitude', 'no_layernorm', 'sigmoid_gate']:  # Removed 'no_z'
            # Collect data across K values for this experiment type
            l0_values = []
            mse_values = []
            k_list = []
            
            for k in k_values:
                if data[k].get(exp_type) and data[k][exp_type]:
                    for run_data in data[k][exp_type]:
                        if layer_name in run_data['layers']:
                            l0 = run_data['layers'][layer_name]['l0']
                            mse = run_data['layers'][layer_name]['mse']
                            
                            # Apply filters
                            if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                                l0_values.append(l0)
                                mse_values.append(mse)
                                k_list.append(k)
            
            if l0_values and len(l0_values) > 1:  # Need at least 2 points for a line
                # Convert to arrays
                l0_values = np.array(l0_values)
                mse_values = np.array(mse_values)
                
                # Sort by K (which determines L0 since L0 = K for these experiments)
                sort_idx = np.argsort(k_list)
                l0_sorted = l0_values[sort_idx]
                mse_sorted = mse_values[sort_idx]
                k_sorted = [k_list[i] for i in sort_idx]
                
                # Plot the line
                ax1.plot(l0_sorted, mse_sorted, 
                        color=colors[exp_type], linewidth=3, alpha=0.8,
                        label=labels[exp_type])
                
                # Plot individual points with K-specific markers
                for l0, mse, k in zip(l0_sorted, mse_sorted, k_sorted):
                    ax1.scatter(l0, mse, 
                               color=colors[exp_type], marker=k_markers[k],
                               s=300, edgecolors='black', linewidth=2, zorder=5)
                    # Add K labels
                    ax1.annotate(f'K={k}', (l0, mse), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')
        
        # Set y-axis limits to include all data points including w/o Magnitude
        all_mse_values = []
        for exp_type in ['topk', 'probabilistic', 'no_magnitude', 'no_layernorm', 'sigmoid_gate']:
            for k in k_values:
                if data[k].get(exp_type) and data[k][exp_type]:
                    for run_data in data[k][exp_type]:
                        if layer_name in run_data['layers']:
                            mse = run_data['layers'][layer_name]['mse']
                            if min_mse <= mse <= max_mse:
                                all_mse_values.append(mse)
        
        if all_mse_values:
            min_mse_plot = min(all_mse_values) * 0.7  # 30% below minimum
            max_mse_plot = max(all_mse_values) * 1.3  # 30% above maximum
            ax1.set_ylim(min_mse_plot, max_mse_plot)
        
        ax1.set_xlabel('L0 Sparsity', fontsize=20)
        ax1.set_ylabel('MSE ← (better)', fontsize=20)
        ax1.set_title('MSE vs L0 Pareto Curves', fontsize=24, pad=20)
        ax1.set_yscale('log')  # Add log scale for MSE
        ax1.legend(loc='upper right', fontsize=16)
        ax1.tick_params(axis='both', labelsize=18)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Plot 2: Explained Variance vs L0
        ax2 = axes[1]
        
        for exp_type in ['topk', 'probabilistic', 'no_magnitude', 'no_layernorm', 'sigmoid_gate']:  # Removed 'no_z'
            # Collect data across K values for this experiment type
            l0_values = []
            ev_values = []
            k_list = []
            
            for k in k_values:
                if data[k].get(exp_type) and data[k][exp_type]:
                    for run_data in data[k][exp_type]:
                        if layer_name in run_data['layers']:
                            l0 = run_data['layers'][layer_name]['l0']
                            mse = run_data['layers'][layer_name]['mse']
                            ev = run_data['layers'][layer_name]['explained_variance']
                            
                            # Apply filters
                            if min_l0 <= l0 <= max_l0 and min_mse <= mse <= max_mse:
                                l0_values.append(l0)
                                ev_values.append(ev)
                                k_list.append(k)
            
            if l0_values and len(l0_values) > 1:  # Need at least 2 points for a line
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
                        color=colors[exp_type], linewidth=3, alpha=0.8,
                        label=labels[exp_type])
                
                # Plot individual points with K-specific markers
                for l0, ev, k in zip(l0_sorted, ev_sorted, k_sorted):
                    ax2.scatter(l0, ev,
                               color=colors[exp_type], marker=k_markers[k],
                               s=300, edgecolors='black', linewidth=2, zorder=5)
                    # Add K labels
                    ax2.annotate(f'K={k}', (l0, ev), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('L0 Sparsity', fontsize=20)
        ax2.set_ylabel('Explained Variance → (better)', fontsize=20)
        ax2.set_title('Explained Variance vs L0 Pareto Curves', fontsize=24, pad=20)
        ax2.legend(loc='lower right', fontsize=16)
        ax2.tick_params(axis='both', labelsize=18)
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout and save
        layer_display_name = layer_name.replace('.', '_')
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"k_pareto_curves_{layer_display_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved K-Pareto curves to: {output_path}")
        
        # Also save as SVG
        output_path_svg = output_dir / f"k_pareto_curves_{layer_display_name}.svg"
        plt.savefig(output_path_svg, bbox_inches='tight', format='svg')
        
        plt.close()


def print_ablation_summary(data: Dict[int, Dict[str, List[Dict]]], layers: List[str]):
    """Print a summary of the ablation experiment results."""
    
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT SUMMARY")
    print("=" * 80)
    
    exp_labels = {
        'topk': 'TopK',
        'probabilistic': 'Probabilistic TopK',
        'no_magnitude': 'w/o Magnitude',
        'no_layernorm': 'w/o LayerNorm',
        'no_z': 'w/o z (reconstruction)',
        'sigmoid_gate': 'Sigmoid Gate',
    }
    
    for layer_name in layers:
        print(f"\n{'='*80}")
        print(f"LAYER: {layer_name}")
        print(f"{'='*80}")
        
        for k in sorted(data.keys()):
            print(f"\nK = {k}:")
            print("-" * 40)
            
            results = []
            for exp_type in ['topk', 'probabilistic', 'no_magnitude', 'no_layernorm', 'no_z', 'sigmoid_gate']:
                if data[k].get(exp_type) and data[k][exp_type]:
                    for run_data in data[k][exp_type]:
                        if layer_name in run_data['layers']:
                            metrics = run_data['layers'][layer_name]
                            results.append({
                                'type': exp_labels[exp_type],
                                'l0': metrics['l0'],
                                'mse': metrics['mse'],
                                'ev': metrics['explained_variance'],
                                'alive': metrics['alive_dict_components']
                            })
            
            if results:
                # Sort by MSE for better readability
                results.sort(key=lambda x: x['mse'])
                
                print(f"{'Experiment':<25} {'L0':>8} {'MSE':>12} {'EV':>8} {'Alive':>8}")
                print("-" * 65)
                for r in results:
                    print(f"{r['type']:<25} {r['l0']:>8.2f} {r['mse']:>12.6f} {r['ev']:>8.4f} {r['alive']:>8.0f}")
    
    print("\n" + "=" * 80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot Pareto curves for ablation experiments with K = 8, 16, 32"
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
        default="plots/ablation",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs='+',
        default=[8, 16, 32],
        help="K values to analyze (default: 8 16 32)"
    )
    parser.add_argument(
        "--max-mse",
        type=float,
        default=float('inf'),
        help="Maximum MSE threshold for filtering (default: inf - no filtering)"
    )
    parser.add_argument(
        "--max-l0",
        type=float,
        default=100,
        help="Maximum L0 threshold for filtering (default: 100)"
    )
    parser.add_argument(
        "--min-mse",
        type=float,
        default=0.0,
        help="Minimum MSE threshold for filtering (default: 0.0)"
    )
    parser.add_argument(
        "--min-l0",
        type=float,
        default=0.0,
        help="Minimum L0 threshold for filtering (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # Collect data
    print("=" * 80)
    print("Collecting ablation experiment data...")
    print("=" * 80)
    data, layers = collect_ablation_metrics_data(args.project, args.k_values)
    
    # Create K-Pareto curve plots
    print("\n" + "=" * 80)
    print("Creating K-Pareto curve plots...")
    print("=" * 80)
    plot_k_pareto_curves(data, layers, Path(args.output_dir), 
                        max_mse=args.max_mse, max_l0=args.max_l0,
                        min_mse=args.min_mse, min_l0=args.min_l0)
    
    # Print summary
    print_ablation_summary(data, layers)
    
    print("\n" + "=" * 80)
    print("Ablation analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main() 