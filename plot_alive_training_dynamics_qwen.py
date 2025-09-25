#!/usr/bin/env python3
"""
Plot training dynamics of alive dictionary components for specific runs.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

from settings import settings

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12


def get_training_data(run_id: str, project: str, metric_name: str):
    """
    Get training data for a specific metric from a wandb run.
    
    Args:
        run_id: Wandb run ID
        project: Wandb project name
        metric_name: Name of the metric to extract
    
    Returns:
        Tuple of (steps, values) or (None, None) if not found
    """
    try:
        # Login to wandb
        wandb.login(key=settings.wandb_api_key)
        api = wandb.Api()
        
        # Get the run
        run = api.run(f"{project}/{run_id}")
        
        # Get the history (training data)
        history = run.history()
        
        if metric_name in history.columns:
            # Filter out NaN values
            valid_data = history[[metric_name, '_step']].dropna()
            steps = valid_data['_step'].values
            values = valid_data[metric_name].values
            return steps, values
        else:
            print(f"Metric '{metric_name}' not found in run {run_id}")
            print(f"Available metrics: {list(history.columns)}")
            return None, None
            
    except Exception as e:
        print(f"Error loading data for run {run_id}: {e}")
        return None, None


def plot_alive_components_training(run_configs: list, metric_name: str, 
                                 output_path: str = "alive_components_training.png"):
    """
    Plot alive dictionary components during training for multiple runs.
    
    Args:
        run_configs: List of dicts with 'run_id', 'project', 'label', 'color', 'linestyle'
        metric_name: Name of the metric to plot
        output_path: Output file path
    """
    
    # Create figure with 2:1 aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    # Store line objects for custom legend ordering
    lines = []
    labels = []
    
    for config in run_configs:
        run_id = config['run_id']
        project = config['project']
        label = config['label']
        color = config['color']
        linestyle = config.get('linestyle', '-')  # Default to solid line
        linewidth = config.get('linewidth', 2.5)  # Default line width
        
        print(f"Loading data for {label} (run: {run_id})...")
        steps, values = get_training_data(run_id, project, metric_name)
        
        if steps is not None and values is not None:
            # Plot line without markers (too many points)
            line, = ax.plot(steps, values, color=color, linewidth=linewidth,
                           linestyle=linestyle, alpha=0.9)
            lines.append(line)
            labels.append(label)
            print(f"  Plotted {len(steps)} data points")
        else:
            print(f"  No data found for {label}")
    
    # Reorder for legend: TopK first column, Probabilistic TopK second column
    topk_indices = [i for i, label in enumerate(labels) if 'Probabilistic' not in label]
    prob_indices = [i for i, label in enumerate(labels) if 'Probabilistic' in label]
    
    # Create ordered lists for legend
    ordered_lines = []
    ordered_labels = []
    
    # Add all TopK entries first (in reverse order: k=32, k=16, k=8), then all Probabilistic TopK entries
    for idx in reversed(topk_indices):
        ordered_lines.append(lines[idx])
        ordered_labels.append(labels[idx])
    
    for idx in reversed(prob_indices):
        ordered_lines.append(lines[idx])
        ordered_labels.append(labels[idx])
    
    # Set axis limits to start from 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Formatting with increased padding
    ax.set_xlabel('Training Steps', fontsize=20, labelpad=15)
    ax.set_ylabel('Alive Dictionary Components → (better)', fontsize=20, labelpad=15)
    ax.set_title('Alive Dictionary Components During Training', fontsize=24, pad=25)
    
    # Create legend with custom order
    ax.legend(ordered_lines, ordered_labels, fontsize=14, loc='best', ncol=2)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show values in thousands for better readability
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'
    ))
    
    # Remove zero from y-axis ticks to avoid duplicate at origin (keep only x-axis zero)
    yticks = list(ax.get_yticks())
    if 0 in yticks:
        yticks.remove(0)
    ax.set_yticks(yticks)
    
    # Clean up axes appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved plot to: {output_path}")
    
    # Also save as SVG
    svg_path = output_path.replace('.png', '.svg')
    plt.savefig(svg_path, bbox_inches='tight', format='svg')
    
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Plot alive dictionary components during training"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="raymondl/gpt2-small",
        help="Wandb project name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="alive_components_training.png",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Configuration for the runs to plot
    run_configs = [
        # K=8 (dash-dot lines, thinnest)
        {
            'run_id': 'j4rbvpod',  # topk_k_8_interpret
            'project': args.project,
            'label': 'TopK (k=16)',
            'color': '#1f77b4',  # Blue
            'linestyle': (0, (3, 1, 1, 1)),  # Custom dash-dot pattern
            'linewidth': 1.5
        },
        {
            'run_id': 'u3mg4omd',  # probabilistic_k_8
            'project': args.project,
            'label': 'Probabilistic TopK (k=16)',
            'color': '#d62728',  # Red
            'linestyle': (0, (3, 1, 1, 1)),  # Custom dash-dot pattern
            'linewidth': 1.5
        },
        # K=16 (dashed lines, medium)
        {
            'run_id': '9hxeoxku',  # topk_k_16
            'project': args.project,
            'label': 'TopK (k=32)',
            'color': '#1f77b4',  # Blue
            'linestyle': (0, (5, 2)),        # Custom dashed pattern
            'linewidth': 2.5
        },
        {
            'run_id': 'q2kbngcu',  # probabilistic_k_16
            'project': args.project,
            'label': 'Probabilistic TopK (k=32)',
            'color': '#d62728',  # Red
            'linestyle': (0, (5, 2)),        # Custom dashed pattern
            'linewidth': 2.5
        },
        # K=32 (solid lines, thick)
        {
            'run_id': '5g56nzay',  # topk_k_32_interpret
            'project': args.project,
            'label': 'TopK (k=64)',
            'color': '#1f77b4',  # Blue
            'linestyle': '-',    # Solid
            'linewidth': 4.0
        },
        {
            'run_id': '5zyfw6zx',  # probabilistic_k_32
            'project': args.project,
            'label': 'Probabilistic TopK (k=64)',
            'color': '#d62728',  # Red
            'linestyle': '-',    # Solid
            'linewidth': 4.0
        }
    ]
    
    # Metric name for alive dictionary components at blocks.8.hook_resid_pre
    metric_name = 'train/alive_dict_components/blocks.26.hook_resid_pre'
    
    print("=" * 80)
    print("Plotting Alive Dictionary Components Training Dynamics")
    print("=" * 80)
    print(f"Metric: {metric_name}")
    print(f"Project: {args.project}")
    
    # Create the plot
    plot_alive_components_training(run_configs, metric_name, args.output)
    
    print("\n" + "=" * 80)
    print("Plot complete!")
    print("=" * 80)


if __name__ == "__main__":
    main() 