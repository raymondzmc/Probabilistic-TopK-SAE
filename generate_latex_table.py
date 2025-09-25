#!/usr/bin/env python3
"""
Generate LaTeX table with activation scores by bin and correlations.
"""

import json
import numpy as np
from pathlib import Path


def load_calibration_data(base_path):
    """Load calibration data from a model directory."""
    metrics_path = Path(base_path) / "calibration_metrics.json"
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    sae_pos = list(metrics.keys())[0]
    bucket_metrics = metrics[sae_pos]['bucket_metrics']
    correlation = metrics[sae_pos]['correlation']
    
    # Extract mean scores for each bucket
    bucket_scores = []
    for bm in bucket_metrics:
        bucket_scores.append(bm['avg_detection_score'])
    
    return bucket_scores, correlation


def main():
    # Model configurations for both sets
    models_set1 = {
        'ReLU': 'artifacts/calibration_gpt2_relu_30_multi',
        'Gated': 'artifacts/calibration_gpt2_gated_9e-02_multi',
        'TopK': 'artifacts/calibration_gpt2_topk_k8_multi',
        'Probabilistic TopK': 'artifacts/calibration_gpt2_probabilistic_k8_multi',
    }
    
    models_set2 = {
        'ReLU': 'artifacts/calibration_gpt2_relu_20_multi',
        'Gated': 'artifacts/calibration_gpt2_gated_6e-02_multi',
        'TopK': 'artifacts/calibration_gpt2_topk_k_16_multi',
        'Probabilistic TopK': 'artifacts/calibration_gpt2_probabilistic_k_16_multi',
    }
    
    # Load data for both sets
    data_set1 = {}
    data_set2 = {}
    
    for model_name, model_path in models_set1.items():
        try:
            scores, corr = load_calibration_data(model_path)
            data_set1[model_name] = {'scores': scores, 'correlation': corr}
        except Exception as e:
            print(f"Warning: Could not load {model_name} from Set 1: {e}")
    
    for model_name, model_path in models_set2.items():
        try:
            scores, corr = load_calibration_data(model_path)
            data_set2[model_name] = {'scores': scores, 'correlation': corr}
        except Exception as e:
            print(f"Warning: Could not load {model_name} from Set 2: {e}")
    
    # Generate LaTeX table
    latex_content = generate_latex_table(data_set1, data_set2)
    
    # Save to file
    output_path = Path('artifacts') / 'calibration_table.tex'
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to: {output_path}")
    print("\nLaTeX table content:")
    print(latex_content)


def generate_latex_table(data_set1, data_set2):
    """Generate LaTeX table with both sets."""
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Interpretability Scores by Activation Percentile and Correlation}
\label{tab:calibration_scores}
\begin{tabular}{l|ccccc|c}
\hline
\textbf{Method} & \textbf{0-20\%} & \textbf{20-40\%} & \textbf{40-60\%} & \textbf{60-80\%} & \textbf{80-100\%} & \textbf{Correlation} \\
\hline
\multicolumn{7}{c}{\textbf{Set 1: $K=8$, $\lambda=30/0.09$}} \\
\hline
"""
    
    # Add Set 1 data
    model_order = ['ReLU', 'Gated', 'TopK', 'Probabilistic TopK']
    for model_name in model_order:
        if model_name in data_set1:
            data = data_set1[model_name]
            scores = data['scores']
            corr = data['correlation']
            
            # Format scores to 3 decimal places
            score_strs = [f"{score:.3f}" for score in scores]
            corr_str = f"{corr:.3f}"
            
            latex += f"{model_name} & " + " & ".join(score_strs) + f" & {corr_str} \\\\\n"
    
    latex += r"""
\hline
\multicolumn{7}{c}{\textbf{Set 2: $K=16$, $\lambda=20/0.06$}} \\
\hline
"""
    
    # Add Set 2 data
    for model_name in model_order:
        if model_name in data_set2:
            data = data_set2[model_name]
            scores = data['scores']
            corr = data['correlation']
            
            # Format scores to 3 decimal places
            score_strs = [f"{score:.3f}" for score in scores]
            corr_str = f"{corr:.3f}"
            
            latex += f"{model_name} & " + " & ".join(score_strs) + f" & {corr_str} \\\\\n"
    
    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    
    return latex.strip()


if __name__ == "__main__":
    main() 