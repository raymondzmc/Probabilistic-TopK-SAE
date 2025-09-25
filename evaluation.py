import os
import asyncio
import numpy as np
import torch
import wandb
import argparse
import json
from pathlib import Path
from settings import settings
from torch.nn.functional import mse_loss
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from data import create_dataloaders
from models import SAETransformer, SAETransformerOutput
from utils.io import (
    load_config, 
    save_activation_data_to_wandb,
    save_metrics_to_wandb,
    save_explanations_to_wandb,
    load_activation_data_from_wandb,
    load_metrics_from_wandb,
    load_explanations_from_wandb
)
from utils.metrics import explained_variance, get_activations_for_sae_type, compute_alive_dictionary_indices
from utils.plotting import create_pareto_plots
from auto_interp.explainers.features import FeatureRecord, Feature
from auto_interp.explainers.explainer import DefaultExplainer, ExplainerResult
from auto_interp.clients import OpenAIClient, TogetherAIClient
from auto_interp.explainers.sampler import stratified_sample_by_max_activation
from auto_interp.scorers.classifier.detection import DetectionScorer
# from auto_interp.scorers.classifier.fuzz import FuzzingScorer  # Disabled - focusing on detection only


def create_calibration_plots(all_explanation_scores: dict, output_path: str, run_id: str):
    """
    Create calibration plots showing how interpretability scores vary with activation strength.
    
    Args:
        all_explanation_scores: Dictionary mapping SAE positions to list of score dictionaries
        output_path: Directory to save plots
        run_id: Run identifier for naming files
    """
    output_dir = Path(output_path) / "calibration_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    for sae_pos, scores_list in all_explanation_scores.items():
        # Filter to only entries with calibration data
        calibration_entries = [s for s in scores_list if 'bucket_detection_scores' in s]
        multi_bucket_entries = [s for s in scores_list if 'multi_bucket' in s and s.get('multi_bucket', False)]
        
        if not calibration_entries:
            continue
            
        # Collect data for plotting
        bucket_percentiles = calibration_entries[0]['bucket_percentiles']
        bucket_midpoints = [(bucket_percentiles[i] + bucket_percentiles[i+1]) / 2 
                           for i in range(len(bucket_percentiles)-1)]
        
        # Aggregate scores across neurons
        detection_scores_by_bucket = {i: [] for i in range(len(bucket_midpoints))}
        # fuzz_scores_by_bucket = {i: [] for i in range(len(bucket_midpoints))}  # Disabled - focusing on detection only
        
        for entry in calibration_entries:
            for bucket_idx, score in entry['bucket_detection_scores'].items():
                # Convert bucket_idx to int (JSON saves integer keys as strings)
                bucket_idx_int = int(bucket_idx) if isinstance(bucket_idx, str) else bucket_idx
                detection_scores_by_bucket[bucket_idx_int].append(score)
            # for bucket_idx, score in entry.get('bucket_fuzz_scores', {}).items():
            #     fuzz_scores_by_bucket[bucket_idx].append(score)
        
        # Compute mean and std for each bucket
        detection_means = []
        detection_stds = []
        fuzz_means = []
        fuzz_stds = []
        
        for i in range(len(bucket_midpoints)):
            if detection_scores_by_bucket[i]:
                detection_means.append(np.mean(detection_scores_by_bucket[i]))
                detection_stds.append(np.std(detection_scores_by_bucket[i]))
            else:
                detection_means.append(np.nan)
                detection_stds.append(np.nan)
                
            # Fuzz scores disabled - just append NaN
            fuzz_means.append(np.nan)
            fuzz_stds.append(np.nan)
        
        # Create figure with single subplot (detection only)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot Detection calibration
        ax.errorbar(bucket_midpoints, detection_means, yerr=detection_stds, 
                    marker='o', markersize=8, capsize=5, capthick=2, linewidth=2)
        ax.set_xlabel('Activation Percentile', fontsize=12)
        ax.set_ylabel('Detection Score', fontsize=12)
        ax.set_title(f'Detection Score vs Activation Strength\n{sae_pos}', fontsize=14)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add ideal calibration line (if perfectly calibrated, higher activations = higher scores)
        ax.plot([0, 100], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f'calibration_{sae_pos.replace(".", "_")}_{run_id}.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'calibration_{sae_pos.replace(".", "_")}_{run_id}.svg', bbox_inches='tight')
        plt.close()
        
        # Create heatmap showing per-neuron calibration
        if len(calibration_entries) > 5:  # Only create heatmap if we have enough neurons
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Prepare data for heatmap
            n_neurons = len(calibration_entries)
            n_buckets = len(bucket_midpoints)
            
            detection_matrix = np.full((n_neurons, n_buckets), np.nan)
            fuzz_matrix = np.full((n_neurons, n_buckets), np.nan)
            
            for i, entry in enumerate(calibration_entries):
                for bucket_idx, score in entry['bucket_detection_scores'].items():
                    # Convert bucket_idx to int (JSON saves integer keys as strings)
                    bucket_idx_int = int(bucket_idx) if isinstance(bucket_idx, str) else bucket_idx
                    detection_matrix[i, bucket_idx_int] = score
                # for bucket_idx, score in entry.get('bucket_fuzz_scores', {}).items():
                #     fuzz_matrix[i, bucket_idx] = score
            
            # Create heatmap (detection only)
            sns.heatmap(detection_matrix, ax=ax, cmap='viridis', vmin=0, vmax=1,
                       xticklabels=[f'{int(p)}%' for p in bucket_midpoints],
                       yticklabels=False, cbar_kws={'label': 'Detection Score'})
            ax.set_xlabel('Activation Percentile', fontsize=12)
            ax.set_ylabel('Neuron Index', fontsize=12)
            ax.set_title(f'Detection Scores by Activation Percentile\n{sae_pos}', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'calibration_heatmap_{sae_pos.replace(".", "_")}_{run_id}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create multi-bucket explanation comparison if available
        if multi_bucket_entries:
            print(f"Creating multi-bucket explanation comparison for {sae_pos}...")
            
            # Create a text-based visualization of how explanations change across buckets
            comparison_path = output_dir / f'multi_bucket_explanations_{sae_pos.replace(".", "_")}_{run_id}.txt'
            
            with open(comparison_path, 'w') as f:
                f.write(f"Multi-Bucket Explanation Analysis for {sae_pos}\n")
                f.write("=" * 80 + "\n\n")
                
                for entry in multi_bucket_entries:
                    f.write(f"Neuron {entry['neuron_idx']}\n")
                    f.write("-" * 40 + "\n")
                    
                    if 'bucket_explanations' in entry:
                        bucket_explanations = entry['bucket_explanations']
                        bucket_det_scores = entry.get('bucket_detection_scores', {})
                        bucket_fuzz_scores = entry.get('bucket_fuzz_scores', {})
                        
                        for bucket_idx in sorted(bucket_explanations.keys(), key=lambda x: int(x) if isinstance(x, str) else x):
                            # Convert bucket_idx to int if it's a string (from JSON)
                            bucket_idx_int = int(bucket_idx) if isinstance(bucket_idx, str) else bucket_idx
                            percentile_start = bucket_idx_int * 100 // len(bucket_explanations)
                            percentile_end = (bucket_idx_int + 1) * 100 // len(bucket_explanations)
                            
                            f.write(f"\nBucket {bucket_idx} ({percentile_start}-{percentile_end}% activation):\n")
                            if bucket_idx in bucket_det_scores:
                                f.write(f"  Detection Score: {bucket_det_scores[bucket_idx]:.3f}\n")
                            # if bucket_idx in bucket_fuzz_scores:
                            #     f.write(f"  Fuzz Score: {bucket_fuzz_scores[bucket_idx]:.3f}\n")
                            f.write(f"  Explanation: {bucket_explanations[bucket_idx]}\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
            
            print(f"Multi-bucket explanations saved to: {comparison_path}")
    
    print(f"\nCalibration plots saved to: {output_dir}")


def run_evaluation(args: argparse.Namespace) -> None:
    """
    Run SAE evaluation including activation data collection, neuron explanation generation, and analysis.
    """
    
    # Set Wandb cache directories to use output_path instead of home directory
    output_path_abs = Path(args.output_path).absolute()
    output_path_abs.mkdir(parents=True, exist_ok=True)
    
    wandb_cache_dir = output_path_abs / "wandb_cache"
    wandb_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables BEFORE any Wandb operations
    os.environ["WANDB_CACHE_DIR"] = str(wandb_cache_dir)
    os.environ["WANDB_DATA_DIR"] = str(wandb_cache_dir)
    os.environ["WANDB_DIR"] = str(wandb_cache_dir)
    os.environ["TMPDIR"] = str(output_path_abs)  # Also set general temp directory
    
    print(f"Using Wandb cache directory: {wandb_cache_dir}")
    print(f"Using temp directory: {output_path_abs}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    wandb.login(key=settings.wandb_api_key)
    api = wandb.Api()
    runs = api.runs(args.wandb_project)

    if args.filter_runs_by_name is not None:
        old_len = len(runs)
        runs = [run for run in runs if args.filter_runs_by_name in run.name]
        print(f"Found {len(runs)}/{old_len} runs matching filter: {args.filter_runs_by_name}")

    # Collect all metrics for pareto plots
    all_run_metrics = []

    for run in runs:
        run_id = run.id
        run_config = run.config
        run_config['data']['n_eval_samples'] = args.n_eval_samples
        try:
            model = SAETransformer.from_wandb(f"{args.wandb_project}/{run_id}").to(device)
            model.saes.eval()
            if args.sae_position is None:
                raw_sae_positions = model.raw_sae_positions
            else:
                assert args.sae_position in model.raw_sae_positions, f"SAE position {args.sae_position} not found in model"
                raw_sae_positions = [args.sae_position]
        except Exception as e:
            print(f"Error loading model from Wandb: {e}")
            continue
        
        # Override n_train_samples if specified (to avoid slow data skipping)
        if args.override_n_train_samples is not None:
            print(f"Overriding n_train_samples: {run_config['data']['n_train_samples']} -> {args.override_n_train_samples}")
            run_config['data']['n_train_samples'] = args.override_n_train_samples
            
        config: Config = load_config(run_config, Config)

        # Initialize Wandb run for this specific run (resume original run)
        wandb.init(
            project=args.wandb_project.split("/")[-1],  # Extract project name
            entity=args.wandb_project.split("/")[0],    # Extract entity  
            id=run_id,  # Use the same run ID
            resume="allow",  # Allow resuming existing run
            reinit="finish_previous",
            dir=str(wandb_cache_dir)  # Explicitly set Wandb's working directory
        )

        metrics = {}
        accumulated_data = None
        all_token_ids = None
        loaded_metrics = None
        tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)

        # Try to load existing data from Wandb
        print(f"Attempting to load activation data from Wandb for run {run_id}...")
        try:
            accumulated_data, all_token_ids = load_activation_data_from_wandb(
                run_id, project=args.wandb_project, output_path=args.output_path
            )
            
            print(f"Successfully loaded activation data from Wandb run files")
            if all_token_ids is not None:
                print(f"Loaded token IDs from Wandb run files")
                
        except (FileNotFoundError, RuntimeError) as e:
            print(f"No existing activation data found: {e}")
            print("Will compute activation data and metrics from scratch")
        
        # Try to load existing metrics separately
        try:
            loaded_metrics = load_metrics_from_wandb(run_id, project=args.wandb_project, output_path=args.output_path)
            if loaded_metrics is not None and not args.force_recompute:
                metrics = loaded_metrics
                print(f"Loaded existing metrics from Wandb for {len(metrics)} SAE positions")
            elif loaded_metrics is not None and args.force_recompute:
                print(f"Found existing metrics but --force_recompute is set, will recompute")
        except Exception as e:
            print(f"No existing metrics found: {e}")
            print("Will compute metrics from scratch")

        # Always generate explanations from scratch (skip loading from wandb)
        all_explanation_scores = None
        if args.generate_explanations:
            print("Generating explanations from scratch (loading from wandb disabled)")
            # Only force recomputation if no activation data exists
            if accumulated_data is None:
                print("No activation data found - will compute from scratch")
                if not args.save_activation_data:
                    print("Warning: --save_activation_data not set, but it's required for explanation generation")
            else:
                print("Using existing activation data for explanation generation")
        if accumulated_data is None or len(metrics) == 0:
            print(f"Obtaining features for {run_id}")

            # Load model and dataloader
            _, eval_loader = create_dataloaders(data_config=config.data, global_seed=config.seed, quick_eval=True)
            total_tokens = 0
            all_token_ids: list[list[str]] = []

            # Create placeholder tensors for efficient batch accumulation
            # Note: For feature extraction, 'nonzero_activations' stores probabilities for Bayesian SAEs, activations for ReLU SAEs
            accumulated_data = {}
            for sae_pos in raw_sae_positions:
                accumulated_data[sae_pos] = {
                    'nonzero_activations': [],
                    'data_indices': [],
                    'neuron_indices': [],
                }
                metrics[sae_pos] = {
                    'alive_dict_components': set(),
                    'sparsity_l0': 0.0,
                    'mse': 0.0,
                    'explained_variance': 0.0,
                }

            total_tokens = 0
            for batch in tqdm(eval_loader, desc="Processing batches"):
                token_ids = batch[config.data.column_name].to(device)
                
                # Reshape token_ids to break into chunks of window_size
                batch_size, seq_len = token_ids.shape
                if seq_len % args.window_size != 0:
                    raise ValueError(f"Sequence length {seq_len} is not divisible by window_size {args.window_size}")

                num_chunks = seq_len // args.window_size
                chunked_batch_size = batch_size * num_chunks
                token_ids_chunked = token_ids.view(batch_size, num_chunks, args.window_size)
                token_ids_chunked = token_ids_chunked.reshape(chunked_batch_size, args.window_size)
                
                n_tokens = token_ids_chunked.shape[0] * token_ids_chunked.shape[1]
                total_tokens += n_tokens

                # Run through the SAE-augmented model
                with torch.no_grad():
                    output: SAETransformerOutput = model.forward(
                        tokens=token_ids_chunked,
                        sae_positions=model.raw_sae_positions,
                        compute_loss=True,
                    )

                for sae_pos in raw_sae_positions:
                    sae_output = output.sae_outputs[sae_pos]
                    
                    # Compute MSE using the same logic as utils/metrics.py
                    mse_val = mse_loss(
                        sae_output.output,
                        sae_output.input,
                        reduction='mean'
                    ).item()
                    metrics[sae_pos]['mse'] += mse_val * n_tokens
                    
                    # Compute explained variance using the shared function from utils.metrics
                    exp_var = explained_variance(
                        sae_output.output,
                        sae_output.input,
                        layer_norm_flag=False
                    ).mean().item()
                    metrics[sae_pos]['explained_variance'] += exp_var * n_tokens
                    
                    # Get activations using the shared function from utils.metrics
                    acts = get_activations_for_sae_type(sae_output, config.saes.sae_type)

                    # Compute L0 sparsity using the same logic as utils/metrics.py
                    l0_val = torch.norm(acts, p=0, dim=-1).mean().item()
                    metrics[sae_pos]['sparsity_l0'] += l0_val * n_tokens
                    
                    # Compute alive dictionary components using the shared helper function
                    alive_indices = compute_alive_dictionary_indices(acts)
                    metrics[sae_pos]['alive_dict_components'].update(alive_indices)

                    if args.save_activation_data:
                        # Collect non-zero activations for explanation generation
                        data_indices, neuron_indices = acts.sum(1).nonzero(as_tuple=True)
                        if data_indices.numel() > 0:
                            # Extract all relevant activations at once (N, seq_len)
                            nonzero_activations = acts[data_indices, :, neuron_indices]

                            # Add the offset to the data indices for global indexing
                            global_data_indices = data_indices + len(all_token_ids)

                            # Accumulate tensors for this SAE position
                            accumulated_data[sae_pos]['nonzero_activations'].append(nonzero_activations.to(torch.float16).cpu())
                            accumulated_data[sae_pos]['data_indices'].append(global_data_indices.cpu())
                            accumulated_data[sae_pos]['neuron_indices'].append(neuron_indices.cpu())

                # Store tokenized sequences for explanation generation
                chunked_tokens = [tokenizer.convert_ids_to_tokens(token_ids_chunked[i]) for i in range(chunked_batch_size)]
                all_token_ids.extend(chunked_tokens)

            for sae_pos in raw_sae_positions:
                if args.save_activation_data:
                    print(f"  Concatenating activation data for {sae_pos}")
                    print(f"    Number of chunks: {len(accumulated_data[sae_pos]['nonzero_activations'])}")
                    accumulated_data[sae_pos]['nonzero_activations'] = torch.cat(accumulated_data[sae_pos]['nonzero_activations'], dim=0).contiguous()
                    accumulated_data[sae_pos]['data_indices'] = torch.cat(accumulated_data[sae_pos]['data_indices'], dim=0).contiguous()
                    accumulated_data[sae_pos]['neuron_indices'] = torch.cat(accumulated_data[sae_pos]['neuron_indices'], dim=0).contiguous()
                    print(f"    Final shape - nonzero_activations: {accumulated_data[sae_pos]['nonzero_activations'].shape}")
                    print(f"    Final shape - neuron_indices: {accumulated_data[sae_pos]['neuron_indices'].shape}")

                metrics[sae_pos]['sparsity_l0'] /= total_tokens
                metrics[sae_pos]['mse'] /= total_tokens
                metrics[sae_pos]['explained_variance'] /= total_tokens
                
                # Convert alive components set to count and proportion
                alive_components = metrics[sae_pos]['alive_dict_components']
                num_alive = len(alive_components)
                total_dict_size = model.saes[sae_pos.replace(".", "-")].n_dict_components
                metrics[sae_pos]['alive_dict_components'] = num_alive
                metrics[sae_pos]['alive_dict_components_proportion'] = num_alive / total_dict_size
            
            # Always save metrics
            print("Saving metrics to Wandb...")
            try:
                save_metrics_to_wandb(metrics=metrics, output_path=args.output_path)
            except Exception as e:
                print(f"Warning: Failed to upload metrics to Wandb: {e}")

            # Save activation data to Wandb
            if args.save_activation_data:
                print("Saving accumulated activation data to Wandb...")
                save_activation_data_to_wandb(
                    accumulated_data=accumulated_data,
                    all_token_ids=all_token_ids,
                    output_path=args.output_path,
                    skip_upload=args.skip_upload,
                    chunk_upload=not args.no_chunk_upload  # Default is True unless disabled
                )

        # Collect metrics for pareto plot
        run_metrics = {
            'run_id': run_id,
            'run_name': run.name,
            'config': config,
            'metrics': metrics
        }
        
        all_run_metrics.append(run_metrics)
        
        if args.generate_explanations and all_explanation_scores is None:
            if accumulated_data is None:
                print("No activation data found, skipping explanation generation")
                continue
            
            # Validate that accumulated_data has the required structure
            print(f"\nValidating activation data for explanation generation...")
            valid_data = True
            for sae_pos in raw_sae_positions:
                if sae_pos not in accumulated_data:
                    print(f"  ERROR: {sae_pos} not found in accumulated_data")
                    valid_data = False
                    continue
                
                required_keys = ['nonzero_activations', 'data_indices', 'neuron_indices']
                missing_keys = [k for k in required_keys if k not in accumulated_data[sae_pos]]
                if missing_keys:
                    print(f"  ERROR: Missing keys for {sae_pos}: {missing_keys}")
                    valid_data = False
                else:
                    # Check if data is empty
                    if len(accumulated_data[sae_pos]['neuron_indices']) == 0:
                        print(f"  ERROR: Empty neuron_indices for {sae_pos}")
                        valid_data = False
                    else:
                        print(f"  ✓ {sae_pos}: Found {len(accumulated_data[sae_pos]['neuron_indices'])} activation records")
            
            if not valid_data:
                print("ERROR: Invalid or missing activation data. Cannot generate explanations.")
                print("Make sure to run with --save_activation_data flag")
                continue
            
            # Initialize dict to store all explanation scores for this run
            all_explanation_scores = {}

            # Initialize explainer
            explainer = DefaultExplainer(
                client=OpenAIClient(
                    api_key=settings.openai_api_key,
                    model=args.explanation_model,
                ),
                tokenizer=tokenizer,
                cot=True,
                threshold=0.3,
                activations=False,
                temperature=0.0,
            )

            for sae_pos in raw_sae_positions:
                print(f"\nProcessing SAE position: {sae_pos}")
                
                if sae_pos not in accumulated_data:
                    print(f"  ERROR: {sae_pos} not found in accumulated_data")
                    print(f"  Available keys: {list(accumulated_data.keys())}")
                    continue
                    
                data = accumulated_data[sae_pos]
                all_explanation_scores[sae_pos] = []
                
                # Debug: Check data structure
                print(f"  Data keys: {list(data.keys())}")
                if 'neuron_indices' in data:
                    print(f"  Neuron indices shape: {data['neuron_indices'].shape if hasattr(data['neuron_indices'], 'shape') else 'N/A'}")
                    print(f"  Neuron indices type: {type(data['neuron_indices'])}")
                
                # Count occurrences of each neuron and calculate total activation
                unique_neurons = torch.unique(data['neuron_indices'], return_counts=False)
                print(f"  Found {len(unique_neurons)} unique neurons")
                
                # Efficient computation using scatter_reduce
                num_neurons = unique_neurons.max().item() + 1
                seq_len = data['nonzero_activations'].shape[1]
                    
                # Create unique (neuron, sequence) pairs to avoid double counting
                neuron_sequence_pairs = torch.stack([data['neuron_indices'], data['data_indices']], dim=1)
                unique_pairs = torch.unique(neuron_sequence_pairs, dim=0)
                
                # Count sequences per neuron using bincount
                neuron_indices_from_pairs = unique_pairs[:, 0]
                neuron_sequence_counts = torch.bincount(neuron_indices_from_pairs, minlength=num_neurons)
                
                # Get counts only for neurons that appear
                counts_for_active_neurons = neuron_sequence_counts[unique_neurons]
                
                min_required_examples = args.num_features_to_explain + args.num_positive_examples
                sufficient_examples_mask = counts_for_active_neurons >= min_required_examples
                filtered_indices = torch.arange(len(unique_neurons))[sufficient_examples_mask]
                
                if len(filtered_indices) == 0:
                    print(f"  No neurons found with at least {min_required_examples} activated sequences")
                    continue
                
                num_to_sample = min(args.num_neurons, len(filtered_indices))
                torch.manual_seed(config.seed)
                random_perm = torch.randperm(len(filtered_indices))
                sampled_indices = filtered_indices[random_perm[:num_to_sample]]
                sampled_neurons = unique_neurons[sampled_indices]
                print(f"SAE position {sae_pos}: {len(unique_neurons)} total neurons, {len(filtered_indices)} with ≥{min_required_examples} examples, randomly sampling {len(sampled_neurons)}")
                print(f"  Processing {len(sampled_neurons)} neurons for explanation...")
                
                # Initialize running statistics
                detection_scores = []
                # fuzz_scores = []  # Disabled - focusing on detection only
                
                # Process each neuron for explanation
                pbar = tqdm(sampled_neurons, desc="Processing neurons")
                for i, neuron_idx in enumerate(pbar):
                    neuron_idx_item = neuron_idx.item()
                    
                    # Update progress bar with running statistics
                    stats_str = ""
                    if detection_scores:
                        stats_str += f"det_mean={np.mean(detection_scores):.3f}"
                    # if fuzz_scores:
                    #     if stats_str:
                    #         stats_str += ", "
                    #     stats_str += f"fuzz_mean={np.mean(fuzz_scores):.3f}"
                    if stats_str:
                        pbar.set_postfix_str(stats_str)
                    
                    try:
                        feature = Feature(sae_pos=sae_pos, neuron_idx=neuron_idx_item)
                        
                        if args.stratified_calibration:
                            # Use stratified approach for calibration analysis
                            result = FeatureRecord.from_data_stratified(
                                data=data,
                                feature=feature,
                                all_token_ids=all_token_ids,
                                neuron_idx=neuron_idx,
                                num_explanation_examples=args.num_features_to_explain,
                                num_examples_per_bucket=args.examples_per_bucket,
                                num_buckets=args.calibration_buckets,
                                min_examples_required=args.num_features_to_explain,
                                seed=config.seed,
                            )
                            
                            if result is None:
                                print(f"  Skipping neuron {neuron_idx_item} - not enough examples")
                                continue
                                
                            feature_record, bucket_examples = result
                        else:
                            # Use regular approach
                            min_required_examples = args.num_features_to_explain + args.num_positive_examples
                            feature_record = FeatureRecord.from_data(
                                data=data,
                                feature=feature,
                                all_token_ids=all_token_ids,
                                neuron_idx=neuron_idx,
                                num_explanation_examples=args.num_features_to_explain,
                                num_positive_examples=args.num_positive_examples,
                                num_negative_examples=args.num_negative_examples,
                                stratified_quantiles=args.stratified_quantiles,
                                min_examples_required=min_required_examples,
                                seed=config.seed,
                            )
                            
                            # Skip if we couldn't create a valid feature record
                            if feature_record is None:
                                print(f"  Skipping neuron {neuron_idx_item} - not enough examples")
                                continue

                        # Create scoring client for Detection and Fuzz scorers
                        score_client = TogetherAIClient(
                            api_key=settings.together_ai_api_key,  # Use together API key
                            model=args.scoring_model
                        )
                        
                        if args.stratified_calibration and args.multi_bucket_explanations:
                            # Multi-bucket explanation approach - generate explanation from each bucket
                            print(f"  Generating explanations from each bucket for neuron {neuron_idx_item}")
                            
                            bucket_explanations = {}
                            bucket_detection_scores = {}
                            # bucket_fuzz_scores = {}  # Disabled - focusing on detection only
                            
                            for bucket_idx in range(args.calibration_buckets):
                                if len(bucket_examples.get(bucket_idx, [])) < args.num_features_to_explain:
                                    print(f"    Bucket {bucket_idx}: Insufficient examples ({len(bucket_examples.get(bucket_idx, []))} < {args.num_features_to_explain})")
                                    continue
                                
                                # Create a feature record for this bucket
                                bucket_record = FeatureRecord(feature=feature)
                                bucket_record.max_activation = feature_record.max_activation
                                
                                # Use first num_features_to_explain for explanation
                                bucket_record.explanation_examples = bucket_examples[bucket_idx][:args.num_features_to_explain]
                                # Use remaining for evaluation (or all if not enough)
                                if len(bucket_examples[bucket_idx]) > args.num_features_to_explain:
                                    bucket_record.positive_examples = bucket_examples[bucket_idx][args.num_features_to_explain:]
                                else:
                                    bucket_record.positive_examples = bucket_examples[bucket_idx]
                                bucket_record.negative_examples = feature_record.negative_examples[:args.examples_per_bucket]
                                
                                # Generate explanation for this bucket
                                bucket_percentile = f"{bucket_idx*100//args.calibration_buckets}-{(bucket_idx+1)*100//args.calibration_buckets}%"
                                print(f"    Generating explanation for bucket {bucket_idx} ({bucket_percentile})...")
                                
                                bucket_explanation_result: ExplainerResult = asyncio.run(explainer(bucket_record))
                                bucket_explanations[bucket_idx] = bucket_explanation_result.explanation
                                
                                print(f"      Explanation: {bucket_explanations[bucket_idx][:80]}...")
                                
                                # Score this bucket's explanation on its own examples
                                if len(bucket_record.positive_examples) > 0:
                                    # Detection Score
                                    detection_scorer = DetectionScorer(
                                        client=score_client,
                                        verbose=False,
                                        batch_size=5,
                                        use_structured_output=True,
                                        temperature=0.0,
                                    )
                                    detection_result = asyncio.run(detection_scorer(bucket_explanation_result))
                                    bucket_detection_scores[bucket_idx] = detection_result.score
                                    
                                    # Fuzz Score - DISABLED (focusing on detection only)
                                    # fuzz_scorer = FuzzingScorer(
                                    #     client=score_client,
                                    #     verbose=False,
                                    #     batch_size=5,
                                    #     threshold=0.3,
                                    #     use_structured_output=True,
                                    #     temperature=0.0,
                                    # )
                                    # fuzz_result = asyncio.run(fuzz_scorer(bucket_explanation_result))
                                    # bucket_fuzz_scores[bucket_idx] = fuzz_result.score
                                    
                                    print(f"      Detection: {bucket_detection_scores[bucket_idx]:.3f}")
                                else:
                                    print(f"      No evaluation examples available for scoring")
                            
                            # Store multi-bucket results
                            all_explanation_scores[sae_pos].append({
                                'neuron_idx': neuron_idx_item,
                                'multi_bucket': True,
                                'bucket_explanations': bucket_explanations,
                                'bucket_detection_scores': bucket_detection_scores,
                                # 'bucket_fuzz_scores': bucket_fuzz_scores,  # Disabled - focusing on detection only
                                'bucket_boundaries': feature_record.bucket_boundaries,
                                'bucket_percentiles': feature_record.bucket_percentiles,
                            })
                            
                            # Print comparison of explanations across buckets
                            print(f"\n    Multi-Bucket Explanation Summary for Neuron {neuron_idx_item}:")
                            for bucket_idx, explanation in bucket_explanations.items():
                                percentile = f"{bucket_idx*100//args.calibration_buckets}-{(bucket_idx+1)*100//args.calibration_buckets}%"
                                det_score = bucket_detection_scores.get(bucket_idx, 'N/A')
                                if isinstance(det_score, float):
                                    print(f"    Bucket {bucket_idx} ({percentile}): Detection={det_score:.3f}")
                                else:
                                    print(f"    Bucket {bucket_idx} ({percentile}): Detection={det_score}")
                                print(f"      {explanation}")
                            print()
                            
                            # Add to running statistics (using average across buckets)
                            if bucket_detection_scores:
                                detection_scores.append(np.mean(list(bucket_detection_scores.values())))
                        
                        elif args.stratified_calibration:
                            # Original single-explanation approach
                            # Generate explanation using the explanation_examples (from top bucket)
                            explanation: ExplainerResult = asyncio.run(explainer(feature_record))

                            # Score the explanation if requested
                            print(f"  Neuron {neuron_idx_item}: {explanation.explanation}")
                            # Score on each bucket separately
                            bucket_detection_scores = {}
                            # bucket_fuzz_scores = {}  # Disabled - focusing on detection only
                            
                            for bucket_idx in range(args.calibration_buckets):
                                if len(bucket_examples.get(bucket_idx, [])) == 0:
                                    print(f"    Bucket {bucket_idx}: No examples, skipping")
                                    continue
                                    
                                print(f"    Bucket {bucket_idx} (percentile {bucket_idx*100//args.calibration_buckets}-{(bucket_idx+1)*100//args.calibration_buckets}%):")
                                
                                # Create a temporary feature record with bucket examples
                                bucket_record = FeatureRecord(feature=feature)
                                bucket_record.explanation_examples = feature_record.explanation_examples
                                bucket_record.positive_examples = bucket_examples[bucket_idx]
                                bucket_record.negative_examples = feature_record.negative_examples
                                bucket_record.max_activation = feature_record.max_activation
                                
                                # Create explanation result for this bucket
                                bucket_explanation = ExplainerResult(
                                    record=bucket_record,
                                    explanation=explanation.explanation
                                )
                                
                                # Detection Score
                                detection_scorer = DetectionScorer(
                                    client=score_client,
                                    verbose=False,
                                    batch_size=5,
                                    use_structured_output=True,
                                    temperature=0.0,
                                )
                                detection_result = asyncio.run(detection_scorer(bucket_explanation))
                                bucket_detection_scores[bucket_idx] = detection_result.score
                                
                                # Fuzz Score - DISABLED (focusing on detection only)
                                # fuzz_scorer = FuzzingScorer(
                                #     client=score_client,
                                #     verbose=False,
                                #     batch_size=5,
                                #     threshold=0.3,
                                #     use_structured_output=True,
                                #     temperature=0.0,
                                # )
                                # fuzz_result = asyncio.run(fuzz_scorer(bucket_explanation))
                                # bucket_fuzz_scores[bucket_idx] = fuzz_result.score
                                
                                print(f"      Detection: {bucket_detection_scores[bucket_idx]:.3f}")
                            
                            # Compute overall scores as weighted average by bucket size
                            bucket_sizes = [len(bucket_examples.get(i, [])) for i in range(args.calibration_buckets)]
                            total_size = sum(bucket_sizes)
                            
                            if total_size > 0:
                                detection_score = sum(bucket_detection_scores.get(i, 0) * bucket_sizes[i] 
                                                    for i in range(args.calibration_buckets)) / total_size
                                # fuzz_score = sum(bucket_fuzz_scores.get(i, 0) * bucket_sizes[i] 
                                #                for i in range(args.calibration_buckets)) / total_size
                            else:
                                detection_score = 0.0
                                # fuzz_score = 0.0
                                
                            print(f"    ✓ Overall Detection score: {detection_score:.3f}")
                            # print(f"    ✓ Overall Fuzz score: {fuzz_score:.3f}")
                            
                            detection_scores.append(detection_score)
                            # fuzz_scores.append(fuzz_score)
                            
                            # Store scores with calibration info
                            all_explanation_scores[sae_pos].append({
                                'neuron_idx': neuron_idx_item,
                                'explanation': explanation.explanation,
                                'detection_score': detection_score,
                                # 'fuzz_score': fuzz_score,
                                'bucket_detection_scores': bucket_detection_scores,
                                # 'bucket_fuzz_scores': bucket_fuzz_scores,
                                'bucket_boundaries': feature_record.bucket_boundaries,
                                'bucket_percentiles': feature_record.bucket_percentiles,
                            })
                        else:
                            # Regular scoring (non-calibration)
                            # Generate explanation using the explanation_examples
                            explanation: ExplainerResult = asyncio.run(explainer(feature_record))
                            
                            # Score the explanation
                            print(f"  Neuron {neuron_idx_item}: {explanation.explanation}")
                            # 1. Detection Score
                            print(f"    Computing Detection score...")
                            detection_scorer = DetectionScorer(
                                client=score_client,
                                verbose=False,
                                batch_size=5,
                                use_structured_output=True,
                                temperature=0.0,
                            )
                            detection_result = asyncio.run(detection_scorer(explanation))
                            detection_score = detection_result.score
                            print(f"    ✓ Detection score: {detection_score:.3f}")
                            detection_scores.append(detection_score)
                            
                            # 2. Fuzz Score - DISABLED (focusing on detection only)
                            # print(f"    Computing Fuzz score...")
                            # fuzz_scorer = FuzzingScorer(
                            #     client=score_client,
                            #     verbose=False,
                            #     batch_size=5,
                            #     threshold=0.3,
                            #     use_structured_output=True,
                            #     temperature=0.0,
                            # )
                            # fuzz_result = asyncio.run(fuzz_scorer(explanation))
                            # # The score is now directly the accuracy
                            # fuzz_score = fuzz_result.score
                            # print(f"    ✓ Fuzz score: {fuzz_score:.3f}")
                            # fuzz_scores.append(fuzz_score)
                            
                            # Store scores
                            all_explanation_scores[sae_pos].append({
                                'neuron_idx': neuron_idx_item,
                                'explanation': explanation.explanation,
                                'detection_score': detection_score,
                                # 'fuzz_score': fuzz_score,
                            })
                        
                    except Exception as e:
                        print(f"  ✗ Error processing neuron {neuron_idx_item}: {e}")
                        print(f"    Skipping this neuron and continuing...")
                        continue
                
                # Print final statistics for this SAE position
                print(f"\n  Final statistics for {sae_pos}:")
                if detection_scores:
                    print(f"    Detection: mean={np.mean(detection_scores):.3f}, std={np.std(detection_scores):.3f}, n={len(detection_scores)}")
                # if fuzz_scores:
                #     print(f"    Fuzz:      mean={np.mean(fuzz_scores):.3f}, std={np.std(fuzz_scores):.3f}, n={len(fuzz_scores)}")
            
            # Save explanations to Wandb
            try:
                print(f"Saving explanations to Wandb for run {run_id}")
                save_explanations_to_wandb(explanations=all_explanation_scores, output_path=args.output_path)
                print(f"Successfully uploaded explanations to Wandb for run {run_id}")
                
            except Exception as e:
                print(f"Warning: Failed to upload explanations to Wandb: {e}")
    
        # Display explanation summary for this run if explanations were generated or loaded
        if args.generate_explanations and all_explanation_scores:
            print("\n" + "=" * 60)
            print(f"EXPLANATION SCORES SUMMARY - Run {run_id}")
            print("=" * 60)
            
            # Save scores to JSON for this run
            scores_path = Path(args.output_path) / f"explanation_scores_{run_id}.json"
            with open(scores_path, 'w') as f:
                json.dump(all_explanation_scores, f, indent=2)
            print(f"\nScores saved to: {scores_path}")
            
            # Create calibration plots if using stratified calibration
            if args.stratified_calibration:
                create_calibration_plots(all_explanation_scores, args.output_path, run_id)
            
            # Display summary statistics by layer
            print("\nScore Statistics by Layer:")
            all_scores = []  # For overall top explanations
            
            for sae_pos, scores_list in all_explanation_scores.items():
                if not scores_list:
                    continue
                    
                # Get scores for this layer (handle both regular and multi-bucket entries)
                layer_detection_scores = []
                for s in scores_list:
                    if 'detection_score' in s and s['detection_score'] is not None:
                        # Regular entry
                        layer_detection_scores.append(s['detection_score'])
                    elif 'bucket_detection_scores' in s:
                        # Multi-bucket entry - compute average across buckets
                        bucket_scores = [score for score in s['bucket_detection_scores'].values()]
                        if bucket_scores:
                            layer_detection_scores.append(np.mean(bucket_scores))
                # layer_fuzz_scores = [s['fuzz_score'] for s in scores_list if s.get('fuzz_score') is not None]
                
                print(f"\n  {sae_pos}:")
                print(f"    Total neurons: {len(scores_list)}")
                if layer_detection_scores:
                    print(f"    Detection: mean={np.mean(layer_detection_scores):.3f}, std={np.std(layer_detection_scores):.3f}, n={len(layer_detection_scores)}")
                # if layer_fuzz_scores:
                #     print(f"    Fuzz:      mean={np.mean(layer_fuzz_scores):.3f}, std={np.std(layer_fuzz_scores):.3f}, n={len(layer_fuzz_scores)}")
                
                # Add calibration metrics if available
                if args.stratified_calibration:
                    calibration_entries = [s for s in scores_list if 'bucket_detection_scores' in s]
                    if calibration_entries:
                        # Compute calibration correlation
                        bucket_percentiles = calibration_entries[0]['bucket_percentiles']
                        bucket_midpoints = [(bucket_percentiles[i] + bucket_percentiles[i+1]) / 2 
                                          for i in range(len(bucket_percentiles)-1)]
                        
                        # Aggregate scores by bucket
                        det_by_bucket = {i: [] for i in range(len(bucket_midpoints))}
                        # fuzz_by_bucket = {i: [] for i in range(len(bucket_midpoints))}  # Disabled
                        
                        for entry in calibration_entries:
                            for bucket_idx, score in entry['bucket_detection_scores'].items():
                                # Convert bucket_idx to int (JSON saves integer keys as strings)
                                bucket_idx_int = int(bucket_idx) if isinstance(bucket_idx, str) else bucket_idx
                                det_by_bucket[bucket_idx_int].append(score)
                            # for bucket_idx, score in entry.get('bucket_fuzz_scores', {}).items():
                            #     fuzz_by_bucket[bucket_idx].append(score)
                        
                        # Compute mean score per bucket
                        det_means = [np.mean(det_by_bucket[i]) if det_by_bucket[i] else np.nan 
                                    for i in range(len(bucket_midpoints))]
                        fuzz_means = [np.mean(fuzz_by_bucket[i]) if fuzz_by_bucket[i] else np.nan 
                                     for i in range(len(bucket_midpoints))]
                        
                        # Remove NaN values for correlation computation
                        valid_det = [(bucket_midpoints[i], det_means[i]) 
                                    for i in range(len(bucket_midpoints)) if not np.isnan(det_means[i])]
                        valid_fuzz = [(bucket_midpoints[i], fuzz_means[i]) 
                                     for i in range(len(bucket_midpoints)) if not np.isnan(fuzz_means[i])]
                        
                        if len(valid_det) > 1:
                            det_corr = np.corrcoef([x[0] for x in valid_det], [x[1] for x in valid_det])[0, 1]
                            print(f"    Detection calibration correlation: {det_corr:.3f}")
                        
                        # if len(valid_fuzz) > 1:
                        #     fuzz_corr = np.corrcoef([x[0] for x in valid_fuzz], [x[1] for x in valid_fuzz])[0, 1]
                        #     print(f"    Fuzz calibration correlation: {fuzz_corr:.3f}")
                
                # Add to overall list with sae_pos for top explanations display
                for score_dict in scores_list:
                    score_dict_with_pos = score_dict.copy()
                    score_dict_with_pos['sae_pos'] = sae_pos
                    all_scores.append(score_dict_with_pos)

            # Display overall statistics (handle both regular and multi-bucket entries)
            overall_detection_scores = []
            for s in all_scores:
                if 'detection_score' in s and s['detection_score'] is not None:
                    overall_detection_scores.append(s['detection_score'])
                elif 'bucket_detection_scores' in s:
                    bucket_scores = [score for score in s['bucket_detection_scores'].values()]
                    if bucket_scores:
                        overall_detection_scores.append(np.mean(bucket_scores))
            # overall_fuzz_scores = [s['fuzz_score'] for s in all_scores if s.get('fuzz_score') is not None]
            
            print(f"\n  Overall (All Layers):")
            print(f"    Total neurons: {len(all_scores)}")
            if overall_detection_scores:
                print(f"    Detection: mean={np.mean(overall_detection_scores):.3f}, std={np.std(overall_detection_scores):.3f}, n={len(overall_detection_scores)}")
            # if overall_fuzz_scores:
            #     print(f"    Fuzz:      mean={np.mean(overall_fuzz_scores):.3f}, std={np.std(overall_fuzz_scores):.3f}, n={len(overall_fuzz_scores)}")

            # Display top scoring explanations across all layers
            print("\nTop 5 explanations by Detection score (across all layers):")
            # Filter and compute scores for sorting
            scores_for_sorting = []
            for s in all_scores:
                if 'detection_score' in s and s['detection_score'] is not None:
                    score_val = s['detection_score']
                    scores_for_sorting.append((s, score_val))
                elif 'bucket_detection_scores' in s:
                    bucket_scores = [score for score in s['bucket_detection_scores'].values()]
                    if bucket_scores:
                        score_val = np.mean(bucket_scores)
                        scores_for_sorting.append((s, score_val))
            
            sorted_by_detection = sorted(scores_for_sorting, key=lambda x: x[1], reverse=True)[:5]
            for i, (score_entry, score_val) in enumerate(sorted_by_detection, 1):
                print(f"  {i}. Neuron {score_entry['neuron_idx']} ({score_entry['sae_pos']}): {score_val:.3f}")
                if 'explanation' in score_entry:
                    print(f"     {score_entry['explanation'][:100]}...")
                elif 'bucket_explanations' in score_entry:
                    # For multi-bucket, show the highest bucket's explanation
                    bucket_scores_dict = score_entry.get('bucket_detection_scores', {})
                    if bucket_scores_dict:
                        best_bucket = max(bucket_scores_dict.items(), key=lambda x: x[1])[0]
                        print(f"     [Multi-bucket, best bucket {best_bucket}]: {score_entry['bucket_explanations'][best_bucket][:100]}...")
    
        # Finish the current wandb run before moving to the next one
        wandb.finish()

    wandb.teardown()
    # Create pareto plots after processing all runs
    create_pareto_plots(all_run_metrics)


def main():
    """Main function with argparse configuration."""
    parser = argparse.ArgumentParser(description="Run SAE evaluation with configurable parameters")
    
    # Window and processing parameters
    parser.add_argument("--window_size", type=int, default=64, 
                       help="Size of token windows for processing (default: 64)")

    # Explanation parameters
    parser.add_argument("--num_neurons", type=int, default=300,
                       help="Number of top neurons to process per layer (default: 300)")
    parser.add_argument("--num_positive_examples", type=int, default=100,
                       help="Number of positive examples (where neuron is active) for scoring (default: 100)")
    parser.add_argument("--num_negative_examples", type=int, default=100,
                       help="Number of negative examples (where neuron is inactive) for scoring (default: 100)")
    parser.add_argument("--max_activated_features_per_neuron", type=int, default=10000,
                       help="Maximum number of activated features to use for explanation per neuron (default: 1000)")
    parser.add_argument("--num_features_to_explain", type=int, default=10,
                       help="Number of top activation examples to use for explanation (default: 10)")
    parser.add_argument("--n_eval_samples", type=int, default=5000,
                       help="Number of evaluation samples to process (default: 50000)")
    parser.add_argument("--stratified_quantiles", type=int, default=20,
                       help="Number of quantiles for stratified sampling of activation examples and neurons (default: None)")
    
    # Stratified calibration parameters
    parser.add_argument("--stratified_calibration", action="store_true", default=False,
                       help="Enable stratified calibration evaluation (default: False)")
    parser.add_argument("--calibration_buckets", type=int, default=5,
                       help="Number of percentile buckets for calibration analysis (default: 5)")
    parser.add_argument("--examples_per_bucket", type=int, default=20,
                       help="Number of examples per bucket for calibration evaluation (default: 20)")
    parser.add_argument("--multi_bucket_explanations", action="store_true", default=False,
                       help="Generate separate explanations from each bucket instead of just top bucket (default: False)")
    
    # Model parameters
    parser.add_argument("--explanation_model", type=str, default="gpt-4o",
                       help="Model to use for generating explanations (default: gpt-4o)")
    parser.add_argument("--scoring_model", type=str, default="llama-3.3-70b",
                       help="Model to use for scoring explanations (default: llama-3.3-70b)")
    
    # Wandb and storage parameters
    parser.add_argument("--wandb_project", type=str, default="raymondl/tinystories-1m",
                       help="Wandb project in format 'entity/project' (default: raymondl/tinystories-1m)")
    parser.add_argument("--filter_runs_by_name", type=str, default=None,
                       help="Filter runs by a specific string in their name (default: None)")
    parser.add_argument("--output_path", type=str, default="./artifacts",
                       help="Path for storing temporary files and artifacts (default: ./artifacts)")
    
    # Execution flags
    parser.add_argument("--save_activation_data", action="store_true", default=False,
                       help="Save activation data (default: False)")
    
    parser.add_argument("--skip_upload", action="store_true", default=False,
                       help="Skip uploading to Wandb, only save locally (default: False)")
    
    parser.add_argument("--no_chunk_upload", action="store_true", default=False,
                       help="Disable chunked upload (upload all files at once) (default: False)")
    
    parser.add_argument("--generate_explanations", action="store_true", default=False,
                       help="Generate neuron explanations (default: False)")

    parser.add_argument("--force_recompute", action="store_true", default=False,
                       help="Force recomputation of metrics even if existing ones are found (default: False)")

    parser.add_argument("--force_recompute_explanations", action="store_true", default=False,
                       help="Force recomputation of explanations only, keeping existing metrics and activations (default: False)")

    # For debugging
    parser.add_argument("--override_n_train_samples", type=int, default=None,
                       help="Override n_train_samples to avoid slow data skipping (default: None - use config value)")
    
    parser.add_argument("--sae_position", type=str, default=None,
                       help="SAE position to evaluate (default: None - use all)")

    args = parser.parse_args()
    
    # Run the evaluation with parsed arguments
    run_evaluation(args)


if __name__ == "__main__":
    main()
