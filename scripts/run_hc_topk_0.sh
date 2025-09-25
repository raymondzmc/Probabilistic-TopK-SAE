
#!/bin/bash
conda activate sae

# export CUDA_VISIBLE_DEVICES=0

# python run_experiments.py \
# --base_config configs/gpt2/gpt2-hc_topk.yaml \
# --sweep_config configs/gpt2/sweep/hc_topk_sweep_0.yaml \
# --output_dir experiment_outputs/hc_topk_sweep_0

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name topk_k_24_interpret

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name probabilistic_k_24

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name hard_concrete_topk_k_32_magnitude_scale_1e-5

python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name relu_sparsity_coeff_30 \
--save_activation_data \
--generate_explanations \
--force_recompute_explanations \
--sae_position blocks.8.hook_resid_pre \
--num_neurons 50 \
--skip_upload \
--stratified_quantiles 20

python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name gated_sparsity_coeff_0.09 \
--save_activation_data \
--generate_explanations \
--force_recompute_explanations \
--sae_position blocks.8.hook_resid_pre \
--num_neurons 50 \
--skip_upload \
--stratified_quantiles 20


python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name topk_k_8_interpret \
--save_activation_data \
--generate_explanations \
--force_recompute_explanations \
--sae_position blocks.8.hook_resid_pre \
--num_neurons 50 \
--skip_upload \
--stratified_quantiles 20

python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name probabilistic_k_8 \
--save_activation_data \
--generate_explanations \
--force_recompute_explanations \
--sae_position blocks.8.hook_resid_pre \
--num_neurons 50 \
--skip_upload \
--stratified_quantiles 20
