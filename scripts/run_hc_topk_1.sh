
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=1

# python run_experiments.py \
# --base_config configs/gpt2/gpt2-hc_topk.yaml \
# --sweep_config configs/gpt2/sweep/hc_topk_sweep_1.yaml \
# --output_dir experiment_outputs/hc_topk_sweep_1

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name dgated_sparsity_coeff_0.09 \
# --save_activation_data \
# --generate_explanations \
# --force_recompute_explanations \
# --sae_position blocks.8.hook_resid_pre \
# --num_neurons 150 \
# --stratified_quantiles 20 \
# --skip_upload

