
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=0

python run_experiments.py \
--base_config configs/gpt2/gpt2-hc_topk.yaml \
--sweep_config configs/gpt2/sweep/hc_topk_sweep_0.yaml \
--output_dir experiment_outputs/hc_topk_sweep_0

python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name k_8

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name hard_concrete_topk_k_32_magnitude_scale_1e-4

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name hard_concrete_topk_k_32_magnitude_scale_1e-5