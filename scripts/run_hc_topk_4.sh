
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=4

python run_experiments.py \
--base_config configs/gpt2/gpt2-hc_topk.yaml \
--sweep_config configs/gpt2/sweep/hc_topk_sweep_4.yaml \
--output_dir experiment_outputs/hc_topk_sweep_4

python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name probabilistic_topk_k_8_initial_beta_1.0

python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name probabilistic_topk_k_16_initial_beta_1.0

python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name probabilistic_topk_k_32_initial_beta_1.0
