
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=5

python run_experiments.py \
--base_config configs/gpt2/gpt2-hc_topk.yaml \
--sweep_config configs/gpt2/sweep/hc_topk_sweep_5.yaml \
--output_dir experiment_outputs/hc_topk_sweep_5

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name k_32