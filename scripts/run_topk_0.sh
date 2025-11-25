
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=1

python run_experiments.py \
--base_config configs/gpt2/gpt2-topk.yaml \
--sweep_config configs/gpt2/sweep/topk_sweep_0.yaml

python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name topk_k_4