
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=0

python run_experiments.py \
--base_config configs/tinystories/tinystories-hc_topk.yaml \
--sweep_config configs/tinystories/sweep/hc_topk_sweep0.yaml \
--output_dir experiment_outputs/hc_topk_sweep0

python evaluation.py \
--wandb_project raymondl/tinystories-1m \
--filter_runs_by_name final_beta_1