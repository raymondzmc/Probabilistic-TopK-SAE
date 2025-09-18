
#!/bin/bash
conda activate sae
export CUDA_VISIBLE_DEVICES=5

python run_experiments.py \
--base_config configs/gpt2/gpt2-gated.yaml \
--sweep_config configs/gpt2/sweep/gated_sweep.yaml \
--output_dir experiment_outputs/gated_sweep

python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name gated