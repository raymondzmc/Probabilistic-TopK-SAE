#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=1
python run_experiments.py \
--base_config configs/gpt2/gpt2-relu.yaml \
--sweep_config configs/gpt2/sweep/relu_sweep.yaml \
--output_dir experiment_outputs/relu_sweep

python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name relu_sparsity_coeff_35