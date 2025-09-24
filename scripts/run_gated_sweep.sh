
#!/bin/bash
conda activate sae
export CUDA_VISIBLE_DEVICES=3

python run_experiments.py \
--base_config configs/gpt2/gpt2-gated.yaml \
--sweep_config configs/gpt2/sweep/gated_sweep_6e-2.yaml \
--output_dir experiment_outputs/gated_sweep_6e-2
python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name gated_sparsity_coeff_0.06


python run_experiments.py \
--base_config configs/gpt2/gpt2-gated.yaml \
--sweep_config configs/gpt2/sweep/gated_sweep_9e-2.yaml \
--output_dir experiment_outputs/gated_sweep_9e-2
python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name gated_sparsity_coeff_0.09


python run_experiments.py \
--base_config configs/gpt2/gpt2-gated.yaml \
--sweep_config configs/gpt2/sweep/gated_sweep_8e-2.yaml \
--output_dir experiment_outputs/gated_sweep_8e-2
python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name gated_sparsity_coeff_0.08


python run_experiments.py \
--base_config configs/gpt2/gpt2-gated.yaml \
--sweep_config configs/gpt2/sweep/gated_sweep_9e-2.yaml \
--output_dir experiment_outputs/gated_sweep_9e-2
python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name gated_sparsity_coeff_0.09