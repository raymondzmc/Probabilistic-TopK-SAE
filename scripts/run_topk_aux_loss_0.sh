
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=0

python run_experiments.py \
--base_config configs/gpt2/gpt2-topk.yaml \
--sweep_config configs/gpt2/sweep/topk_aux_loss_sweep_0.yaml \
--output_dir experiment_outputs/topk_aux_loss_sweep

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name topk_k_8