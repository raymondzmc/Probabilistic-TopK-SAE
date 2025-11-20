
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=3

python run_experiments.py \
--base_config configs/gpt2/gpt2-batch_topk.yaml \
--sweep_config configs/gpt2/sweep/batch_topk_aux_loss_sweep_1.yaml \
--output_dir experiment_outputs/batch_topk_aux_loss_sweep

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name topk_k_8