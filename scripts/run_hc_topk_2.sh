
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=2

python run_experiments.py \
--base_config configs/gpt2/gpt2-hc_topk.yaml \
--sweep_config configs/gpt2/sweep/hc_topk_sweep_2.yaml \
--output_dir experiment_outputs/hc_topk_sweep_2
# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name probabilistic_k_8 \
# --save_activation_data \
# --generate_explanations \
# --skip_upload 

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name topk_k_8_interpret \
# --save_activation_data \
# --generate_explanations \
# --skip_upload 

# # python evaluation.py \
# # --wandb_project raymondl/gpt2-small \
# # --filter_runs_by_name probabilistic_k_32 \
# # --save_activation_data \
# # --generate_explanations \
# # --skip_upload 


# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name probabilistic_k_32 \
# --save_activation_data \
# --generate_explanations \
# --skip_upload 


# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name topk_k_32_interpret \
# --save_activation_data \
# --generate_explanations \
# --skip_upload

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name gated_sparsity_coeff_0.03 \
# --save_activation_data \
# --generate_explanations \
# --skip_upload