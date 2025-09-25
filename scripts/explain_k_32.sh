
# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name relu_sparsity_coeff_30 \
# --save_activation_data \
# --sae_position blocks.8.hook_resid_pre \
# --skip_upload

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name gated_sparsity_coeff_0.09\
# --save_activation_data \
# --sae_position blocks.8.hook_resid_pre \
# --skip_upload

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name probabilistic_k_8 \
# --save_activation_data \
# --sae_position blocks.8.hook_resid_pre \
# --skip_upload

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name topk_k_8_interpret \
# --save_activation_data \
# --sae_position blocks.8.hook_resid_pre \
# --skip_upload




# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name probabilistic_k_16 \
# --save_activation_data \
# --generate_explanations \
# --force_recompute_explanations \
# --sae_position blocks.8.hook_resid_pre \
# --num_neurons 150 \
# --skip_upload


python evaluation.py \
--wandb_project raymondl/gpt2-small \
--filter_runs_by_name topk_k_16_interpret \
--save_activation_data \
--generate_explanations \
--force_recompute_explanations \
--sae_position blocks.8.hook_resid_pre \
--num_neurons 150 \
--skip_upload

# python evaluation.py \
# --wandb_project raymondl/gpt2-small \
# --filter_runs_by_name gated_sparsity_coeff_0.06 \
# --save_activation_data \
# --generate_explanations \
# --sae_position blocks.8.hook_resid_pre \
# --num_neurons 150 \
# --skip_upload