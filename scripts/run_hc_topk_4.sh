
#!/bin/bash
conda activate sae

export CUDA_VISIBLE_DEVICES=5

python run_experiments.py \
--base_config configs/tinystories/tinystories-hc_topk.yaml \
--sweep_config configs/tinystories/sweep/hc_topk_sweep4.yaml \
--output_dir experiment_outputs/hc_topk_sweep4

python evaluation.py \
--wandb_project raymondl/tinystories-1m-test \
--filter_runs_by_name z_scale_0.9

python evaluation.py \
--wandb_project raymondl/tinystories-1m-test \
--filter_runs_by_name z_scale_1.0