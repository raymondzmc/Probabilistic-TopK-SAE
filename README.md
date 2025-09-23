module load gcc arrow/21.0.0

pip install torch
pip install pyyaml
pip install wandb
pip install jaxtyping
pip install huggingface_hub
pip install einops

XDG_CACHE_HOME=/home/cli2711/scratch/.cache HF_HOME=/home/cli2711/scratch/.cache/huggingface

export XDG_CACHE_HOME=/home/cli2711/scratch/.cache
export HF_HOME=/home/cli2711/scratch/.cache/huggingface


export XDG_CACHE_HOME=/scratch/.cache
export HF_HOME=/scratch/.cache/huggingface



python run_experiments.py --base_config configs/qwen3-0.6b/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b/sweep/hc_topk_sweep_0.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_0

python run_experiments.py --base_config configs/qwen3-0.6b/qwen3-0.6b-relu.yaml \
--sweep_config configs/qwen3-0.6b/sweep/relu_sweep.yaml \
--output_dir experiment_outputs/qwen3-0.6b/relu_sweep

python run_experiments.py --base_config configs/qwen3-0.6b/qwen3-0.6b-gated.yaml \
--sweep_config configs/qwen3-0.6b/sweep/gated_sweep.yaml \
--output_dir experiment_outputs/qwen3-0.6b/gated_sweep

python run_experiments.py --base_config configs/qwen3-0.6b/qwen3-0.6b-topk.yaml \
--sweep_config configs/qwen3-0.6b/sweep/topk_sweep_0.yaml \
--output_dir experiment_outputs/qwen3-0.6b/topk_sweep_0

python run_experiments.py --base_config configs/qwen3-0.6b/qwen3-0.6b-topk.yaml \
--sweep_config configs/qwen3-0.6b/sweep/topk_sweep_1.yaml \
--output_dir experiment_outputs/qwen3-0.6b/topk_sweep_1



python run_experiments.py --base_config configs/qwen3-0.6b/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b/sweep/hc_topk_sweep_1.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_1

python run_experiments.py --base_config configs/qwen3-0.6b/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b/sweep/hc_topk_sweep_2.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_2

python run_experiments.py --base_config configs/qwen3-0.6b/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b/sweep/hc_topk_sweep_3.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_3

python run_experiments.py --base_config configs/qwen3-0.6b/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b/sweep/hc_topk_sweep_4.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_4

python run_experiments.py --base_config configs/qwen3-0.6b/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b/sweep/hc_topk_sweep_5.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_5