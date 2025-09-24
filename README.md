module load gcc arrow/21.0.0

pip install torch
pip install pyyaml
pip install wandb
pip install jaxtyping
pip install huggingface_hub
pip install einops

pip install numpy
pip install datasets
pip install transformers



uv pip install torch
uv pip install pyyaml
uv pip install wandb
uv pip install jaxtyping
uv pip install huggingface_hub
uv pip install einops
uv pip install numpy
uv pip install datasets
uv pip install transformers
uv pip install transformer_lens
uv pip install dotenv


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



uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/hc_topk_sweep_0.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_0

uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/hc_topk_sweep_1.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_1

CUDA_VISIBLE_DEVICES=0 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/hc_topk_sweep_2.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_2


CUDA_VISIBLE_DEVICES=1 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-topk.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/topk_sweep_0.yaml \
--output_dir experiment_outputs/qwen3-0.6b/topk_sweep_0

CUDA_VISIBLE_DEVICES=2 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-topk.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/topk_sweep_1.yaml \
--output_dir experiment_outputs/qwen3-0.6b/topk_sweep_1

CUDA_VISIBLE_DEVICES=3 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-topk.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/topk_sweep_2.yaml \
--output_dir experiment_outputs/qwen3-0.6b/topk_sweep_2



export XDG_CACHE_HOME=/home/ubuntu/anji/.cache
export HF_HOME=/home/ubuntu/anji/.cache/huggingface




CUDA_VISIBLE_DEVICES=0 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-topk.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/topk_sweep_3.yaml \
--output_dir experiment_outputs/qwen3-0.6b/topk_sweep_3

CUDA_VISIBLE_DEVICES=1 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/hc_topk_sweep_3.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_3




CUDA_VISIBLE_DEVICES=0 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/hc_topk_sweep_2.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_2

CUDA_VISIBLE_DEVICES=1 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-hc_topk.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/hc_topk_sweep_2-1.yaml \
--output_dir experiment_outputs/qwen3-0.6b/hc_topk_sweep_2-1






source .env/bin/activate

export XDG_CACHE_HOME=/home/ubuntu/Anji/.cache
export HF_HOME=/home/ubuntu/Anji/.cache/huggingface


CUDA_VISIBLE_DEVICES=0 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-gated.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/gated_sweep_0.yaml \
--output_dir experiment_outputs/qwen3-0.6b/gated_sweep_0

CUDA_VISIBLE_DEVICES=1 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-gated.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/gated_sweep_1.yaml \
--output_dir experiment_outputs/qwen3-0.6b/gated_sweep_1

CUDA_VISIBLE_DEVICES=2 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-gated.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/gated_sweep_2.yaml \
--output_dir experiment_outputs/qwen3-0.6b/gated_sweep_2

CUDA_VISIBLE_DEVICES=3 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-gated.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/gated_sweep_3.yaml \
--output_dir experiment_outputs/qwen3-0.6b/gated_sweep_3


CUDA_VISIBLE_DEVICES=4 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-relu.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/relu_sweep_0.yaml \
--output_dir experiment_outputs/qwen3-0.6b/relu_sweep_0

CUDA_VISIBLE_DEVICES=5 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-relu.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/relu_sweep_1.yaml \
--output_dir experiment_outputs/qwen3-0.6b/relu_sweep_1

CUDA_VISIBLE_DEVICES=6 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-relu.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/relu_sweep_2.yaml \
--output_dir experiment_outputs/qwen3-0.6b/relu_sweep_2

CUDA_VISIBLE_DEVICES=7 uv run run_experiments.py --base_config configs/qwen3-0.6b-0923/qwen3-0.6b-relu.yaml \
--sweep_config configs/qwen3-0.6b-0923/sweep/relu_sweep_3.yaml \
--output_dir experiment_outputs/qwen3-0.6b/relu_sweep_3




python evaluation.py --wandb_project lisa27chuyuan-university-of-british-columbia/Qwen3-0.6B-0923 \
--filter_runs_by_name topk_k_64

python evaluation.py --wandb_project lisa27chuyuan-university-of-british-columbia/Qwen3-0.6B-0923 \
--filter_runs_by_name topk_k_32

python evaluation.py --wandb_project lisa27chuyuan-university-of-british-columbia/Qwen3-0.6B-0923 \
--filter_runs_by_name topk_k_16