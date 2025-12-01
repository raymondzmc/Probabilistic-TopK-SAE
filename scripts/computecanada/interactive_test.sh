salloc \
  --account=def-carenini \
  --time=04:00:00 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:h100:1 \
  --constraint=h100
