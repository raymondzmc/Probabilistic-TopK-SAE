#!/bin/bash
#SBATCH --account=def-carenini
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=124gb
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-user=lisa27chuyuan@gmail.com
#SBATCH --mail-type=ALL   

echo 'Hello, world!'

echo "SLURM_TMPDIR" $SLURM_TMPDIR
echo "SLURM_JOB_ID" $SLURM_JOB_ID
echo "SLURM_NODELIST" $SLURM_NODELIST

sleep 7d