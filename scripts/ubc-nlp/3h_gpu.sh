#!/bin/bash

#SBATCH --job-name=3h_gpu
#SBATCH --account=cli2711
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=124gb
#SBATCH --gres=gpu:gpu:1
#SBATCH --partition=nlpgpo  

echo 'Hello, world!'

echo "SLURM_TMPDIR" $SLURM_TMPDIR
echo "SLURM_JOB_ID" $SLURM_JOB_ID
echo "SLURM_NODELIST" $SLURM_NODELIST

sleep 7d