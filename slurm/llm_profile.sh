#!/bin/bash

#SBATCH --job-name=llm_profile
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --gres=gpu:a100:1
#SBATCH -A bii_dsc_community
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --error="slurm/llm_profile.err"
#SBATCH --output="slurm/llm_profile.output"

module load gcc/13.3.0 nccl/2.21.5-CUDA-12.4.1
conda activate gpt_fast
which python
which gcc
cd /scratch/fad3ew/gpt-fast/
python generate_proton.py --compile