#!/bin/bash
#SBATCH -J tom_sft
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH -p HGX,DGX
#SBATCH --mem=64G
#SBATCH --qos=lv1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH -t 24:00:00

source ~/anaconda3/etc/profile.d/conda.sh
conda activate multiturn
cd /home/zhangchunhui/scratch/exps/multi-turn-verl

bash ./examples/sft/tom/run_qwen.sh