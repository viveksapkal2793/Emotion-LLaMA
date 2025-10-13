#!/bin/bash
#SBATCH --job-name=eval_test_meld
#SBATCH --output=/scratch/data/bikash_rs/vivek/Emotion-LLaMA/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/vivek/Emotion-LLaMA/logs/%x_%j.err
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --qos=fatqos
#SBATCH -D /scratch/data/bikash_rs/vivek/Emotion-LLaMA

# Create logs directory
mkdir -p logs

# Load CUDA module (adjust version based on your system)
# module load cuda/11.8

# Activate virtual environment
source emotion-llama-env/bin/activate

# Set CUDA 12.2 environment variables
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# Set environment variables
export HF_HOME=/scratch/data/bikash_rs/vivek/huggingface_cache
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

python eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset meld_caption
# python -m bitsandbytes