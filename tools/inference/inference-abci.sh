#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:00:30:00
#$ -j y
#$ -o outputs/inference/mixtral-8x7b/
#$ -cwd

source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

set -e

# swich virtual env
source .env/bin/activate

python tools/inference/inference-mixtral.py \
  --model-path /bb/llm/gaf51275/llama/huggingface-checkpoint/Mixtral-8x7B-v0.1 \
  --tokenizer-path /bb/llm/gaf51275/llama/huggingface-checkpoint/Mixtral-8x7B-v0.1 \
  --prompt "Tokyo is the capital of Japan."

python tools/inference/inference-mixtral.py \
  --model-path /bb/llm/gaf51275/llama/huggingface-checkpoint/Mixtral-8x7B-v0.1 \
  --tokenizer-path /bb/llm/gaf51275/llama/huggingface-checkpoint/Mixtral-8x7B-v0.1 \
  --prompt "東京工業大学のキャンパスは"
