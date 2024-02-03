#!/bin/bash
#YBATCH -r dgx-a100_8
#SBATCH --job-name=inference
#SBATCH --time=24:00:00
#SBATCH --output outputs/inference/%j.out
#SBATCH --error errors/inference/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.8
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

set -e

# swich virtual env
source .env/bin/activate

python tools/inference/inference.py \
  --model-path /home/kazuki/converted_checkpoints/Mistral-8x7b/iter_0000020 \
  --tokenizer-path /home/kazuki/hf_checkpoints/Mixtral-8x7B-v0.1 \
  --prompt "Tokyo is the capital of Japan."

python tools/inference/inference.py \
  --model-path /home/kazuki/converted_checkpoints/Mistral-8x7b/iter_0000020 \
  --tokenizer-path /home/kazuki/hf_checkpoints/Mixtral-8x7B-v0.1 \
  --prompt "東京工業大学のキャンパスは"
