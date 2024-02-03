#!/bin/bash
#YBATCH -r dgx-a100_8
#SBATCH --job-name=check
#SBATCH --time=6:00:00
#SBATCH --output outputs/check/%j.out
#SBATCH --error errors/check/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.8
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

# python virtualenv
source .env/bin/activate

python tools/checkpoint-convert/scripts/ylab/check.py \
  --base-hf-model-path /home/kazuki/hf_checkpoints/Mixtral-8x7B-v0.1 \
  --converted-hf-model-path /home/kazuki/converted_checkpoints/Mistral-8x7b/iter_0000005
