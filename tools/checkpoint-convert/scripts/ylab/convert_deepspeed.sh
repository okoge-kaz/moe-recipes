#!/bin/bash
#YBATCH -r dgx-a100_8
#SBATCH --job-name=convert
#SBATCH --time=24:00:00
#SBATCH --output outputs/checkpoint-convert/%j.out
#SBATCH --error errors/checkpoint-convertk/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.8
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

set -e

# swich virtual env
source .env/bin/activate

ITERATION=2000
FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

CHECK_POINT_DIR=/home/kazuki/checkpoints/Mixtral-8x7b/${FORMATTED_ITERATION}

python tools/checkpoint-convert/zero_to_fp32.py \
  --checkpoint-dir $CHECK_POINT_DIR \
  --output-file $CHECK_POINT_DIR/model.pt \
  --debug
