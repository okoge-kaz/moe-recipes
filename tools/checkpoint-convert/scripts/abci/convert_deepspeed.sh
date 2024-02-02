#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -o outputs/convert/ckpt/
#$ -cwd
# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

set -e

# swich virtual env
source .env/bin/activate

ITERATION=20
FORMATTED_ITERATION=$(printf "iter_%07d" $ITERATION)

CHECK_POINT_DIR=/bb/llm/gaf51275/llama/checkpoints/Mixtral-8x7b/okazaki-cc-lr_1e-4-minlr_1e-5_warmup_1000_sliding_window_4096_test/${FORMATTED_ITERATION}

python tools/checkpoint-convert/zero_to_fp32.py \
  --checkpoint-dir $CHECK_POINT_DIR \
  --output-file $CHECK_POINT_DIR/fp32_model.bin \
  --debug
