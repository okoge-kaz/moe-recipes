#!/bin/bash
#$ -l rt_AF=4
#$ -l h_rt=10:00:00
#$ -j y
#$ -o outputs/mixtral-8x7b/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module use /groups/gag51395/modules/modulefiles

module load cuda/12.1/12.1.1
module load cudnn/cuda-12.1/9.0.0
module load nccl/2.17/2.17.1-1
module load hpcx/2.12
module load gcc/11.4.0

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done <"$SGE_JOB_HOSTLIST" >"$HOSTFILE_NAME"

# training config
# Mixtral-8x7B https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
SEQ_LENGTH=4096
SLIDING_WINDOW_SIZE=4096
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
GRADIENTS_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE / NUM_GPUS))

if [ $GRADIENTS_ACCUMULATION_STEPS -lt 1 ]; then
  echo "Global batch size is too small for the number of GPUs"
  exit 1
fi

TRAIN_STEPS=25000

# optimizer config
LR=2e-5
MIN_LR=2e-6
LR_WARMUP_STEPS=1000
LR_DECAY_STEPS=25000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

ADAMW_BETA1=0.9
ADAMW_BETA2=0.95
ADAMW_EPS=1e-5

# checkpoint & tokenizer
TOKENIZER_MODEL=/bb/llm/gaf51275/llama/huggingface-checkpoint/Mixtral-8x7B-Instruct-v0.1/tokenizer.model
CHECKPOINT_DIR=/bb/llm/gaf51275/llama/huggingface-checkpoint/Mixtral-8x7B-Instruct-v0.1
CHECKPOINT_SAVE_DIR="/bb/llm/gaf51275/checkpoints/Mixtral-8x7B-Instruct-v0.1/LR_${LR}-MIN-LR_${MIN_LR}_WARMUP_${LR_WARMUP_STEPS}_WD_${WEIGHT_DECAY}_GC_${GRAD_CLIP}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH=""

# ja okazaki lab cc
DATA_PATH="${DATA_PATH} 7417203315 /groups/gag51395/datasets/binarized/mistral_original/Llama2Tokenizer/okazaki_lab_cc_03_1500_split_0_text_document"
DATA_PATH="${DATA_PATH} 7340333648 /groups/gag51395/datasets/binarized/mistral_original/Llama2Tokenizer/okazaki_lab_cc_03_1500_split_1_text_document"
DATA_PATH="${DATA_PATH} 8766459643 /groups/gag51395/datasets/binarized/mistral_original/Llama2Tokenizer/okazaki_lab_cc_03_1500_split_2_text_document"
DATA_PATH="${DATA_PATH} 11561683685 /groups/gag51395/datasets/binarized/mistral_original/Llama2Tokenizer/okazaki_lab_cc_03_1500_split_3_text_document"
DATA_PATH="${DATA_PATH} 27050402839 /groups/gag51395/datasets/binarized/mistral_original/Llama2Tokenizer/okazaki_lab_cc_03_1500_split_4_text_document"

# ja wikipedia
DATA_PATH="${DATA_PATH} 2245464469 /groups/gag51395/datasets/binarized/mistral_original/Llama2Tokenizer/ja_wiki_merged_text_document"

# en arxiv
DATA_PATH="${DATA_PATH} 14315663909 /groups/gag51395/datasets/binarized/mistral_original/Llama2Tokenizer/arxiv_text_document"

# en refinedweb
DATA_PATH="${DATA_PATH} 11302788492 /groups/gag51395/datasets/binarized/mistral_original/Llama2Tokenizer/falcon_text_document"

# algebraic stack
DATA_PATH="${DATA_PATH} 5000000000 /groups/gag51395/datasets/binarized/mistral_original/Llama2Tokenizer/algebraic_stack_text_document"

# The Vault
DATA_PATH="${DATA_PATH} 5000000000 /groups/gag51395/datasets/binarized/mistral_original/Llama2Tokenizer/The_Vault_text_text_document"

# deepspeed config
DEEPSPEED_CONFIG="configs/mixtral-8x7b.json"

BF16_ENABLED=true
DEEPSPEED_ZERO_STAGE=3

OVERLAP_COMMUNICATION=true
CONTINOUS_GRADIENTS=true

DEEPSPEED_SUB_GROUP_SIZE=1e9
DEEPSPEED_REDUCE_BUCKET_SIZE="auto"
DEEPSPEED_STAGE3_PREFETCH_BUCKET_SIZE=0
DEEPSPEED_STAGE3_PARAM_PERSISTENCE_THRESHOLD="auto"

DEEPSPEED_STAGE3_MAX_LIVE_PARAMETERS=1e9
DEEPSPEED_STAGE3_MAX_REUSE_DISTANCE=1e9

WALL_CLOCK_BREAKDOWN=false

DEEPSPEED_CONGIG_CONTENT=$(
  cat <<EOF
{
  "bf16": {
    "enabled": $BF16_ENABLED
  },
  "zero_optimization": {
    "stage": $DEEPSPEED_ZERO_STAGE,
    "overlap_comm": $OVERLAP_COMMUNICATION,
    "contiguous_gradients": $CONTINOUS_GRADIENTS,
    "sub_group_size": $DEEPSPEED_SUB_GROUP_SIZE,
    "reduce_bucket_size": "$DEEPSPEED_REDUCE_BUCKET_SIZE",
    "stage3_prefetch_bucket_size": $DEEPSPEED_STAGE3_PREFETCH_BUCKET_SIZE,
    "stage3_param_persistence_threshold": "$DEEPSPEED_STAGE3_PARAM_PERSISTENCE_THRESHOLD",
    "stage3_max_live_parameters": $DEEPSPEED_STAGE3_MAX_LIVE_PARAMETERS,
    "stage3_max_reuse_distance": $DEEPSPEED_STAGE3_MAX_REUSE_DISTANCE
  },
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "gradient_accumulation_steps": $GRADIENTS_ACCUMULATION_STEPS,
  "gradient_clipping": $GRAD_CLIP,
  "wall_clock_breakdown": $WALL_CLOCK_BREAKDOWN
}
EOF
)

mkdir -p ./configs

# write deepspeed config file
echo "$DEEPSPEED_CONGIG_CONTENT" >$DEEPSPEED_CONFIG

# job name
JOB_NAME="Mixtral-8x7B-Instruct-v0.1-NVE-ABCI-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# CUTLASS build
# git clone git@github.com:NVIDIA/cutlass.git
# cd cutlass && mkdir -p build && cd build
# module load cuda
# module load cmake/3.29.0
# cmake .. -DCUTLASS_NVCC_ARCHS=80 (for Ampere)

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -x CUTLASS_PATH=/bb/llm/gaf51275/2024/cutlass \
  python examples/finetuning.py \
  --seq-length ${SEQ_LENGTH} \
  --sliding-window-size ${SLIDING_WINDOW_SIZE} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --data-path ${DATA_PATH} \
  --split 949,50,1 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --lr-decay-iters ${LR_DECAY_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip-norm ${GRAD_CLIP} \
  --optimizer adam \
  --adam-beta1 $ADAMW_BETA1 \
  --adam-beta2 $ADAMW_BETA2 \
  --adam-eps $ADAMW_EPS \
  --save-interval 250 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --mixed-precision \
  --base-model ${CHECKPOINT_DIR} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --load ${CHECKPOINT_SAVE_DIR} \
  --use-zero \
  --zero-config ${DEEPSPEED_CONFIG} \
  --zero-stage ${DEEPSPEED_ZERO_STAGE} \
  --no-meta-device \
  --output-router-logits \
  --use-mpi \
  --continual-pretraining \
  --wandb-entity "okoge" \
  --wandb-project "Mixtral-8x7b" \
  --wandb-name "${JOB_NAME}"
