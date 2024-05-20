#!/bin/sh
#$ -cwd
#$ -l node_f=4
#$ -l h_rt=0:30:00
#$ -o outputs/mixtral-8x7b-VE/$JOB_ID
#$ -e outputs/mixtral-8x7b-VE/$JOB_ID
#$ -p -5

# Load modules
module use ~/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.1/2.18.3
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# swich virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

# training config
# Mixtral-8x7B https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
SEQ_LENGTH=4096
SLIDING_WINDOW_SIZE=4096
DATA_PARALLEL_SIZE=$NUM_GPUS

MICRO_BATCH_SIZE=2
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
TOKENIZER_MODEL=/gs/bs/tge-gc24sp01/hf-checkpoints/Mixtral-8x7B-Instruct-v0.1-VE/tokenizer.model
CHECKPOINT_DIR=/gs/bs/tge-gc24sp01/hf-checkpoints/Mixtral-8x7B-Instruct-v0.1-VE
CHECKPOINT_SAVE_DIR="/gs/bs/tge-gc24sp01/checkpoints/Mixtral-8x7B-Instruct-v0.1-VE/code-math-lr_${LR}-minlr_${MIN_LR}_warmup_${LR_WARMUP_STEPS}_seq_${SEQ_LENGTH}"

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config

DATA_PATH=""

# ja okazaki lab cc
DATA_PATH="${DATA_PATH} 7417203315 /gs/bs/tge-gc24sp01/datasets/mistral_original_Llama2Tokenizer/okazaki_lab_cc_03_1500_split_0_text_document"

# deepspeed config
DEEPSPEED_CONFIG="mixtral-8x7b.json"

BF16_ENABLED=true
DEEPSPEED_ZERO_STAGE=3

OVERLAP_COMMUNICATION=true
CONTINOUS_GRADIENTS=true

DEEPSPEED_SUB_GROUP_SIZE=1e12
DEEPSPEED_REDUCE_BUCKET_SIZE=1e9
DEEPSPEED_STAGE3_PREFETCH_BUCKET_SIZE=5e8
DEEPSPEED_STAGE3_PARAM_PERSISTENCE_THRESHOLD=1e6

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
    "reduce_bucket_size": $DEEPSPEED_REDUCE_BUCKET_SIZE,
    "stage3_prefetch_bucket_size": $DEEPSPEED_STAGE3_PREFETCH_BUCKET_SIZE,
    "stage3_param_persistence_threshold": $DEEPSPEED_STAGE3_PARAM_PERSISTENCE_THRESHOLD,
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

# write deepspeed config file
echo "$DEEPSPEED_CONGIG_CONTENT" >$DEEPSPEED_CONFIG

# job name
JOB_NAME="Mixtral-8x7B-Instruct-v0.1-NVE-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none \
  -x PATH \
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
