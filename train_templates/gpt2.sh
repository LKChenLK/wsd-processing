#!/bin/sh

WORK_DIR="/home/nlplab/$(whoami)/wsd-processing/"

# activate virtual env
source ${WORK_DIR}/env/bin/activate

set -ex

MODEL_DIR="${WORK_DIR}/model/gpt2"
PRETRAINED_MODEL="gpt2"
CUDA_VISIBLE_DEVICES=0,1

python "${WORK_DIR}/train.py" \
    --pretrained_model ${PRETRAINED_MODEL} \
    --dataset_path "${WORK_DIR}/data/cambridge/generative" \
    --model_dir ${MODEL_DIR} \
    --train_batch_size 16 \
    --max_length 100 \
    --do_eval \
    --evaluation_strategy epoch \
    --save_steps 2000 \
    --learning_rate 0.0001 \
    --epochs 4 \
    --weight_decay 0.1 \
    --warmup_steps 1000 \
    --lr_scheduler_type cosine

set +ex