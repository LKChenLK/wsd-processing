#!/bin/sh

MODEL_DIR=...
PRETRAINED_MODEL="gpt2"

python train.py \
    --pretrained_model $PRETRAINED_MODEL \
    --dataset_path ./data/cambridge/generative \
    --model_dir $MODEL_DIR \
    --prompt_type generative \
    --train_batch_size 8 \
    --max_length 100 \
    --do_eval \
    --evaluation_strategy epoch \
    --save_steps 2000 \
    --learning_rate 0.00001 \
    --epochs 4 \
    --weight_decay 0.1 \
    --warmup_steps 1000 \
    --lr_scheduler_type cosine