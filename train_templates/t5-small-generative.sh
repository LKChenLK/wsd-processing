
PRETRAINED_MODEL=t5-small
MODEL_DIR=...

python train.py \
    --pretrained_model ${PRETRAINED_MODEL} \
    --dataset_path ./data/cambridge/generative_choices \
    --model_dir ${MODEL_DIR} \
    --prompt_type generative \
    --train_batch_size 16 \
    --max_length 100 \
    --do_eval \
    --evaluation_strategy epoch \
    --save_steps 1000 \
    --learning_rate 0.001 \
    --epochs 6