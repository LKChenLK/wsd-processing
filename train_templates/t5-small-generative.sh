{
    "training_args": {
        "train_batch_size": 16,
        "max_length": 100,
        "do_eval": true,
        "evaluation_strategy": "epoch",
        "save_steps": 1000,
        "learning_rate": 1e-3,
        "epochs": 5
    },
    "prompt_type": "generative",
    "dataset_path": "data/cambridge/generative",
    "model_name": "t5-small"
}

python train.py \
    --pretrained_model t5 \
    --dataset_path ./data/cambridge/generative_choices \
    --model_dir $MODEL_DIR \
    --prompt_type generative \
    --train_batch_size 16 \
    --max_length 100 \
    --do_eval \
    --evaluation_strategy epoch \
    --save_steps 1000 \
    --learning_rate 0.001 \
    --num_train_epochs 6