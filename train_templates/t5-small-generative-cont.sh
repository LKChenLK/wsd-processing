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
    "model_name": "t5-small",
    "cont_train_model_dir": ""
}


# for loading the tokeniser
PRETRAINED_MODEL="t5-small"

# <model_dir>/checkpoint-XXXX ; load model from here.
CONT_DIR=".../checkpoint-XXXX"

# save continue-trained model here
MODEL_DIR=...

python train.py \
    --pretrained_model $PRETRAINED_MODEL \
    --dataset_path ./data/cambridge/generative \
    --model_dir $MODEL_DIR \
    --prompt_type generative \
    --train_batch_size 16 \
    --max_length 100 \
    --do_eval \
    --evaluation_strategy epoch \
    --save_steps 1000 \
    --learning_rate 0.001 \
    --epochs 6 \
    --cont_train_model_dir
