

WORK_DIR="/home/nlplab/$(whoami)/wsd-processing/"

# activate virtual env
source ${WORK_DIR}/env/bin/activate

set -ex

# for loading the tokeniser
PRETRAINED_MODEL="t5-small"

# <model_dir>/checkpoint-XXXX ; LOAD model from here.
CONT_DIR=".../checkpoint-XXXX"

# SAVE continue-trained model here
MODEL_DIR="${WORK_DIR}/model/..."

python "${WORK_DIR}/train.py" \
    --pretrained_model ${PRETRAINED_MODEL} \
    --dataset_path "${WORK_DIR}/data/cambridge/generative" \
    --model_dir ${MODEL_DIR} \
    --prompt_type generative \
    --train_batch_size 16 \
    --max_length 100 \
    --do_eval \
    --evaluation_strategy epoch \
    --save_steps 1000 \
    --learning_rate 0.001 \
    --epochs 6 \
    --cont_train_model_dir ${CONT_DIR}

set +ex
