
WORK_DIR="/home/nlplab/$(whoami)/wsd-processing/"

# activate virtual env
source ${WORK_DIR}/env/bin/activate

set -ex

MODEL_DIR="${WORK_DIR}/model/..."
PRETRAINED_MODEL=t5-small

python "${WORK_DIR}/train.py" \
    --pretrained_model ${PRETRAINED_MODEL} \
    --dataset_path "${WORK_DIR}/data/cambridge/generative" \
    --model_dir ${MODEL_DIR} \
    --train_batch_size 16 \
    --max_length 100 \
    --do_eval \
    --evaluation_strategy epoch \
    --save_steps 1000 \
    --learning_rate 0.001 \
    --epochs 6

set +ex
