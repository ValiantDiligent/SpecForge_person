SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_offline.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/723724.jsonl \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states/723824 \
    --output-dir $ROOT_DIR/outputs/qwen3_offline \
    --num-epochs 6 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen3 \
    --cache-dir $ROOT_DIR/cache/qwen3
