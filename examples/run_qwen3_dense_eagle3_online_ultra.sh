#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# support tp8 train eagle3 for Qwen3-4B/8B/32B
NUM_GPUS=${1:-4}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path /media/qwen3_32b \
    --draft-model-config $ROOT_DIR/configs/qwen3-32b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/ultrachat.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen3-32B-eagle3_32b_ultra \
    --num-epochs 6 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen3 \
    --cache-dir $ROOT_DIR/cache_qwen3_32B_ultra \
    --embedding-key model.embed_tokens.weight \
    --tp-size $NUM_GPUS \
    --ttt-length 7
