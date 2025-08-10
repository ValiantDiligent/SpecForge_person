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
    --train-data-path $ROOT_DIR/cache/dataset/train-eagle-wq-0802-qwen3-32b-100000.json \
    --output-dir $ROOT_DIR/outputs/Qwen3-32B-eagle3_increment \
    --num-epochs 5 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen3 \
    --cache-dir $ROOT_DIR/cache_qwen3_32B_private_increment \
    --load-from-checkpoint $ROOT_DIR/outputs/Qwen3-32B-eagle3_32b_ultra \
    --weights-only \
    --resume \
    --tp-size $NUM_GPUS \
    --vm-cache-key d33b394de1d2566a72e2bf42d1c6f341 \
    --ttt-length 7
