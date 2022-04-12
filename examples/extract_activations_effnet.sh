#!/bin/sh

set -e

# Extracts activations with EffNet

filelist=$1
gpu_id=$2

CUDA_VISIBLE_DEVICES=${gpu_id} TF_CPP_MIN_LOG_LEVEL=2 python ../src/extract.py \
    /home/palonso/data/specs-dir/ \
    activations-dir/ \
    ${filelist} \
    --feature effnet_b0_3M \
    --model discogs-effnet-bs64-1.pb \
    --from-melspectrogram \
    --batch-size 64 \
    --output activations
