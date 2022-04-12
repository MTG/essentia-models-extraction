#!/bin/sh

set -e

# Extracts musicnn-style melspectrograms

TF_CPP_MIN_LOG_LEVEL=2 python ../src/extract.py \
    /home/palonso/audio-dir/ \
    melspectrorgam-dir/ \
    filelist.txt \
    --feature musicnn_melspectrogram
