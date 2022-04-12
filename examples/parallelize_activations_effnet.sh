#!/bin/bash

# Uses tmux to launch 4 processes to parallelize the extractors.
# A single filelist can be split using the `split` command.

tmux start-server

tmux new-session -d -s sess -n gpu0 './extract_activations_effnet.sh gt_test_all_400l_processed.tsv.aa 0'
tmux split-window -t sess:0 './extract_activations_effnet.sh gt_test_all_400l_processed.tsv.ab 1'
tmux split-window -t sess:0 './extract_activations_effnet.sh gt_test_all_400l_processed.tsv.ac 2'
tmux split-window -t sess:0 './extract_activations_effnet.sh gt_test_all_400l_processed.tsv.ad 3'
