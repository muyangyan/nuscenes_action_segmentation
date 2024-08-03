#!/bin/bash

PYTHON_SCRIPT="main_nusc.py"
ARGS="--predict --seg --anticipate --pos_emb --n_query 20 --n_encoder_layer 2 --n_decoder_layer 2 --max_pos_len 3100 \
--input_type nusc_scenegraphs \
--batch_size 16 \
--epochs 9999 \
--lr 1e-4 \
--hidden_dim 256 \
--dropout 0.5 \
--conv_type gat \
--gat_heads 4 \
--gat_dropout 0.6 \
--split=1 \
--gnn_pool"


if [ "$1" == "debug" ] ; then
    python -m debugpy --wait-for-client --listen 5678 $PYTHON_SCRIPT $ARGS --test_run $2 --test_checkpoint $3
else
    python $PYTHON_SCRIPT $ARGS --test_run $1 --test_checkpoint $2
fi