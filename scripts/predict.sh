python main_nusc.py --hidden_dim 512 --n_encoder_layer 2 --n_decoder_layer 2 \
    --n_query 20 --seg --task long --pos_emb --anticipate \
    --input_type=nusc_bitmasks_scenegraphs --max_pos_len 3100 --sample_rate 6 --predict --mode=train --split=1