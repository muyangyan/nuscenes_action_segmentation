# first line for fixed args
python main_nusc.py \
--seg --anticipate --pos_emb --n_query 20 --n_encoder_layer 2 --n_decoder_layer 2 --max_pos_len 3100 \
\
--batch_size 32 \
--epochs 100 \
--lr 1e-4 \
\
--hidden_dim 512 \
--dropout 0.5 \
\
--conv_type gat \
--gat_heads 4 \
--gat_dropout 0.6 \
--split=1