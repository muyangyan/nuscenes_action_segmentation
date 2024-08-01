import argparse
parser = argparse.ArgumentParser()

#main options
parser.add_argument('--predict', "-p", action='store_true', help="predict for whole videos mode")
parser.add_argument("--split", default="1", help='split number')
parser.add_argument("--save_path", default="./runs")
parser.add_argument("--input_type", type=str, default='nusc_bitmasks_scenegraphs')

#Training hardware and parallelization
parser.add_argument("--cpu", action='store_true', help='run in cpu')
parser.add_argument("--ddp", action='store_true', help='use DistributedDataParallel')
parser.add_argument("--world_size", type=int, default=None, help='number of processes for parallelization')


#Training options
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--workers", type=int, default= 10)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_mul", type=float, default=2.0)
parser.add_argument("--weight_decay", type=float, default=5e-3) #5e-3
parser.add_argument("-warmup", '--n_warmup_steps', type=int, default=500)
parser.add_argument("--obs_perc", default=30)
parser.add_argument("--n_query", type=int, default=8)


#FUTR specific parameters
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--n_encoder_layer", type=int, default=2)
parser.add_argument("--n_decoder_layer", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--input_dim", type=int, default=2048)

#Model parameters
parser.add_argument("--seg", action='store_true', help='action segmentation')
parser.add_argument("--anticipate", action='store_true', help='future anticipation')
parser.add_argument("--pos_emb", action='store_true', help='positional embedding')
parser.add_argument("--max_pos_len", type=int, default=2000, help='position embedding number for linear interpolation')

#nusc args
#TODO: make layers selectable
parser.add_argument("--bitmask_channels", default=10)

#graph params
parser.add_argument("--node_categorical_dim", type=int, default=44)
parser.add_argument("--node_hidden_categorical_dim", type=int, default=5)
parser.add_argument("--node_continuous_dim", type=int, default=5)
parser.add_argument("--hidden_edge_dim", type=int, default=5)
parser.add_argument("--sg_hidden_dim", type=int, default=8)
parser.add_argument("--conv_type", default='gat')
parser.add_argument("--gat_heads", type=int, default=4)
parser.add_argument("--gat_dropout", type=float, default=0.6)
parser.add_argument("--gnn_pool", action='store_true')

#testing params
parser.add_argument("--test_run", type=str, default=None)
parser.add_argument("--test_checkpoint", type=int, default=0)


