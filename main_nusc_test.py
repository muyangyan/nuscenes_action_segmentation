import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_utils import *

from opts import parser
from model.futr import FUTR


nusc_data = NuScenesDataset('./data', ['0','1','2'], mode='train')

args=parser.parse_args()

n_class = len(actions) + 1
pad_idx = n_class + 1
device = torch.device('cpu')

model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                        n_query=args.n_query, n_head=args.n_head,
                        num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer).to(device)

#model = nn.DataParallel(model).to(device)

