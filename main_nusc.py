import os
import sys
import warnings
#warnings.simplefilter("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from opts import parser
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from model.futr import FUTR
from train import train
from predict import predict

from dataset_utils import *
import randomname




# Seed fix
#seed = 13452
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#cudnn.benchmark, cudnn.deterministic = False, True

data_path = '/data/Datasets/nuscenes_custom/data'

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def get_split(data_path, split):
    with open(data_path + '/../splits.json', 'r') as f:
        splits = json.load(f)
        train_traj_list = [str(i) for i in splits[str(split)]['train']]
        test_traj_list = [str(i) for i in splits[str(split)]['test']]
    return train_traj_list, test_traj_list

def create_run_dirs(args):
    while True:
        run_path = os.path.join(args.save_path, randomname.get_name())
        if not os.path.exists(run_path):
            break
    print('saving to', run_path)

    model_save_path = os.path.join(run_path, 'checkpoints')
    log_save_path = os.path.join(run_path, 'log')
    params_save_path = os.path.join(run_path, 'params.json')

    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)


    with open(params_save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    return model_save_path, log_save_path

def prepare_train_objs(args, n_class, pad_idx, device):
    model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer).to(device)
    if args.predict:
        return model

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs)
    criterion = nn.MSELoss(reduction = 'none')

    if args.ddp:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    return model, optimizer, scheduler, criterion

def prepare_dataloader(args, data_path, traj_list, pad_idx, n_class):
    dataset = NuScenesDataset(data_path, traj_list, pad_idx, n_class, n_query=args.n_query)
    if args.ddp:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, \
                                                    shuffle=False, num_workers=args.workers,
                                                    collate_fn=dataset.my_collate,
                                                    sampler=DistributedSampler(dataset))
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, \
                                                    shuffle=True, num_workers=args.workers,
                                                    collate_fn=dataset.my_collate)
    return dataloader

def main_train(device, args, model_save_path, log_save_path, world_size=None):
    #device is the index of GPU being used, or torch.device('cpu')

    if args.ddp:
        ddp_setup(device, world_size)
        print("Process started. device: %d" % device)
        if device != 0:
            sys.stdout = open(os.devnull, 'w')

    print('Epoch : ', args.epochs)
    print("batch size : ", args.batch_size)
    print("Split : ", args.split)

    n_class = len(actions) + 1
    pad_idx = n_class + 1

    train_traj_list, _ = get_split(data_path, args.split)

    model, optimizer, scheduler, criterion = prepare_train_objs(args, n_class, pad_idx, device)

    train_loader = prepare_dataloader(args, data_path, train_traj_list, pad_idx, n_class)

    # Training ===========================================
    train(args, model, train_loader, optimizer, scheduler, criterion,
                    model_save_path, log_save_path, pad_idx, device)
    if args.ddp:
        destroy_process_group()

def main_predict(args):

    if args.cpu:
        device = torch.device('cpu')
        print('using cpu')
    else:
        device = torch.device('gpu')
        print('using gpu' + device)
    print('Epoch : ', args.epochs)
    print("batch size : ", args.batch_size)
    print("Split : ", args.split)

    n_class = len(actions) + 1
    pad_idx = n_class + 1

    _, test_traj_list = get_split(data_path, args.split)

    model = prepare_train_objs(args, n_class, pad_idx, device)

    #TODO: create way to save prediction results
    obs_perc = [0.2, 0.3, 0.5]
    run_path = os.path.join(args.save_path, args.test_run)
    model_path = os.path.join(run_path, 'checkpoints/checkpoint%d.ckpt' % args.test_checkpoint)
    print("Predict with ", model_path)

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    for obs_p in obs_perc :
        testset = NuScenesDataset(data_path, test_traj_list, pad_idx, n_class, n_query=args.n_query, obs_p=obs_p, mode='test')
        #predict(data_path, model, test_traj_list, obs_p, n_class, actions, actions_dict, device)
        predict(args, testset, model, device)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.predict:
        print('predict mode')
        main_predict(args)
    else:
        print('train mode')
        model_save_path, log_save_path = create_run_dirs(args)

        if args.ddp:
            print('using gpu with ddp')

            if not args.world_size:
                world_size = torch.cuda.device_count()
            else:
                world_size = args.world_size
            mp.spawn(main_train, args=(args, model_save_path, log_save_path, world_size), nprocs=world_size)

        else:
            if args.cpu:
                print('using cpu')
                main_train(torch.device('cpu'), args, model_save_path, log_save_path)
            else:
                print('using gpu')
                main_train(0, args, model_save_path, log_save_path)
