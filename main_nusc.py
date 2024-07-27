import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from opts import parser
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

#from utils import read_mapping_dict
#from data.basedataset import BaseDataset
from model.futr import FUTR
from train import train
from predict import predict

from dataset_utils import *
import randomname


from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda')

# Seed fix
#seed = 13452
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#cudnn.benchmark, cudnn.deterministic = False, True

data_path = '/data/Datasets/nuscenes_custom/data'

def main():
    args = parser.parse_args()


    if args.cpu:
        device = torch.device('cpu')
        print('using cpu')
    else:
        device = torch.device('cuda')
        print('using gpu')
    print('Epoch : ', args.epochs)
    print("batch size : ", args.batch_size)
    print("Split : ", args.split)

    n_class = len(actions) + 1
    pad_idx = n_class + 1
    node_dim = args.node_encoding_dim

    with open(data_path + '/../splits.json', 'r') as f:
        splits = json.load(f)
        train_traj_list = [str(i) for i in splits[str(args.split)]['train']]
        test_traj_list = [str(i) for i in splits[str(args.split)]['test']]

    # Model specification
    model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer).to(device)

    model = nn.DataParallel(model).to(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs)
    criterion = nn.MSELoss(reduction = 'none')


    if args.predict :
        obs_perc = [0.2, 0.3, 0.5]
        run_path = os.path.join(args.save_path, args.test_run)
        model_path = os.path.join(run_path, 'checkpoints/checkpoint%d.ckpt' % args.test_checkpoint)
        print("Predict with ", model_path)

        for obs_p in obs_perc :
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            testset = NuScenesDataset(data_path, test_traj_list, pad_idx, n_class, node_dim, n_query=args.n_query, obs_p=obs_p, mode='test')
            #predict(data_path, model, test_traj_list, obs_p, n_class, actions, actions_dict, device)
            predict(args, testset, model, device)
    else :

        # Setup experiment directory=============================

        #generate a unique run directory
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

        writer = SummaryWriter(log_save_path)

        with open(params_save_path, 'w') as f:
            json.dump(args.__dict__, f, indent=2)


        # Training ===========================================
        trainset = NuScenesDataset(data_path, train_traj_list, pad_idx, n_class, node_dim, n_query=args.n_query, mode='train')
        train_loader = DataLoader(trainset, batch_size=args.batch_size, \
                                                    shuffle=True, num_workers=args.workers,
                                                    collate_fn=trainset.my_collate)
        train(args, model, train_loader, optimizer, scheduler, criterion,
                     model_save_path, pad_idx, device, writer )


if __name__ == '__main__':
    main()