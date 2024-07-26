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
import mlflow

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
    print('runs : ', args.runs)
    print('model type : ', args.model)
    print('input type : ', args.input_type)
    print('Epoch : ', args.epochs)
    print("batch size : ", args.batch_size)
    print("Split : ", args.split)

    split = args.split


    n_class = len(actions) + 1
    pad_idx = n_class + 1

    with open('splits.json', 'r') as f:
        splits = json.load(f)
        train_traj_list = [str(i) for i in splits[str(split)]['train']]
        test_traj_list = [str(i) for i in splits[str(split)]['test']]

    # Model specification
    model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer).to(device)

    #delim = '_'
    #inputs_string = delim.join(args.input_types)
    model_save_path = os.path.join('./save_dir', args.dataset, args.task, 'model/transformer', 'split'+split, args.input_type, \
                                    'runs'+str(args.runs))
    results_save_path = os.path.join('./save_dir', args.dataset, args.task, 'results/transformer', 'split'+split,
                                    args.input_type )
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)


    #model = nn.DataParallel(model).to(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs)
    criterion = nn.MSELoss(reduction = 'none')


    if args.predict :
        obs_perc = [0.2, 0.3, 0.5]
        results_save_path = results_save_path +'/runs'+ str(args.runs) +'.txt'
        model_path = os.path.join(model_save_path, 'checkpoint%d.ckpt' % args.checkpoint)
        print("Predict with ", model_path)

        for obs_p in obs_perc :
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            testset = NuScenesDataset(data_path, test_traj_list, pad_idx, n_class, args.node_encoding_dim, n_query=args.n_query, obs_p=obs_p, mode='test')
            #predict(data_path, model, test_traj_list, obs_p, n_class, actions, actions_dict, device)
            predict(args, testset, model, device)
    else :
        # Training
        #trainset = BaseDataset(video_list, actions_dict, features_path, gt_path, pad_idx, n_class, n_query=args.n_query, args=args)
        trainset = NuScenesDataset(data_path, train_traj_list, pad_idx, n_class, args.node_encoding_dim, n_query=args.n_query, mode='train')
        train_loader = DataLoader(trainset, batch_size=args.batch_size, \
                                                    shuffle=True, num_workers=args.workers,
                                                    collate_fn=trainset.my_collate)
        train(args, model, train_loader, optimizer, scheduler, criterion,
                     model_save_path, pad_idx, device )


if __name__ == '__main__':
    main()