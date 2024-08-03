import torch
import torch.nn as nn
import numpy
import pdb
import os
import json
import pandas as pd
import copy
from collections import defaultdict
import numpy as np
from utils import normalize_duration, eval_file
from dataset_utils import NuScenesDataset
from dataset_utils import actions, actions_dict
from utils import cal_performance

def predict(args, testset, model, device):
    model.eval()
    with torch.no_grad():

        obs_p = testset.obs_p
        
        eval_p = [0.1, 0.2, 0.3, 0.5]

        T_actions = np.zeros((len(eval_p), len(actions_dict)))
        F_actions = np.zeros((len(eval_p), len(actions_dict)))

        pred_p = 0.5
        NONE = testset.n_class-1
        actions_dict_with_NONE = copy.deepcopy(actions_dict)
        actions_dict_with_NONE['NONE'] = NONE

        products = []
        for idx, (traj, data) in enumerate(testset):

            #features, scene_graphs, past_label, trans_dur_future, trans_future_target = data.values()
            features, scene_graphs, _, _, _, gt_seq = data.values()
            features = features.to(device) #[B, S, C]
            if args.input_type in ['nusc_bitmasks_scenegraphs', 'nusc_scenegraphs']:
                scene_graphs = [sg.to(device) for sg in scene_graphs] #[T,B,F] a list of PyG Batches (1 per timestep)
            gt_seq = gt_seq.int().tolist()

            #======================================================================

            #redundant but works for now
            gt_seq = [actions[i] for i in gt_seq]

            vid_len = len(gt_seq)
            past_len = int(obs_p*vid_len)
            future_len = int(pred_p*vid_len)

            past_seq = gt_seq[:past_len]

            if args.input_type in ["nusc_bitmasks_scenegraphs", "nusc_scenegraphs"]:
                inputs = (features.unsqueeze(0), scene_graphs)
            elif args.input_type == "nusc_bitmasks":
                inputs = (features.unsqueeze(0), None)

            outputs = model(inputs, mode='test')
            #======================================================================

            output_action = outputs['action']
            output_dur = outputs['duration']
            output_label = output_action.max(-1)[1]

            # fine the forst none class
            none_mask = None
            none_idx = None
            for i in range(output_label.size(1)) :
                if output_label[0,i] == NONE :
                    none_idx = i
                    break
            none_mask = torch.ones(output_label.shape).type(torch.bool)
            if none_idx is not None :
                #none_mask = torch.ones(output_label.shape).type(torch.bool)
                none_mask[0, none_idx:] = False

            output_dur = normalize_duration(output_dur, none_mask.to(device))

            pred_len = (0.5+future_len*output_dur).squeeze(-1).long()

            pred_len = torch.cat((torch.zeros(1).to(device), pred_len.squeeze()), dim=0)
            predicted = torch.ones(future_len)
            action = output_label.squeeze()

            for i in range(len(action)) :
                predicted[int(pred_len[i]) : int(pred_len[i] + pred_len[i+1])] = action[i]
                pred_len[i+1] = pred_len[i] + pred_len[i+1]
                if i == len(action) - 1 :
                    predicted[int(pred_len[i]):] = action[i]


            prediction = past_seq
            for i in range(len(predicted)):
                prediction = np.concatenate((prediction, [list(actions_dict_with_NONE.keys())[list(actions_dict_with_NONE.values()).index(predicted[i].item())]]))

            #evaluation
            for i in range(len(eval_p)):
                p = eval_p[i]
                eval_len = int((obs_p+p)*vid_len)
                eval_prediction = prediction[:eval_len]
                T_action, F_action = eval_file(gt_seq, eval_prediction, obs_p, actions_dict)
                T_actions[i] += T_action
                F_actions[i] += F_action

            #save actual results to files
            product = {'traj':traj, 'obs_p':obs_p, 'past_len':past_len, 'gt_seq':gt_seq, 'prediction':list(prediction)}
            products.append(product)

        products_df = pd.DataFrame(products)

        results = []
        total_actions = T_actions + F_actions
        for i in range(len(eval_p)):
            acc = 0
            n = 0
            for j in range(len(actions_dict)):
                if total_actions[i,j] != 0:
                    acc += float(T_actions[i,j]/total_actions[i,j])
                    n+=1
            moc = float(acc)/n
            results.append({'obs_perc':obs_p, 'pred_perc':eval_p[i], 'MoC':moc})
            result_str = 'obs. %d '%int(100*obs_p) + 'pred. %d '%int(100*eval_p[i])+'--> MoC: %.4f'%(moc)
            print(result_str)
        print('--------------------------------')

        results_df = pd.DataFrame(results)

        return products_df, results_df
