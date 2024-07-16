
import pickle
import json

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import imageio
import os

from matplotlib.patches import Arrow
from shapely.geometry import Polygon
from descartes import PolygonPatch
#doesn't import anything from dataset creation scripts, only uses custom dataset directy

actions = ['stop', 'back', 'drive straight', 'accelerate', 'decelerate', 'turn left', 'turn right', 'uturn', 'change lane left', 'change lane right', 'overtake', 'END']

edge_labels = ["close to", "far from", "in front of", "behind", "left of", "right of", "on", "adjacent to", "overlapping with"]

#visualization
colors_actions = ['red', 'white', 'blue', 'green', 'yellow', 'orange', 'magenta', 'c', 'Salmon', 'Salmon', 'aquamarine', 'black'] 

colors_map = dict(drivable_area='#a6cee3',
                             road_segment='#1f78b4',
                             road_block='#b2df8a',
                             lane='#33a02c',
                             ped_crossing='#fb9a99',
                             walkway='#e31a1c',
                             stop_line='#fdbf6f',
                             carpark_area='#ff7f00',
                             road_divider='#cab2d6',
                             lane_divider='#6a3d9a',
                             traffic_light='#7e772e')

#reading data
folders_exts = {'cam_poses':'.pt', 'bitmasks':'.pt', 'scene_graphs':'.pt', 'actions':'.pt', 'objects':'.pkl', 'metadata':'.json'}

actions_dict = {val:idx for idx, val in enumerate(actions)}

non_geometric_polygon_layers = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                                             'walkway', 'stop_line', 'carpark_area']
non_geometric_line_layers = ['road_divider', 'lane_divider', 'traffic_light']
non_geometric_layers = non_geometric_polygon_layers+non_geometric_line_layers

#visualization methods

def select_frame(traj_data, i):
    traj_data.pop('metadata', None)
    return {key: seq[i] for key, seq in traj_data.items()}

def render_trajectory(filename, traj_data, layers=non_geometric_layers, figsize=(10,10), alpha=0.5):
    frames = []
    for i in range(len(traj_data['cam_poses'])):
        frame_data = select_frame(traj_data, i)
        fig = render_frame(frame_data, layers, figsize=figsize, alpha=alpha)

        #add frame to gif
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    imageio.mimsave(filename, frames, duration=0.2)


def render_frame(frame_data, layers=non_geometric_layers, figsize=(10,10), alpha=0.5):

    pos = frame_data['cam_poses'][:2]
    angle = frame_data['cam_poses'][2]
    radius = frame_data['cam_poses'][3]
    action = int(frame_data['actions'])
    objects = frame_data['objects']
    map_objects = [val for val in objects.values() if val['type'] == 'map']
    instance_objects = [val for val in objects.values() if val['type'] == 'instance']

    #scene_graph = frame_data['scene_graphs']

    #ego_pos = frame_data['ego_pos']

    #calculations==============
    outer_radius = radius * ( abs(np.sin(angle)) + abs(np.cos(angle)) )
    r = radius
    R = outer_radius
    c = R - ( 2 * r * np.cos(angle % (np.pi/2)) )
    patch = [pos[0]-R,pos[1]-R,pos[0]+R,pos[1]+R]
    rotated_box_points = [(R,c), (-c, R), (-R, -c), (c, -R)] #global positions of box vertices
    rotated_box_points = [(i[0]+pos[0], i[1]+pos[1]) for i in rotated_box_points] #shift center to pos

    #plot settings
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(patch[0], patch[2])
    ax.set_ylim(patch[1], patch[3])

    #render viewport
    viewport_poly = Polygon(rotated_box_points)
    ax.add_patch(PolygonPatch(viewport_poly, fc='none'))

    #render map objects
    for map_object in map_objects:
        layer = map_object['layer']
        if layer not in layers:
            continue

        #assumes layers are added in the correct order 
        if layer in non_geometric_polygon_layers:
            if layer == 'drivable_area':
                polygons = map_object['geoms']
                for polygon in polygons:
                    ax.add_patch(PolygonPatch(polygon, fc=colors_map[layer], alpha=alpha,
                                                        label=None))
            else:
                polygon = map_object['geom']
                ax.add_patch(PolygonPatch(polygon, fc=colors_map[layer], alpha=alpha,
                                                        label=None))
        else: # line layer
            line = map_object['geom']
            xs, ys = line.xy
            if layer == 'traffic_light':
                ax.add_patch(Arrow(xs[0], ys[0], xs[1]-xs[0], ys[1]-ys[0], color=colors_map[layer],
                                   label=None))
            else:
                ax.plot(xs, ys, color=colors_map[layer], alpha=alpha, label=None)

    #render instance objects
    for instance_object in instance_objects:
        x = instance_object['pos'][0]
        y = instance_object['pos'][1]
        category = instance_object['category']

        if category == 'ego':
            color = colors_actions[action]
            label = actions[action]
        else:
            label = category
            color='m'

        ax.scatter(x, y, color=color, marker='o', s=40)
        ax.annotate(label, (x, y))

    return fig


# Pytorch interfaces

class NuScenesDataset(Dataset):
        
    #pad index?
    #nclass?
    #nquery?
    #other args
    def __init__(self, root, traj_list, pad_idx, n_class, n_query=8, obs_perc=0.2, mode='test'):
        self.root = root
        self.mode = mode
        self.traj_list = []
        self.n_class = n_class
        self.n_query = n_query
        self.pad_idx = pad_idx
        self.NONE = self.n_class - 1

        traj_list = traj_list
        if self.mode == 'train' or self.mode == 'val':
            for traj in traj_list:
                self.traj_list.append([traj, .2])
                self.traj_list.append([traj, .3])
                self.traj_list.append([traj, .5])
        elif self.mode == 'test' :
            for traj in traj_list:
                self.traj_list.append([traj, obs_perc])

    def __len__(self):
        return len(self.traj_list)

    def __getitem__(self, idx):
        traj_file, obs_perc = self.traj_list[idx]
        obs_perc = float(obs_perc)
        item = self._make_input(traj_file, obs_perc)
        return item

    def _make_input(self, traj_file, obs_perc):

        #keys are subfolders, values are extensions
        item = {}

        #read all files in
        for folder in folders_exts.keys():
            folder_path = self.root + '/' + folder

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            ext = folders_exts[folder]
            file_path = folder_path + '/' + traj_file + ext

            if ext == '.pt':
                subdata = torch.load(file_path)
            elif ext == '.pkl':
                with open(file_path, 'rb') as file:
                    subdata = pickle.load(file)
            else:
                with open(file_path, 'r') as file:
                    subdata = json.load(file)

            item.update({folder:subdata})

        traj_len = len(item['actions'])
        observed_len = int(obs_perc*traj_len)
        pred_len = int(0.5*traj_len)

        start_frame = 0

        #just get bitmasks for now
        features = item['bitmasks'][start_frame : start_frame + observed_len]
        past_label = item['actions'][start_frame : start_frame + observed_len] #[S]

        future_content = \
        item['actions'][start_frame + observed_len : start_frame + observed_len + pred_len] #[T]
        trans_future, trans_future_dur = self.seq2transcript(future_content.int())
        trans_future = torch.cat((trans_future, torch.Tensor([self.NONE]).int()))
        trans_future_target = trans_future #target

        # add padding for future input seq
        trans_seq_len = len(trans_future_target)
        diff = self.n_query - trans_seq_len

        '''
        if diff > 0 :
            tmp = np.ones(diff)*self.pad_idx
            trans_future_target = np.concatenate((trans_future_target, tmp))
            tmp_len = np.ones(diff+1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))
        elif diff < 0 :
            trans_future_target = trans_future_target[:self.n_query]
            trans_future_dur = trans_future_dur[:self.n_query]
        else :
            tmp_len = np.ones(1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))
        '''


        if diff > 0:
            tmp = torch.ones(diff) * self.pad_idx
            trans_future_target = torch.cat((trans_future_target, tmp))
            tmp_len = torch.ones(diff + 1) * self.pad_idx
            trans_future_dur = torch.cat((trans_future_dur, tmp_len))
        elif diff < 0:
            trans_future_target = trans_future_target[:self.n_query]
            trans_future_dur = trans_future_dur[:self.n_query]
        else:
            tmp_len = torch.ones(1) * self.pad_idx
            trans_future_dur = torch.cat((trans_future_dur, tmp_len))

        tmp_item = {'features' : features,
                    'past_label': past_label,
                    'trans_future_dur':trans_future_dur,
                    'trans_future_target' : trans_future_target,
                    }
 
        return tmp_item

    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['features'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_dur = [item['trans_future_dur'] for item in batch]
        b_trans_future_target = [item['trans_future_target'] for item in batch]

        batch_size = len(batch)

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self.pad_idx)
        b_trans_future_dur = torch.nn.utils.rnn.pad_sequence(b_trans_future_dur, batch_first=True,
                                                        padding_value=self.pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self.pad_idx)

        batch = [b_features, b_past_label, b_trans_future_dur, b_trans_future_target]

        return batch

    def seq2idx(self, seq):
        idx = torch.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = self.actions_dict[seq[i]]
        return idx

    def seq2transcript(self, seq):
        transcript_action = []
        transcript_dur = []
        action = seq[0]
        transcript_action.append(action)
        last_i = 0
        for i in range(len(seq)):
            if action != seq[i]:
                action = seq[i]
                transcript_action.append(action)
                duration = (i-last_i)/len(seq)
                last_i = i
                transcript_dur.append(duration)
        duration = (len(seq)-last_i)/len(seq)
        transcript_dur.append(duration)
        return torch.stack(transcript_action), torch.Tensor(transcript_dur)

class NuScenesSimple(Dataset):
    def __init__(self, root, traj_list, obs_perc=0.2, mode='test'):
        self.root = root
        self.mode = mode
        self.traj_list = []
        traj_list = traj_list
        if self.mode == 'train' or self.mode == 'val':
            for traj in traj_list:
                self.traj_list.append([traj, .2])
                #self.traj_list.append([traj, .3])
                #self.traj_list.append([traj, .5])
        elif self.mode == 'test' :
            for traj in traj_list:
                self.traj_list.append([traj, obs_perc])

    def __len__(self):
        return len(self.traj_list)

    def __getitem__(self, idx):
        traj_file, obs_perc = self.traj_list[idx]
        obs_perc = float(obs_perc)
        item = self._make_input(traj_file, obs_perc)
        return item

    def _make_input(self, traj_file, obs_perc):

        #TODO: use obs percentage, feature slicing

        #TODO: write collate function

        #keys are subfolders, values are extensions
        item = {}

        #read all files in
        #for now just do actions, no past-future split, no transcript
        for folder in folders_exts.keys():
            folder_path = self.root + '/' + folder

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            ext = folders_exts[folder]
            file_path = folder_path + '/' + traj_file + ext

            if ext == '.pt':
                subdata = torch.load(file_path)
            elif ext == '.pkl':
                with open(file_path, 'rb') as file:
                    subdata = pickle.load(file)
            else:
                with open(file_path, 'r') as file:
                    subdata = json.load(file)

            item.update({folder:subdata})
        
        return item