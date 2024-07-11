
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

from nuscenes_utils import colors_map, colors_actions, actions, edge_labels, folders_exts
from nuscenes_utils import nusc_map_so as nusc_map


#visualization methods

def select_frame(traj_data, i):
    traj_data.pop('metadata', None)
    return {key: seq[i] for key, seq in traj_data.items()}

def render_trajectory(filename, traj_data, layers=nusc_map.non_geometric_layers, figsize=(10,10), alpha=0.5):
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


def render_frame(frame_data, layers=nusc_map.non_geometric_layers, figsize=(10,10), alpha=0.5):

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
        if layer in nusc_map.non_geometric_polygon_layers:
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

class NuScenesCustom(Dataset):
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

    #def my_collate():
