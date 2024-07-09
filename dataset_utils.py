
import pickle

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import imageio

from matplotlib.patches import Arrow
from shapely.geometry import Polygon
from descartes import PolygonPatch

from nuscenes_utils import colors_map, colors_actions
from nuscenes_utils import nusc_map_so as nusc_map


#visualization methods

def render_trajectory(filename, traj_data, layers=nusc_map.non_geometric_layers, figsize=(10,10), alpha=0.5):
    frames = []
    for frame_data in traj_data:
        fig = render_frame(frame_data, layers, figsize=figsize, alpha=alpha)

        #add frame to gif
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    imageio.mimsave(filename, frames, duration=0.2)


def render_frame(frame_data, layers=nusc_map.non_geometric_layers, figsize=(10,10), alpha=0.5):

    pos = frame_data['pose'][:2]
    angle = frame_data['pose'][2]
    radius = frame_data['radius']
    action = frame_data['action']
    ego_pos = frame_data['ego_pos']

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
    for map_object in frame_data['map_objects']:
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
    for instance_object in frame_data['instance_objects']:
        x = instance_object['pos'][0]
        y = instance_object['pos'][1]
        category = instance_object['category']
        ax.scatter(x, y, color='m', marker='o', s=40)
        ax.annotate(category, (x, y))

    #render ego vehicle and label with action
    ax.scatter(ego_pos[0], ego_pos[1], color=colors_actions[action], marker='o', s=40, alpha=1)
    ax.annotate(action, (ego_pos[0], ego_pos[1]))


    return fig


# Pytorch interfaces

class NuScenesCustom(Dataset):
    def __init__(self, path):
        self.path = path

        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #unflatten actions
        return self.data[idx]
