
from nuscenes_utils import *

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse

'''
create the scene table using nuscenes_utils
'''
def load_scene_data():
    custom_blacklist = range(481, 518)

    scene_names = []
    for i in range(780):
        if i in nusc_can.can_blacklist or i in custom_blacklist:
            continue
        scene_name = 'scene-%s' % str(i).zfill(4)
        scene = [ i for i in nusc.scene if scene_name in i['name'].lower() ]
        if len(scene) > 0:
            scene_names.append(scene_name)

    print('total valid scene count:', len(scene_names))

    scene_data = []
    scenes = []
    for i,scene_name in enumerate(scene_names):
        s = Scene(scene_name)
        scenes.append(s)
        s.extract_data(map=True)
        s.segment_actions()
        print(i, scene_name, 'actions:', len(s.rich_actions), 'frames:', len(s.data))
        scene_data.append(s.output_data())
    return scene_data


'''
get trajectories for each scene
trajectories are each a sequence of vectors in the format (x, y, angle)
for now just use ego pose as trajectory, introduce noise later
'''
def create_trajectories(scene_data):
    trajs = []
    traj_scene_map = [] #takes in trajectory index, outputs the corresponding scene index
    for i,s in enumerate(scene_data):
        traj = []
        traj_scene_map.append(i)
        for t in s:
            traj.append(t['pos'] + [0]) #use constant angle of 0 for now, can be noiseified later

        trajs.append(traj)
    return trajs, traj_scene_map


def output_to_file(filename, data):

    with open(filename, "wb") as file:
        pickle.dump(data, file)

    print('pickled')


def get_bitmask(args, pose, nusc_map):
    radius = args.radius
    bitmask_dim = args.bitmask_dim

    pos = pose[:2]
    angle = pose[2]
    box = [pos[0],pos[1],radius*2,radius*2]
    map = nusc_map.get_map_mask(box, angle, canvas_size=bitmask_dim)
    return map

def get_map_objects(args, pose, nusc_map):
    radius = args.radius

    pos = pose[:2]
    angle = pose[2] #how to deal with angles in trajectories? for now just assume angle will be 0

    records_intersect_patch = nusc_map.get_records_in_radius(pos[0], pos[1], radius, nusc_map.non_geometric_layers, mode='intersect')
    return records_intersect_patch

def get_frame_data(args, this_scene, pose):

    # to make the scene graph, we need the annotated instances as well as the road elements visible within our view
    nusc_map = this_scene['map']
    instance_objects = this_scene['anns'] #query token, category, position from scene table
    bitmask = get_bitmask(args, pose, nusc_map)
    map_objects = get_map_objects(args, pose, nusc_map)
    #for now, leave graph creation to pipeline. 
    objects = [instance_objects, map_objects]
    return bitmask, objects

def main(args):

    radius = args.radius
    bitmask_dim = args.bitmask_dim
    filename = args.filename
    #filename = 'nuscenes_processed_train.pkl'

    scene_data = load_scene_data()
    trajs, traj_scene_map = create_trajectories(scene_data)

    #loop through all trajectories and generate final data
    full_data = []
    for traj_idx, traj in enumerate(trajs):
        scene_idx = traj_scene_map[traj_idx]
        traj_data = []

        for time_idx, pose in enumerate(traj):
            this_scene = scene_data[scene_idx][time_idx]
            frame_data = get_frame_data(args, this_scene, pose)
            traj_data.append(frame_data)
        full_data.append(traj_data)
    output_to_file(filename, full_data)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str, default='nuscenes_processed_train.pkl')
parser.add_argument('-r', '--radius', type=float, default=10)
parser.add_argument('-d', '--bitmask_dim', type=tuple, default=(10,10))

args = parser.parse_args()

main(args)

