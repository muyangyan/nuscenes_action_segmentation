
import os
import sys
sys.path.append(os.getcwd())

from dataset_generation.nuscenes_utils import *

from shapely.geometry import Polygon

import torch
import pickle
import argparse
import random
import os
import json

import warnings

from dataset_utils import actions, actions_dict, edges_dict
from dataset_generation.scene_graph_predicates import *

warnings.filterwarnings("ignore")

custom_scene_blacklist = range(481, 518)


#HELPERS===========================================================

'''
used for adding noise to trajectories
'''
def brownian_motion_2d_normalized(n_steps, dt=1.0, sigma=1, d=1):
    random_steps = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size=(n_steps, 2))
    path = np.cumsum(random_steps, axis=0)
    path -= np.mean(path, axis=0)
    max_distance = np.max(np.linalg.norm(path, axis=1))
    path *= d / max_distance
    return path

def output_trajectory(datafolder, data, metadata, name):
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)
        print('Data folder not found, creating')

    metapath = datafolder + '/metadata'
    if not os.path.exists(metapath):
        os.makedirs(metapath)
        print('metadata folder not found, creating')
    
    with open(metapath + '/' + name + '.json', 'w') as file:
        json.dump(metadata, file)

    non_torch_keys = ['objects']

    keys = data.keys()
    for key in keys:
        path = datafolder + '/' + key 
        if not os.path.exists(path):
            os.makedirs(path)
            print('path %s not found, creating' % path)
        
        # how to name the files?
        # Scene number and then index?
        # for now, just a number will do. will need to change when we have splits

        if key in non_torch_keys:
            with open(path + '/' + name + '.pkl', 'wb') as file:
                pickle.dump(data[key], file)
        else:
            torch.save(data[key], path + '/' + name + '.pt')
    

'''
create the scene table using nuscenes_utils
'''
def load_scene_data(nusc):

    scene_names = []
    for i in range(2000):
        if i in nusc_can.can_blacklist or i in custom_scene_blacklist:
            continue
        scene_name = 'scene-%s' % str(i).zfill(4)
        scene = [ i for i in nusc.scene if scene_name in i['name'].lower() ]
        if len(scene) > 0:
            scene_names.append(scene_name)

    print('total valid scene count:', len(scene_names))

    scene_data = []
    scenes = []
    for i,scene_name in enumerate(scene_names):
        s = Scene(nusc, scene_name)
        scenes.append(s)
        s.extract_data(map=True)
        s.segment_actions()
        print(i, scene_name, 'actions:', len(s.rich_actions), 'frames:', len(s.data))
        scene_data.append(s.output_data())
    return scene_data

#==================================================================


'''
get trajectories for each scene
trajectories are each a sequence of vectors in the format (x, y, angle)
without augmentation, just use ego pose as trajectory
augmentation simply adds a displacement vector
'''
def create_trajectories(scene_data, augment=True, trajs_per_scene=5, noise_radius=5, viewport_size=10):
    trajs = []
    traj_scene_map = [] #takes in trajectory index, outputs the corresponding scene index
    for i, scene in enumerate(scene_data):
        for _ in range(trajs_per_scene):
            traj = []
            traj_scene_map.append(i)

            #add noise in the form of displacement at each frame and a random fixed angle
            #TODO: add noise in angle and viewport radius as well (simulating height of drone off ground)
            delta_pos = brownian_motion_2d_normalized(len(scene),d=noise_radius)
            angle = random.uniform(0, 2*np.pi)
            for k,frame in enumerate(scene):
                cam_pos = np.array(frame['pose'][:2])
                if augment:
                    cam_pos += delta_pos[k]
                traj.append([cam_pos[0], cam_pos[1], angle, float(viewport_size)]) 
            trajs.append(traj) #convert to tensor now for dataset
    return trajs, traj_scene_map


def mask_visible_instance_objects(pose, objects):
    pos = pose[:2]
    angle = pose[2]
    radius = pose[3]

    #check if position of object is within rotated square viewport
    def is_visible(object, radius, pos, angle):
        point = np.array(object['pose'][:2])
        center = np.array(pos)
        translated_point = point - center
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])
        
        rotated_point = np.dot(rotation_matrix, translated_point)
        return np.all(np.abs(rotated_point) <= radius)

    visible_objects = {i['token']:{k:i[k] for k in ('category', 'pose')} for i \
                       in objects if is_visible(i, radius, pos, angle)}
    return visible_objects

def get_bitmask(pose, nusc_map, layers, bitmask_dim):
    pos = pose[:2]
    angle = pose[2]
    radius = pose[3]
    box = [pos[0],pos[1],radius*2,radius*2]
    map = nusc_map.get_map_mask(box, angle, layer_names=layers, canvas_size=bitmask_dim)
    map = torch.Tensor(map)
    return map

def get_visible_map_objects(pose, nusc_map, layers):
    radius = pose[3]
    pos = pose[:2]
    angle = pose[2]

    #calculations==============
    outer_radius = radius * ( abs(np.sin(angle)) + abs(np.cos(angle)) )
    
    r = radius
    R = outer_radius
    c = R - ( 2 * r * np.cos(angle % (np.pi/2)) )
    rotated_box_points = [(R,c), (-c, R), (-R, -c), (c, -R)] #global positions of box vertices
    rotated_box_points = [(i[0]+pos[0], i[1]+pos[1]) for i in rotated_box_points] #shift center to pos
    
    viewport_poly = Polygon(rotated_box_points)

    #fetch records==============
    map_records = nusc_map.get_records_in_radius(pos[0], pos[1], outer_radius, 
                                            nusc_map.non_geometric_layers, mode='intersect')

    map_objects = {}
    polygon_layers = [i for i in nusc_map.non_geometric_polygon_layers if i in layers]
    line_layers = [i for i in nusc_map.non_geometric_line_layers if i in layers]
    for layer in polygon_layers:
        if layer == 'drivable_area':
            for record_token in map_records[layer]:
                record = nusc_map.get(layer, record_token)
                polygons = [nusc_map.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
                visible_polygons = []
                for polygon in polygons:
                    if not polygon.is_valid:
                        continue
                    visible_record_polygon = polygon.intersection(viewport_poly)
                    if visible_record_polygon.area > 0: #don't include records without any intersection
                        visible_polygons.append(visible_record_polygon)
                if len(visible_polygons) > 0:
                    obj = {'layer':layer, 'geoms':visible_polygons}
                    map_objects.update({record_token:obj})
        else:
            for record_token in map_records[layer]:
                record = nusc_map.get(layer, record_token)
                polygon_token = record['polygon_token']
                polygon = nusc_map.extract_polygon(polygon_token)
                if not polygon.is_valid:
                    continue
                visible_polygon = polygon.intersection(viewport_poly)
                if visible_polygon.area > 0: #don't include records without any intersection
                    obj = {'layer':layer, 'geom':visible_polygon}
                    map_objects.update({record_token:obj})
    for layer in line_layers:
        for record_token in map_records[layer]:
            record = nusc_map.get(layer, record_token)
            line = nusc_map.extract_line(record['line_token'])
            if line.is_empty:
                continue
            visible_line = line.intersection(viewport_poly)
            if visible_line.length > 0: #don't include records without any intersection
                obj = {'layer':layer, 'geom':visible_line}
                map_objects.update({record_token:obj})
    
    return map_objects

'''
predicts edges between two objects
does not do any pruning for ego vehicle
prunes for invalid edges
map_buffer_radius is the buffer we add around map objects to check for adjacency
dist_threshhold is the radius for front, behind, left and right relationships
'''
def predict_edges(src, dest, map_buffer_radius, instance_dist_threshhold):
    if src['type'] == 'map':
        if dest['type'] == 'instance':
            return edges_dict['NONE']
        if intersects_with(src, dest):
            return edges_dict['intersects with']
        if adjacent_to(src, dest, map_buffer_radius):
            return edges_dict['adjacent to']

    elif src['type'] == 'instance':
        if dest['type'] == 'map':
            if on(src, dest):
                return edges_dict['on']
        elif dest['type'] == 'instance':
            if in_front_of(src, dest, instance_dist_threshhold):
                return edges_dict['in front of']
            if behind(src, dest, instance_dist_threshhold):
                return edges_dict['behind']
            if left_of(src, dest, instance_dist_threshhold):
                return edges_dict['left of']
            if right_of(src, dest, instance_dist_threshhold):
                return edges_dict['right of']

    return edges_dict['NONE']

'''
object_dict maps object tokens to their index in the object_token list in the outer context
objects is a dictionary containing the CURRENT objects in the frame and their CURRENT data
output is a 2D torch tensor adjacency matrix (H,W) of integers denoting edge label
'''
def create_scene_graph(args, objects, object_dict):

    n = len(object_dict) #total
    m = len(objects) #visible

    adj_matrix = torch.zeros(n, n)

    # TODO: don't consider pairs in a dense manner, select for just ego and such
    for tok_i, src in objects.items():
        if src['type'] == 'instance' and src['category'] != 'ego':
            continue
        for tok_j, dest in objects.items():
            i = object_dict[tok_i]
            j = object_dict[tok_j]
            if i == j:
                continue
            adj_matrix[i][j] = predict_edges(src, dest, args.map_buffer_radius, args.instance_dist_threshhold)

    return adj_matrix

def get_frame_data(args, pose, frame):
    #to make the scene graph, we need the annotated instances as well as the road elements visible within our view
    nusc_map = frame['map']
    frame_anns = frame['anns'] #query token, category, position from scene table
    action_key = frame['action']
    ego_pose = frame['pose']

    instance_objs = mask_visible_instance_objects(pose, frame_anns)
    bitmask = get_bitmask(pose, nusc_map, args.layers, args.bitmask_dim)
    map_objs = get_visible_map_objects(pose, nusc_map, args.layers)

    for key in instance_objs.keys():
        instance_objs[key].update({'type':'instance'})

    for key in map_objs.keys():
        map_objs[key].update({'type':'map'})

    objects = {'ego':{'type':'instance', 'category':'ego', 'pose':ego_pose}}
    objects.update(instance_objs) #combine all visible objects in this frame into one dictionary
    objects.update(map_objs) #combine all visible objects in this frame into one dictionary

    #convert action to index
    action_idx = actions_dict[action_key]

    return {'bitmask':bitmask, 'objects':objects, 'action':action_idx}

def main(args):

    version = args.version
    viewport_radius = args.viewport_radius
    datafolder = args.datafolder
    trajs_per_scene = args.trajs_per_scene
    noise_radius = args.noise_radius
    start_idx = args.start_idx

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    args.nusc = nusc

    print('====loading scene data==========')
    scene_data_path = './dataset_generation/' + version + '.pkl'
    if os.path.exists(scene_data_path):
        with open(scene_data_path, 'rb') as f:
            scene_data = pickle.load(f)
    else:
        scene_data = load_scene_data(nusc)
        with open(scene_data_path, 'wb') as f:
            pickle.dump(scene_data, f)
            print('dumped pickle file')
    print('done')

    print('====creating trajectories=======')
    trajectories, traj_scene_map = create_trajectories(scene_data, trajs_per_scene=trajs_per_scene, noise_radius=noise_radius, viewport_size=viewport_radius)
    print('done')

    #loop through all trajectories and generate final data
    print('====getting and writing data====')

    trajectories = trajectories[start_idx:]

    for idx, traj in enumerate(trajectories):
        traj_idx = idx + start_idx
        scene_idx = traj_scene_map[traj_idx]

        bitmasks = []
        scene_graphs = []
        objects = []
        actions = []

        object_tokens = set()

        #first pass collects object tokens
        for time_idx, pose in enumerate(traj):
            scene_frame = scene_data[scene_idx][time_idx]
            frame_data = get_frame_data(args, pose, scene_frame)

            object_tokens = object_tokens.union(frame_data['objects'].keys()) # the keys are the object tokens

            bitmasks.append(frame_data['bitmask']) #3D multichannel bitmask
            objects.append(frame_data['objects']) # non-tensor dictionary of objects
            actions.append(frame_data['action'])

        object_tokens = [t for t in object_tokens] #assign indices to objects
        object_dict = { v:i for i,v in enumerate(object_tokens) } #maps token to index in object_tokens

        #second pass create scene graphs
        for time_idx, frame_objs in enumerate(objects):
            #scene graph creation
            scene_graphs.append(create_scene_graph(args, frame_objs, object_dict))
        
        traj = torch.Tensor(traj)
        actions = torch.Tensor(actions)
        bitmasks = torch.stack(bitmasks)
        scene_graphs = torch.stack(scene_graphs)

        data = {'cam_poses':traj, 'bitmasks':bitmasks, 'scene_graphs':scene_graphs, 'objects':objects, 'actions':actions}

        metadata = {'scene_idx':scene_idx, 'object_tokens':object_tokens}

        #write trajectory files
        name = str(traj_idx) #TODO: get a better naming scheme
        output_trajectory(datafolder, data, metadata, name)
        print('written:', name)

    print('done')

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', type=str, default='v1.0-mini')
parser.add_argument('-f', '--datafolder', type=str, default='./data')
parser.add_argument('-t', '--trajs_per_scene', type=float, default=3)
parser.add_argument('-n', '--noise_radius', type=float, default=2)
parser.add_argument('-r', '--viewport_radius', type=float, default=10)
parser.add_argument('-d', '--bitmask_dim', type=tuple, default=(32,32))
parser.add_argument('-l', '--layers', type=list, default=nusc_map_bs.non_geometric_layers)

parser.add_argument('--start_idx', type=int, default=0)

#SG args
parser.add_argument('--map_buffer_radius', type=float, default=1)
parser.add_argument('--instance_dist_threshhold', type=float, default=10)

args = parser.parse_args()

main(args)

