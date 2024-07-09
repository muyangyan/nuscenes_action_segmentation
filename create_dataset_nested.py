
from nuscenes_utils import *

from shapely.geometry import Polygon

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pickle
import argparse
import random



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

def mask_visible_instance_objects(args, pose, objects):
    radius = args.radius
    pos = pose[:2]
    angle = pose[2]

    #check if position of object is within rotated square viewport
    def is_visible(object, radius, pos, angle):
        point = np.array(object['pos'])
        center = np.array(pos)
        translated_point = point - center
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])
        
        rotated_point = np.dot(rotation_matrix, translated_point)
        return np.all(np.abs(rotated_point) <= radius)

    visible_objects = [i for i in objects if is_visible(i, radius, pos, angle)]
    return visible_objects

#==================================================================

def create_scene_graphs(ego_pos, instance_objects, map_objects):
    return None

'''
create the scene table using nuscenes_utils
'''
def load_scene_data(nusc):
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
        s = Scene(nusc, scene_name)
        scenes.append(s)
        s.extract_data(map=True)
        s.segment_actions()
        print(i, scene_name, 'actions:', len(s.rich_actions), 'frames:', len(s.data))
        scene_data.append(s.output_data())
    return scene_data


'''
get trajectories for each scene
trajectories are each a sequence of vectors in the format (x, y, angle)
without augmentation, just use ego pose as trajectory
augmentation simply adds a displacement vector
'''
def create_trajectories(scene_data, augment=True, trajs_per_scene=5, radius=5):
    trajs = []
    traj_scene_map = [] #takes in trajectory index, outputs the corresponding scene index
    for i, scene in enumerate(scene_data):
        for _ in range(trajs_per_scene):
            traj = []
            traj_scene_map.append(i)

            #add noise in the form of displacement at each frame and a random fixed angle
            delta_pos = brownian_motion_2d_normalized(len(scene),d=radius)
            angle = random.uniform(0, 2*np.pi)
            for k,frame in enumerate(scene):
                cam_pos = np.array(frame['pos'])
                if augment:
                    cam_pos += delta_pos[k]
                traj.append([cam_pos[0], cam_pos[1], angle]) #use constant angle of 0 for now, can be noiseified later
            trajs.append(traj)
    return trajs, traj_scene_map


def output_to_file(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file)

def get_bitmask(args, pose, nusc_map):
    radius = args.radius
    bitmask_dim = args.bitmask_dim

    pos = pose[:2]
    angle = pose[2]
    box = [pos[0],pos[1],radius*2,radius*2]
    map = nusc_map.get_map_mask(box, angle, canvas_size=bitmask_dim)
    return map

def get_visible_map_objects(args, pose, nusc_map):
    radius = args.radius
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

    map_objects = []
    for layer in nusc_map.non_geometric_polygon_layers:
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
                    obj = {'token':record_token, 'layer':layer, 'geoms':visible_polygons}
                    map_objects.append(obj)
        else:
            for record_token in map_records[layer]:
                record = nusc_map.get(layer, record_token)
                polygon_token = record['polygon_token']
                polygon = nusc_map.extract_polygon(polygon_token)
                if not polygon.is_valid:
                    continue
                visible_polygon = polygon.intersection(viewport_poly)
                if visible_polygon.area > 0: #don't include records without any intersection
                    obj = {'token':record_token, 'layer':layer, 'geom':visible_polygon}
                    map_objects.append(obj)
    for layer in nusc_map.non_geometric_line_layers:
        for record_token in map_records[layer]:
            record = nusc_map.get(layer, record_token)
            line = nusc_map.extract_line(record['line_token'])
            if line.is_empty:
                continue
            visible_line = line.intersection(viewport_poly)
            if visible_line.length > 0: #don't include records without any intersection
                obj = {'token':record_token, 'layer':layer, 'geom':visible_line}
                map_objects.append(obj)
    
    return map_objects

def get_frame_data(args, pose, frame):

    radius = args.radius

    #to make the scene graph, we need the annotated instances as well as the road elements visible within our view
    nusc_map = frame['map']
    frame_anns = frame['anns'] #query token, category, position from scene table
    action = frame['action']
    ego_pos = frame['pos']

    instance_objects = mask_visible_instance_objects(args, pose, frame_anns)
    bitmask = get_bitmask(args, pose, nusc_map)
    map_objects = get_visible_map_objects(args, pose, nusc_map)

    #scene graph creation
    scene_graph = create_scene_graphs(ego_pos, instance_objects, map_objects)

    return {'pose':pose, 'ego_pos':ego_pos, 'radius':radius, 'bitmask':bitmask, 'instance_objects':instance_objects, 'map_objects':map_objects, 'action':action, 'scene_graph':scene_graph}

def main(args):

    version = args.version
    radius = args.radius
    bitmask_dim = args.bitmask_dim
    filename = args.filename
    trajs_per_scene = args.trajs_per_scene
    noise_radius = args.noise_radius

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    args.nusc = nusc

    print('====loading scene data==========')
    scene_data = load_scene_data(nusc)
    print('done')

    print('====creating trajectories=======')
    trajs, traj_scene_map = create_trajectories(scene_data, trajs_per_scene=trajs_per_scene, radius=noise_radius)
    print('done')

    #loop through all trajectories and generate final data
    print('====getting data================')
    full_data = []
    for traj_idx, traj in enumerate(trajs):
        scene_idx = traj_scene_map[traj_idx]
        traj_data = []

        for time_idx, pose in enumerate(traj):
            this_scene = scene_data[scene_idx][time_idx]
            frame_data = get_frame_data(args, pose, this_scene)
            traj_data.append(frame_data)
        full_data.append(traj_data)
    print('done')

    
    print('====outputting to %s========' % filename)
    output_to_file(filename, full_data)
    print('done')

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', type=str, default='v1.0-mini')
parser.add_argument('-f', '--filename', type=str, default='nuscenes_processed_train.pkl')
parser.add_argument('-t', '--trajs_per_scene', type=float, default=3)
parser.add_argument('-l', '--noise_radius', type=float, default=3)
parser.add_argument('-r', '--radius', type=float, default=10)
parser.add_argument('-d', '--bitmask_dim', type=tuple, default=(20,20))

args = parser.parse_args()

main(args)

