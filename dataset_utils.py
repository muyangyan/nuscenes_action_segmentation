
import pickle
import json

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, remove_isolated_nodes, mask_select

import matplotlib.pyplot as plt
import imageio
import os

from matplotlib.patches import Arrow
from shapely.geometry import Polygon, MultiPolygon
from descartes import PolygonPatch
import networkx as nx
#doesn't import anything from dataset creation scripts, only uses custom dataset directy

actions = ['stop', 'back', 'drive straight', 'accelerate', 'decelerate', 'turn left', 'turn right', 'uturn', 'change lane left', 'change lane right', 'overtake', 'END', None]

edge_labels = ["NONE", "in front of", "behind", "left of", "right of", "on", "adjacent to", "intersects with"]

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
edges_dict = {val:idx for idx, val in enumerate(edge_labels)}

non_geometric_polygon_layers = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                                             'walkway', 'stop_line', 'carpark_area']
non_geometric_line_layers = ['road_divider', 'lane_divider', 'traffic_light']
non_geometric_layers = non_geometric_polygon_layers+non_geometric_line_layers

categories = [['ego', 'animal', 'human', 'movable_object', 'static_object', 'vehicle'], 
              ['pedestrian', 'barrier', 'debris', 'pushable_pullable', 'trafficcone', 'bicycle_rack', 'bicycle', 'bus', 'car', 'construction', 'emergency', 'motorcycle', 'trailer', 'truck'],
              ['adult', 'child', 'construction_worker', 'personal_mobility', 'police_officer', 'stroller', 'wheelchair', 'bendy', 'rigid', 'ambulance', 'police']]



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
        x = instance_object['pose'][0]
        y = instance_object['pose'][1]
        yaw = instance_object['pose'][2]
        category = instance_object['category']

        if category == 'ego':
            color = colors_actions[action]
            label = actions[action]
        else:
            label = category
            color='m'

        ax.scatter(x, y, color=color, marker='o', s=40)
        ax.annotate(label, (x, y))
        length = radius/10
        ax.arrow(x, y, length*np.cos(yaw), length*np.sin(yaw), width=length*0.05, head_width=length*0.1, linewidth=0)

    return fig

def get_label(node, object_list, objects):
    object = objects[ object_list[node] ]
    if object['type'] == 'map':
        return str(object['layer'])
    if object['type'] == 'instance':
        return str(object['category'])

def get_color(node, object_list, objects):
    object = objects[ object_list[node] ]
    if object['type'] == 'map':
        return 'blue'
    if object['type'] == 'instance':
        return 'red'

def get_layer(node, object_list, objects):
    object = objects[ object_list[node] ]
    if object['type'] == 'map':
        return object['layer']
    else:
        return None


def visualize_graph(adjacency_matrix, object_list, objects, layout, excluded_layers=None, k=3, iterations=50):
    # Convert PyTorch tensor to numpy array
    adj_matrix = adjacency_matrix.cpu().numpy()

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Remove isolated nodes (nodes with no connections)
    G.remove_nodes_from(list(nx.isolates(G)))

    G.remove_nodes_from([node for node in G.nodes if get_layer(node, object_list, objects) in excluded_layers])

    # If the graph is empty after removing isolated nodes, return
    if len(G) == 0:
        print("No connected nodes to display.")
        return

    # Set up the plot
    fig = plt.figure(figsize=(10, 10))
    
    # Generate a layout for the nodes
    if layout == 'spring':
        pos = nx.spring_layout(G, k=k, iterations=iterations)
    elif layout == 'kk':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'fr':
        pos = nx.fruchterman_reingold_layout(G)
    
    # Draw the nodes
    node_colors = [get_color(node, object_list, objects) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors)
    
    # Draw the edges
    nx.draw_networkx_edges(G, pos, arrows=True, width=7)
    
    # Add labels to the nodes
    labels = {node: get_label(node, object_list, objects) for node in G.nodes()}
    #nx.draw_networkx_labels(G, pos, labels, font_size=16)

    edge_labels_graph = {(u, v): edge_labels[ adj_matrix[u][v] ] for u, v in G.edges()}
    #edge_labels = {(u, v): adj_matrix[u][v] for u, v in G.edges()}
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_graph, font_size=16)

    
    # Show the plot
    plt.axis('off')
    plt.tight_layout()
    return fig


# Pytorch interfaces

class NuScenesDataset(Dataset):
        
    #pad index?
    #nclass?
    #nquery?
    #other args
    def __init__(self, root, traj_list, pad_idx, n_class, node_dim, n_query=8, obs_p=0.2, mode='test'):
        self.root = root
        self.mode = mode
        self.traj_list = []
        self.n_class = n_class
        self.node_dim = node_dim
        self.n_query = n_query
        self.pad_idx = pad_idx
        self.obs_p = obs_p
        self.NONE = self.n_class - 1

        if self.mode == 'train' or self.mode == 'val':
            for traj in traj_list:
                self.traj_list.append([traj, .2])
                self.traj_list.append([traj, .3])
                self.traj_list.append([traj, .5])
        elif self.mode == 'test' :
            for traj in traj_list:
                self.traj_list.append([traj, obs_p])

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
            elif ext == '.json':
                with open(file_path, 'r') as file:
                    subdata = json.load(file)

            item.update({folder:subdata})

        traj_len = len(item['actions'])
        observed_len = int(obs_perc*traj_len)
        pred_len = int(0.5*traj_len)

        start_frame = 0

        features = item['bitmasks'][start_frame : start_frame + observed_len]
        sg_adj_matrices = item['scene_graphs'][start_frame : start_frame + observed_len]
        past_label = item['actions'][start_frame : start_frame + observed_len] #[S]
        objects = item['objects'][start_frame : start_frame + observed_len]
        metadata = item['metadata']
        object_tokens = metadata['object_tokens']
        #object_dict = { v:i for i,v in enumerate(object_tokens) } #maps token to index in object_tokens

        # turn scene graphs into PyG data:
        scene_graphs = []
        for sg, objs in zip(sg_adj_matrices, objects):
            node_features = self.encode_objects(objs, object_tokens, self.node_dim)

            edge_index, edge_attr = dense_to_sparse(sg)
            edge_index, edge_attr, mask = remove_isolated_nodes(edge_index, edge_attr, num_nodes=len(node_features))

            node_features = mask_select(node_features, 0, mask)

            data = Data(x=node_features, edge_attr=edge_attr, edge_index=edge_index)
            scene_graphs.append(data)


        future_content = \
        item['actions'][start_frame + observed_len : start_frame + observed_len + pred_len] #[T]
        trans_future, trans_future_dur = self.seq2transcript(future_content.int())
        trans_future = torch.cat((trans_future, torch.Tensor([self.NONE]).int()))
        trans_future_target = trans_future #target

        # add padding for future input seq
        trans_seq_len = len(trans_future_target)
        diff = self.n_query - trans_seq_len

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

        final_item = {'features' : features,
                        'scene_graphs' : scene_graphs,
                        'past_label' : past_label,
                        'trans_future_dur' : trans_future_dur,
                        'trans_future_target' : trans_future_target,
                        }
        if self.mode == 'test':
            final_item.update({'actions' : item['actions']})
 
        return final_item

    def encode_object(self, object, dim):
        #fixed_dim = 5 + len(non_geometric_layers) + sum(len(categories[i]) for i in len(categories))
        #poly_dim = dim - fixed_dim

        type = torch.zeros(2)
        layer = torch.zeros(len(non_geometric_layers))
        polygon = torch.zeros(3) #centroid x, centroid y, area
        category = [torch.zeros(len(c)) for c in categories]
        pose = torch.zeros(3)
        if object['type'] == 'map':
            type[0] = 1

            layer_index = non_geometric_layers.index(object['layer'])
            layer[layer_index] = 1

            if object['layer'] == 'drivable_area':
                geom = object['geoms'][0]
                #possible edge case: geoms is a list of many Polygons, must be made into MultiPolygon
            else:
                geom = object['geom']
            polygon[0] = geom.centroid.x
            polygon[1] = geom.centroid.y
            polygon[2] = geom.area

        else:
            type[1] = 1
            category_string = object['category']
            substrings = category_string.split('.')
            for i,s in enumerate(substrings):
                
                index = categories[i].index(s)
                category[i][index] = 1
            pose = torch.Tensor(object['pose'])

        embedding = torch.cat((type, layer, polygon, category[0], category[1], category[2], pose))

        assert len(embedding) == dim

        return embedding


    def encode_objects(self, objs, object_tokens, dim):
        embedded_objects = torch.zeros((len(object_tokens), dim))
        for i, token in enumerate(object_tokens):
            if token not in objs.keys():
                continue

            object = objs[token]

            embedded_objects[i] = self.encode_object(object, dim)
        
        return embedded_objects
        

    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['features'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_dur = [item['trans_future_dur'] for item in batch]
        b_trans_future_target = [item['trans_future_target'] for item in batch]

        b_scene_graphs = [item['scene_graphs'] for item in batch]

        batch_size = len(batch)

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self.pad_idx)
        b_trans_future_dur = torch.nn.utils.rnn.pad_sequence(b_trans_future_dur, batch_first=True,
                                                        padding_value=self.pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self.pad_idx)

        #batch scene graphs by timestep
        max_time = max(len(series) for series in b_scene_graphs)
        batched_data = []
        for t in range(max_time):
            graphs_at_t = [series[t] if t < len(series) else None for series in b_scene_graphs]
            graphs_at_t = [g for g in graphs_at_t if g is not None]
            if graphs_at_t:
                batched_data.append(Batch.from_data_list(graphs_at_t))
            else:
                batched_data.append(None)
        b_scene_graphs = batched_data

        batch = [b_features, b_scene_graphs, b_past_label, b_trans_future_dur, b_trans_future_target]

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