import matplotlib.pyplot as plt
import numpy as np
import imageio
import copy
from pyquaternion import Quaternion

from nuscenes.eval.common.utils import quaternion_yaw, angle_diff
from nuscenes.map_expansion import arcline_path_utils


from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

dataroot='/data/Datasets/nuscenes'

colors = {'stop':'red', 'back':'white', 'drive straight':'blue', 'accelerate':'green', 'decelerate':'yellow', 'turn left':'orange', 'turn right':'magenta', 'uturn':'c', 'change lane left':'Salmon', 'change lane right':'Salmon', 'overtake':'aquamarine'} 

#nusc_mini = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
#nusc_trainval = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
nusc_map_so = NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth')
nusc_map_sh = NuScenesMap(dataroot=dataroot, map_name='singapore-hollandvillage')
nusc_map_sq = NuScenesMap(dataroot=dataroot, map_name='singapore-queenstown')
nusc_map_bs = NuScenesMap(dataroot=dataroot, map_name='boston-seaport')
nusc_maps = [nusc_map_so, nusc_map_sh, nusc_map_sq, nusc_map_bs]
nusc_can = NuScenesCanBus(dataroot=dataroot)

class Scene:

    def __init__(self, nusc, scene_name):
        #settings
        scene = [ i for i in nusc.scene if scene_name in i['name'].lower() ][0]
        log_token = scene['log_token']
        log_record = nusc.get('log', log_token)
        map_location = log_record['location']
        maps = {'singapore-onenorth':nusc_map_so, 'singapore-hollandvillage':nusc_map_sh, 'singapore-queenstown':nusc_map_sq, 'boston-seaport':nusc_map_bs}

        self.nusc = nusc
        self.scene_name = scene_name
        self.scene_token = scene['token']
        self.map = maps[map_location]
        self.data = None
        self.rich_actions = []
        self.prim_actions = []

    '''
    HELPERS======================================================================
    '''
    def convert_utime_secs(self):
        #convert to seconds
        start_utime = self.data[0]['utime']
        for d in self.data:
            d['time'] = (d['utime'] - start_utime) * 1e-6
            del d['utime']

    def query_map(self, x, y, radius, layers):
        map_records = self.map.get_records_in_radius(x, y, radius, layers, mode='intersect')
        return map_records
    
    def last_action_label(self, i, actions):
        last_actions = [a for a in actions if a['index'] <= i]
        if not last_actions:
            return 'none'
        return last_actions[-1]['label']
    
    @staticmethod
    def flatten_actions(actions):
        flat_actions = [None] * actions[-1]['index']

        for i in range(len(actions)-1):
            this = actions[i]
            next = actions[i+1]
            idx1 = this['index']
            idx2 = next['index']
            flat_actions[idx1:idx2] = [this['label'] for j in range(idx2-idx1)]
        
        flat_actions.append('END')
        return flat_actions

    '''
    METHODS======================================================================
    '''

    '''
    extracts keyframes with their features from the core dataset
    get the datapoints that best match in time
    should be called before extracting other data
    '''
    def extract_core_data(self):
        scene = self.nusc.get('scene', self.scene_token)
        first_sample_token = scene['first_sample_token']
        
        sample = self.nusc.get('sample', first_sample_token)
        self.data = []

        while True:
            # no data for now, just token to query annotations table with later
            self.data.append({'token':sample['token'], 'utime':sample['timestamp']})

            next_token = sample['next']
            if next_token == '':
                break
            sample = self.nusc.get('sample', next_token)
    
    '''
    given the utimes of the keyframes, get the closest matching CAN data from each channel and add it to the data
    also get the map records in the surrounding area
    '''
    def add_CAN_data(self):
        pose_data = nusc_can.get_messages(self.scene_name, 'pose')
        steer_data = nusc_can.get_messages(self.scene_name, 'steeranglefeedback')
        i = 0
        j = 0
        for k,d in enumerate(self.data):
            while pose_data[i]['utime'] < d['utime']:
                if i == len(pose_data)-1:
                    break
                i+=1
            while steer_data[j]['utime'] < d['utime']:
                if j == len(steer_data)-1:
                    break
                j+=1

            # process the data before adding to core
            del pose_data[i]['utime']
            del steer_data[j]['utime']
            pose_data[i]['vel'] = pose_data[i]['vel'][0]
            steer_data[j]['steer'] = steer_data[j].pop('value')

            self.data[k].update(pose_data[i])
            self.data[k].update(steer_data[j])

            i+=1
            j+=1
    
    '''
    adds map data to a data list
    '''
    def add_map_data(self, radius, layers):
        for d in self.data:
            x = d['pos'][0]
            y = d['pos'][1]
            d['map'] = self.query_map(x, y, radius, layers)

    '''
    adds closest lane 
    '''
    def add_lane_data(self, radius):
        for d in self.data:
            x = d['pos'][0]
            y = d['pos'][1]

            o = Quaternion(d['orientation'])
            yaw = quaternion_yaw(o)
            closest_lane = self.map.get_closest_lane(x, y, radius=radius)
            lane_record = self.map.get_arcline_path(closest_lane)
            closest_pose_on_lane, distance_along_lane = arcline_path_utils.project_pose_to_lane((x, y, yaw), lane_record)
            lane_point = np.array(closest_pose_on_lane[:2])
            car_point = np.array([x,y])
            dist = np.linalg.norm(lane_point - car_point)

            d['dist_centerline'] = dist
            d['closest_lane'] = closest_lane

    def add_ann_data(self):
        for d in self.data:
            sample = self.nusc.get('sample', d['token'])
            d['anns'] = []
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                x = ann['translation'][0]
                y = ann['translation'][1]
                category = ann['category_name']
                d['anns'].append({'token':ann_token, 'category':category ,'pos':(x,y)})

    '''
    extracts all data we want from all nuscenes expansions
    '''
    def extract_data(self, map=False, radius=10):
        self.extract_core_data()
        self.add_CAN_data()
        self.add_ann_data()
        if map:
            self.add_lane_data(radius)
            #self.add_map_data(radius, self.map.non_geometric_layers)
        self.convert_utime_secs()

    def segment_actions(self, straight_thresh=0.7, coast_thresh=0.1, stop_thresh=0.05, uturn_thresh=0.3, uturn_radius=15, centerline_thresh=1, primitive=False):
        actions = []


        #frame by frame primitive labeling
        for i, d in enumerate(self.data[:len(self.data)-1]):
            next_speed = self.data[i+1]['vel']
            action = 'none'
            speed = d['vel']
            #acceleration = np.linalg.norm(np.array(d['accel'])) #TODO: dot with orientation
            d_speed = next_speed - speed
            steer = d['steer']
            time = d['time']

            if abs(speed) < stop_thresh:
                action = 'stop'
            elif speed < 0:
                action = 'back'
            else:
                if abs(steer) < straight_thresh:
                    if abs(d_speed) < coast_thresh:
                        action = 'drive straight'
                    elif d_speed > 0:
                        action = 'accelerate'
                    elif d_speed < 0:
                        action = 'decelerate'
                elif steer > 0:
                    action = 'turn left'
                elif steer < 0:
                    action = 'turn right'

            if not actions or actions[len(actions)-1]['label'] != action:
                actions.append({'label':action, 'index':i, 'time':time})

        actions.append({'label':'END', 'index':len(self.data)-1, 'time':self.data[len(self.data)-1]['time']})

        self.prim_actions = copy.deepcopy(actions)

        if primitive:
            return
        
        #check for rich actions
        rich_actions = copy.deepcopy(actions)
        #uturns
        i = 0
        while i < len(actions)-1:
            start = actions[i]
            #if self.is_turn(start):
            if 'turn' in start['label']:
                for k, last in enumerate(actions[i:len(actions)-1]):
                    #last is the last action within the two endpoint timestamps
                    #end is the action starting at the latter endpoint timestamp
                    end = actions[i+k+1]
                    #check that start and last are both turns in the same direction
                    if last['label'] == start['label']:
                        #check for 180 direction difference
                        yaw1 = quaternion_yaw(Quaternion(self.data[start['index']]['orientation']))
                        yaw2 = quaternion_yaw(Quaternion(self.data[end['index']]['orientation']))
                        diff = abs(angle_diff(yaw1, yaw2, np.pi*2))

                        if abs(diff - np.pi) < uturn_thresh:
                            #loop through all keyframes in between to make sure car didn't move too much

                            valid = True
                            start_pos = np.array(self.data[start['index']]['pos'][:2])
                            for d in self.data[start['index']:end['index']]:
                                pos = np.array(d['pos'][:2])
                                if np.linalg.norm(start_pos - pos) > uturn_radius:
                                    valid = False
                                    break

                            if valid:
                                rich_actions[i]['label'] = 'uturn'
                                #print("UTURN:", self.scene_name)
                                d_i = 0
                                while rich_actions[i+1] != end:
                                    d_i+=1
                                    del rich_actions[i+1]
                                i+=d_i
                                break
                #if breaker:
                #    break
            i+=1
        
        #lane changes
        i = 0
        while i < len(self.data)-1:
            d = self.data[i]
            next = self.data[i+1]
            last_action = self.last_action_label(i, rich_actions)
            #print(last_action)
            if d['closest_lane'] != next['closest_lane'] and last_action != 'uturn':
                #print('found cross point')
                #found cross point, check that distances to each lane is less than width thresh
                if d['dist_centerline'] > centerline_thresh and next['dist_centerline'] > centerline_thresh:
                    #propogate in each direction
                    j = i+1
                    k = i

                    while j < len(self.data)-1 and self.data[j+1]['dist_centerline'] < self.data[j]['dist_centerline']:
                        j+=1
                    while k > 0 and self.data[k-1]['dist_centerline'] < self.data[k]['dist_centerline']:
                        k-=1
                    #print("LANE_CHANGE\n", "index: [", k, j, "]", "time: [", self.data[k]['time'], self.data[j]['time'], "]")

                    start_direction = quaternion_yaw(Quaternion(self.data[k]['orientation']))
                    mid_direction = quaternion_yaw(Quaternion(self.data[(k+j)//2]['orientation']))
                    if angle_diff(mid_direction, start_direction, np.pi*2) > 0:
                        direction = 'left'
                    else:
                        direction = 'right'

                    #add lane change to actions
                    right_posted = False
                    left_posted = False
                    for l in reversed(range(len(rich_actions)-1)): # consider length of range
                        if rich_actions[l]['index'] <= k and not left_posted:
                            left_posted = True
                            left_action = {'label':'change lane %s' % direction, 'index':k, 'time':self.data[k]['time']}
                            if rich_actions[l]['index'] == k:
                                rich_actions[l] = left_action
                            else:
                                rich_actions.insert(l+1, left_action)
                            continue
                        if rich_actions[l]['index'] <= j and not right_posted:
                            right_posted = True
                            right_action = {'label':rich_actions[l]['label'], 'index':j, 'time':self.data[j]['time']}
                            rich_actions[l] = right_action
                            continue
                        if right_posted and not left_posted:
                            del rich_actions[l]
                        
                    '''
                    c = test_list.count(item) 
                    for i in range(c): 
                        test_list.remove(item) 
                    '''
                    i = j-1
            i+=1
        
        #overtakes
        '''
        lane_changes = [i for i in rich_actions if i['label'] == 'change lane']
        for i,action in enumerate(lane_changes[:len(lane_changes)-1]): 
            #check if the two lane changes are in opposite directions
            if action['label'] != lane_changes[i+1]['label']:
                print('OVERTAKE: ', action['label'], lane_changes[i+1])
                action['label'] = "overtake"
                del lane_changes[i+1]
        '''

        i = 0
        while i < len(rich_actions)-1:
            #check if the two lane changes are in opposite directions
            action = rich_actions[i]
            next_action = rich_actions[i+1]
            if 'change lane' in action['label'] and 'change lane' in next_action['label']:
                if action['label'] != next_action['label']:
                    #print('OVERTAKE: ', action['label'], action['index'], next_action['label'], next_action['index'])
                    action['label'] = "overtake"
                    del rich_actions[i+1]
                    i-=1
            i+=1

                
                

        self.rich_actions = rich_actions

    def output_data(self):
        print('output_data')
        out = []
        flat_actions = self.flatten_actions(self.rich_actions)
        for i,d in enumerate(self.data):
            out.append({ 'map':self.map, 'scene':self.scene_token, 'pos':d['pos'][:2], 'anns':d['anns'], 'action':flat_actions[i] })
        return out

    def plot_actions(self, features=None, primitive=False):

        actions = self.rich_actions
        if primitive:
            actions = self.prim_actions
        
        # create action periods
        periods = []
        for i in range(len(actions)-1):
            from_time = actions[i]['time']
            to_time = actions[i+1]['time']
            color = colors[actions[i]['label']]
            periods.append((from_time, to_time, color))
        print(periods)

        xlabel = 'time'

        # choose feature keys

        #all
        if features is None:
            feature_keys = list(set().union(*(d.keys() for d in self.data)))
            feature_keys.remove(xlabel)  # Remove 'time' key
            feature_keys.remove('token')  # Remove 'token' key
            if 'map' in feature_keys:
                feature_keys.remove('map')  # Remove 'token' key
            if 'closest_lane' in feature_keys:
                feature_keys.remove('closest_lane')  # Remove 'token' key
            if 'anns' in feature_keys:
                feature_keys.remove('anns')  # Remove 'token' key

        # Create a figure with subplots
        num_subplots = len(feature_keys)
        fig, axs = plt.subplots(num_subplots, 1, figsize=(8, 3 * num_subplots), sharex=True)

        # Plot each feature in a separate subplot
        for i, feature in enumerate(feature_keys):
            x = [d[xlabel] for d in self.data]
            feature_values = [d[feature] for d in self.data]
            
            if num_subplots == 1:
                ax = axs
            else:
                ax = axs[i]

            for start, end, color in periods:
                ax.axvspan(start, end, alpha=0.3, color=color)

            ax.plot(x, feature_values)
            ax.set_ylabel(feature)

        # Set x-axis label and title
        axs[-1].set_xlabel(xlabel)
        fig.suptitle('Feature Plots')

        plt.tight_layout()
        plt.show()

    '''
    show the ego poses over time, highlight according to color
    render into a gif on th map
    data list must contain map field
    '''
    def render_actions_map(self, filename, radius=10, layers=nusc_map_so.non_geometric_layers, primitive=False):

        #TODO: check for adverse effects of fixing indexing
        actions = self.rich_actions
        if primitive:
            actions = self.prim_actions
        
        frames = []
        j = 0
        for d in self.data:

            #get the current action
            if d['time'] > actions[j]['time'] and j < len(actions):
                j+=1
            
            action = actions[0]
            if j > 0:
                action = actions[j-1]

            x = d['pos'][0]
            y = d['pos'][1]
            patch = (x-radius, y-radius, x+radius, y+radius)
            fig, ax = self.map.render_map_patch(patch, layers, figsize=(6, 5))
            ax.scatter(x, y, color=colors[action['label']], marker='o', s=40)
            ax.annotate(action['label'], (x, y))

            #render other objects
            for ann in d['anns']:
                x = ann['pos'][0]
                y = ann['pos'][1]
                category = ann['category']
                ax.scatter(x, y, color='m', marker='o', s=40)
                ax.annotate(category, (x, y))

            #add frame to gif
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)

        imageio.mimsave(filename, frames, duration=0.2)
