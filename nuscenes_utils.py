import matplotlib.pyplot as plt
import numpy as np
import imageio

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

dataroot='/data/Datasets/nuscenes'

nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
nusc_map_so = NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth')
nusc_map_sh = NuScenesMap(dataroot=dataroot, map_name='singapore-hollandvillage')
nusc_map_sq = NuScenesMap(dataroot=dataroot, map_name='singapore-queenstown')
nusc_map_bs = NuScenesMap(dataroot=dataroot, map_name='boston-seaport')
nusc_maps = [nusc_map_so, nusc_map_sh, nusc_map_sq, nusc_map_bs]
nusc_can = NuScenesCanBus(dataroot=dataroot)

class Scene:

    def __init__(self, scene_name):
        #settings
        scene = [ i for i in nusc.scene if scene_name in i['name'].lower() ][0]
        log_token = scene['log_token']
        log_record = nusc.get('log', log_token)
        map_location = log_record['location']
        maps = {'singapore-onenorth':nusc_map_so, 'singapore-hollandvillage':nusc_map_sh, 'singapore-queenstown':nusc_map_sq, 'boston-seaport':nusc_map_bs}

        self.scene_name = scene_name
        self.scene_token = scene['token']
        self.map = maps[map_location]
        self.data = None
        self.actions = []

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

    '''
    METHODS======================================================================
    '''

    '''
    extracts keyframes with their features from the core dataset
    get the datapoints that best match in time
    should be called before extracting other data
    '''
    def extract_core_data(self):
        scene = nusc.get('scene', self.scene_token)
        first_sample_token = scene['first_sample_token']
        
        sample = nusc.get('sample', first_sample_token)
        self.data = []

        while True:
            # no data for now, just token to query annotations table with later
            self.data.append({'token':sample['token'], 'utime':sample['timestamp']})

            next_token = sample['next']
            if next_token == '':
                break
            sample = nusc.get('sample', next_token)
    
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
    extracts all data we want from all nuscenes expansions
    '''
    def extract_data(self, map=False):
        self.extract_core_data()
        self.add_CAN_data()
        if map:
            self.add_map_data()
        self.convert_utime_secs()

    def segment(self, straight_thresh=0.7, coast_thresh=0.1, stop_thresh=0.05):
        self.actions = []

        last_speed = self.data[0]['vel']
        last_time = self.data[0]['time']

        for d in self.data:
            action = 'none'
            speed = d['vel']
            #acceleration = np.linalg.norm(np.array(d['accel'])) #TODO: dot with orientation
            d_speed = speed - last_speed
            steer = d['steer']
            time = d['time']

            if speed < stop_thresh:
                action = 'stop'
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

            if not self.actions or self.actions[len(self.actions)-1]['label'] != action:
                self.actions.append({'label':action, 'time':last_time})

            #update bookkeeping
            last_speed = speed
            last_time = time

        self.actions.append({'label':'END', 'time':self.data[len(self.data)-1]['time']})

    def plot_actions(self, features=None):

        # create action periods
        colors = {'stop':'red', 'drive straight':'blue', 'accelerate':'green', 'decelerate':'yellow', 'turn left':'orange', 'turn right':'magenta'}
        periods = []
        for i in range(len(self.actions)-1):
            from_time = self.actions[i]['time']
            to_time = self.actions[i+1]['time']
            color = colors[self.actions[i]['label']]
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
    def render_actions_map(self, filename, radius=10, layers=nusc_map_so.non_geometric_layers):
        colors = {'stop':'red', 'drive straight':'blue', 'accelerate':'green', 'decelerate':'yellow', 'turn left':'orange', 'turn right':'magenta'}
        frames = []
        j = 0
        for d in self.data:

            #get the current action
            if d['time'] > self.actions[j]['time'] and j < len(self.actions):
                j+=1
            
            action = self.actions[0]
            if j > 0:
                action = self.actions[j-1]

            x = d['pos'][0]
            y = d['pos'][1]
            patch = (x-radius, y-radius, x+radius, y+radius)
            fig, ax = self.map.render_map_patch(patch, layers, figsize=(6, 5))
            ax.scatter(x, y, color=colors[action['label']], marker='o', s=40)
            ax.annotate(action['label'], (x, y))

            #add frame to gif
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)

        imageio.mimsave(filename, frames, duration=0.2)