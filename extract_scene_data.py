from nuscenes_utils import *

custom_blacklist = range(481, 518)

scene_names = []
for i in range(20):
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

print("==============================")        
print(scene_data)



    
