from nuscenes_utils import *

scene_num = 43
s = Scene('scene-%s' % str(scene_num).zfill(4))
s.extract_data(map=True)
s.segment_actions()

print(s.rich_actions)
print(s.prim_actions)

s.plot_actions()
#print(s.data)