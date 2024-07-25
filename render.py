from dataset_utils import *
from main_nusc import data_path
nusc_data = NuScenesSimple(data_path, [str(i) for i in range(128)])

print('loaded, running')

plt.rcParams['font.size'] = 24

plt.ioff()
for i in range(128):
    for j in range(0, 35, 5):
        frame = select_frame(nusc_data[i], j)
        fig = render_frame(frame)
        fig.savefig('test_frames/test_frame_%d_%d.png' % (i, j) )

'''
print('trajectory')
render_trajectory('test_traj.gif', nusc_data[0])
print('rendered')
'''