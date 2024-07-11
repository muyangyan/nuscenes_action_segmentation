from dataset_utils import *
nusc_data = NuScenesCustom('./data', ['0','1','2'])

print('loaded, running')

plt.ioff()
frame = select_frame(nusc_data[0], 0)
print('selected')
fig = render_frame(frame)
print('rendered')
fig.savefig('test_frame.png')

print('trajectory')
render_trajectory('test_traj.gif', nusc_data[0])
print('rendered')