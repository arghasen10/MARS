import pickle
from sklearn.model_selection import train_test_split
import h5py
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

NUMBER_OF_POINTS = 1024
NUMBER_OF_FRAMES = 8
np.random.seed(42)

file = open("./nvGesture-point-cloud.pkl", 'rb')
data = pickle.load(file)

def down_sample_frames(data):
    for index, gesture in enumerate(data):
        normalized_gesture = []
        for f_index, frame in enumerate(data[index]):
            if len(frame) <= 0:
                continue
            frame = np.array(frame)
            if len(frame) == NUMBER_OF_POINTS:
                normalized_gesture.append(frame)
            else:
                point_indices = np.random.choice(len(frame), NUMBER_OF_POINTS, replace=len(frame) < NUMBER_OF_POINTS)
                normalized_gesture.append(frame[point_indices])
        data[index] = np.array(normalized_gesture)
        # while len(data[index]) > NUMBER_OF_FRAMES:
        #     frames_to_remove = []
        #     for i in range(int(len(data[index]) / 2)):
        #         frames_to_remove.append(2 * i + 1)
        #         if len(data[index]) - len(frames_to_remove) == NUMBER_OF_FRAMES:
        #             break
        #     data[index] = np.delete(data[index], frames_to_remove, axis=0)
    return data

data['train'] = down_sample_frames(data['train'])
data['test'] = down_sample_frames(data['test'])

train_indices_to_delete = [index for index, gesture in enumerate(data['train']) if len(gesture) != NUMBER_OF_FRAMES]
test_indices_to_delete = [index for index, gesture in enumerate(data['test']) if len(gesture) != NUMBER_OF_FRAMES]

data['train'] = np.delete(data['train'], train_indices_to_delete, axis=0)
data['train_label'] = np.delete(data['train_label'], train_indices_to_delete, axis=0)

data['test'] = np.delete(data['test'], test_indices_to_delete, axis=0)
data['test_label'] = np.delete(data['test_label'], test_indices_to_delete, axis=0)

data['train_label'] = data['train_label'].reshape(-1, 1)
data['test_label'] = data['test_label'].reshape(-1, 1)

with h5py.File('nv_gesture_gesturenet_ply_hdf5/ply_data_train0.h5', 'w') as f:
    f.create_dataset("data", data=data['train'].tolist())
    f.create_dataset("label", data=data['train_label'].tolist())

with h5py.File('nv_gesture_gesturenet_ply_hdf5/ply_data_test0.h5', 'w') as f:
    f.create_dataset("data", data=data['test'].tolist())
    f.create_dataset("label", data=data['test_label'].tolist())


# f = h5py.File('/run/user/1001/gvfs/sftp:host=triton.aalto.fi/home/salamid1/pointnet2/data/modelnet40_ply_hdf5_2048/ply_data_train0.h5')
# print(f['label'][200])

f = h5py.File('nv_gesture_gesturenet_ply_hdf5/ply_data_train0.h5')

print(f['data'].shape)
print(f['label'].shape)

def animate(frame_index, ax, frames):
    ax.clear()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.scatter3D(frames[frame_index, :, 0],
                 frames[frame_index, :, 1],
                 frames[frame_index, :, 2],
                 c='blue')

for gesture_class in np.unique(f['label']):
    for index, gesture in enumerate(f['data']):
        if f['label'][index] == gesture_class:
            print(f['label'][index])
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ani = animation.FuncAnimation(fig, animate, len(gesture), fargs=(ax, gesture), interval=500, blit=False)
            plt.show()
    