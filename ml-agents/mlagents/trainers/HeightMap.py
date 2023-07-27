import sys
sys.path.insert(0,'/home/zhx/anaconda3/envs/mlagents/lib/python3.8/site-packages')
import torch  # noqa I201
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
import uuid
import ot
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 转换成voxel
csv_file_path = '/home/zhx/PycharmProjects/ml-agents/ml-agents/mlagents/trainers/particle_positions5.csv'
df = pd.read_csv(csv_file_path, header=None)
particle_positions = df.to_numpy()
# Create histograms for the particle_positions and uniform distribution
a, _ = np.histogramdd(particle_positions, bins=20, range=[[-0.6, 0.6], [0.4, 0.6], [-0.6, 0.6]])
b = np.ones((20, 20, 20))
b[:, 8:, :] = 0
# Normalize the histogram to represent probabilities

# 找到每个x-z平面列的最高点（最大y值）
a = a[:, ::-1, :]
height_map = np.argmax(a > 0, axis=1)

height_map = a.shape[0] - height_map - 1

# 计算每个体素的实际高度。注意这里假设y坐标的范围是[0.4, 0.6]，并且有20个体素。
y_values = np.linspace(0.4, 0.6, 20)

# 转换体素索引为实际高度
height_map = y_values[height_map]
# 计算高程图的平均值
mean_height = np.mean(height_map)

# 用每个高度值减去平均值
height_map -= mean_height
abs_height_map = np.abs(height_map)
reward = np.mean(abs_height_map)
print(reward)

# 计算
import matplotlib.pyplot as plt

# 使用imshow()函数绘制高程图
plt.imshow(height_map, cmap='jet', origin='lower')
plt.colorbar(label='Height')
plt.title('Height map')
plt.xlabel('x')
plt.ylabel('z')
plt.show()
print(np.sum(a))
# a是particle_positions在空间中三维分布的histogram，b使用均匀分布

# Create the x, y, and z coordinate arrays. We use
# np.indices() to create a 3D grid of coordinates.
x, y, z = np.indices((np.array(a.shape) + 1))

# Create a 3D figure
fig = plt.figure(figsize=(12, 6))

# Add a 3D subplot for histogram a
ax1 = fig.add_subplot(121, projection='3d')
ax1.voxels(x, z, y, a > 0, facecolors='red', edgecolor='k', alpha=0.5)
ax1.set_title("Histogram of particle positions")

# Add a 3D subplot for histogram b
ax2 = fig.add_subplot(122, projection='3d')
ax2.voxels(x, z, y, b > 0, facecolors='blue', edgecolor='k', alpha=0.5)
ax2.set_title("Uniform distribution histogram")

plt.show()
