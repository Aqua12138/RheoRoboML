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

# 对于不同任务需要修改求解范围 self.workArea
# 需要提供目标表征self.get_targetPositions(), unity get
# 如果需要更加精细的处理，请修改pixel_size

class ParticlePositionChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("79eeb5a6-6650-4bc1-9a80-1289b9a224f7"))
        self.targetDistribution = self.get_targetPositions("shape")
        # pour [[-2.0, 2.0], [0.0, 2.0], [-2.0, 2.0]]、
        # shape  [[-0.6, 0.6], [0.4, 0.7], [-0.6, 0.6]]
        # gather [[-1.2, 1.2], [0.4, 0.7], [-1.2, 1.2]]
        self.workArea = [[-0.6, 0.6], [0.4, 0.7], [-0.6, 0.6]]
        # pour [20, 10, 20]
        # shape  [12, 3, 12]
        # gather [24, 3, 24]
        self.pixel_size = [12, 3, 12]
        self.M = self.create_cost_matrix(self.workArea) # 求解最优传输矩阵


    def get_targetPositions(self, task_name):
        # csv_file_path = '/home/zhx/PycharmProjects/ml-agents/ml-agents/mlagents/trainers/targetPositions.csv'
        # df = pd.read_csv(csv_file_path, header=None)
        # particle_positions = df.to_numpy()
        if task_name == "gather":
            b = np.zeros((24, 3, 24))
            # 将中心位置的元素值设为1
            b[7:17, 0, 7:17] = 1
            b[9:15, 1, 9:15] = 1
            b[11:13, 2, 11:13] = 1
            b /= np.sum(b)

            # b = np.zeros((12, 6, 12))
            # b[1:11, 0, 1:11] = 1
            # b[2:10, 1, 2:10] = 1
            # b[3:9, 2, 3:9] = 1
            # b[4:8, 3, 4:8] = 1
            # b[5:7, 4, 5:7] = 1
            # b /= np.sum(b)

        elif task_name == "shape":
            b = np.ones((12, 3, 12))
            b[:, 1:, :] = 0
            b /= np.sum(b)

        elif task_name == "pour":
            b = np.zeros([20, 10, 20])
            b[8:11, 2, 8:11] = 1
            b /= np.sum(b)

        else:
            b = np.ones((12, 3, 12))
            b /= np.sum(b)

        return b

    def on_message_received(self, msg: IncomingMessage) -> None:
        particle_positions = []
        num_particles = msg.read_int32()
        for _ in range(num_particles):
            x = msg.read_float32()
            y = msg.read_float32()
            z = msg.read_float32()
            particle_positions.append(np.array([x, y, z]))
        # 计算HeightMaps
        reward = self.compute_wassersteinDistance(particle_positions)

        # 计算
        # 发送Wasserstein距离回Unity
        msg_out = OutgoingMessage()
        msg_out.write_float32(reward)
        self.queue_message_to_send(msg_out)

    def compute_wassersteinDistance(self, particle_positions):
        # 计算Wasserstein距离
        a, _ = np.histogramdd(np.array(particle_positions), bins=self.pixel_size, range=self.workArea, density=True)
        a /= np.sum(a)
        b = self.targetDistribution
        wasserstein_distance = ot.emd2(a.flatten(), b.flatten(), self.M)
        return wasserstein_distance

    def compute_heightMap_mean(self, particle_positions):
        # 在range范围内生成voxelcell
        voxelcell, _ = np.histogramdd(np.array(particle_positions), bins=20, range = self.workArea)

        # 计算每一列最高点位置（y轴）
        # 反转y轴数据（颗粒位置集中在小索引区域）
        voxelcell = voxelcell[:, ::-1, :]
        # 获取第一个不为0索引
        height_map = np.argmax(voxelcell > 0, axis=1)

        height_map = voxelcell.shape[0] - height_map - 1

        # 计算每个体素的实际高度。注意这里假设y坐标的范围是[0.4, 0.6]，并且有20个体素。
        y_values = np.linspace(0.4, 0.7, 20)

        # 转换体素索引为实际高度
        height_map = y_values[height_map]
        # 计算高程图的平均值
        mean_height = np.mean(height_map)
        # 用每个高度值减去平均值
        height_map -= mean_height
        abs_height_map = np.abs(height_map)
        meanHeight = np.mean(abs_height_map)

        return meanHeight.astype(float)

    def create_cost_matrix(self, range_param):
        # range_param = np.array([[-0.6, 0.6], [0.4, 0.6], [-0.6, 0.6]])

        # Make sure that pixel_size is a numpy array
        if not isinstance(self.pixel_size, np.ndarray):
            self.pixel_size = np.array(self.pixel_size)

        # Calculate the step size in each dimension
        step_sizes = [(r[1] - r[0]) / p for r, p in zip(range_param, self.pixel_size)]

        # Calculate the center of each bin in each dimension
        centers = [np.linspace(r[0] + step / 2, r[1] - step / 2, p) for r, step, p in
                   zip(range_param, step_sizes, self.pixel_size)]

        # Create a grid of center coordinates
        grid = np.meshgrid(*centers, indexing='ij')

        # Flatten and stack the coordinates
        coords = np.column_stack([g.flatten() for g in grid])

        # Calculate the cost matrix
        M = ot.dist(coords, coords)

        return M


if __name__ == '__main__':

    def func1():
        # 测试显示
        P = ParticlePositionChannel()
        # M = P.create_cost_matrix(np.random.randn(1000, 3), np.random.randn(1000, 3))
        c = P.get_targetPositions("gather")
        csv_file_path = '/home/zhx/Unity/ml-plaster/Assets/ManualRecord/frame_3.csv'
        df = pd.read_csv(csv_file_path, header=None)
        particle_positions = df.to_numpy()
        t, _ = np.histogramdd(np.array(particle_positions), bins=P.pixel_size, range=P.workArea, density=True)
        t /= np.sum(t)
        # print(np.sum(c))
        # a是particle_positions在空间中三维分布的histogram，b使用均匀分布

        # Create the x, y, and z coordinate arrays. We use
        # np.indices() to create a 3D grid of coordinates.
        x, y, z = np.indices((np.array(c.shape) + 1))

        # Create a 3D figure
        fig = plt.figure(figsize=(12, 6))

        # Add a 3D subplot for histogram a
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.voxels(x, z, y, c > 0, facecolors='red', edgecolor='k', alpha=0.5)
        ax1.set_title("Histogram of particle positions")

        # Add a 3D subplot for histogram b
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.voxels(x, z, y, t > 0, facecolors='blue', edgecolor='k', alpha=0.5)
        ax2.set_title("Uniform distribution histogram")

        plt.show()


    def func2():
        # 测试wasserstein距离计算是否计算正确
        P = ParticlePositionChannel()
        # M = P.create_cost_matrix(np.random.randn(1000, 3), np.random.randn(1000, 3))
        csv_file_path = '/home/zhx/Unity/ml-plaster/Assets/ManualRecord/frame_2.csv'
        df = pd.read_csv(csv_file_path, header=None)
        particle_positions = df.to_numpy()

        print(P.compute_wassersteinDistance(particle_positions))

    def func3():
        plt.figure()
        P = ParticlePositionChannel()
        csv_foder = "/home/zhx/Unity/ml-plaster/Assets/ManualRecord/" # particle_0.csv
        wassersteinDistances = []
        for i in range(400):
            df = pd.read_csv(csv_foder + "frame_" + str(i+10) + ".csv", header=None)
            particle_positions = df.to_numpy()
            wassersteinDistances.append(P.compute_wassersteinDistance(particle_positions))
        plt.plot(wassersteinDistances)
        plt.title('Wasserstein Distances over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Wasserstein Distance')
        plt.show()
    func3()
