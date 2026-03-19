import numpy as np
import matplotlib.pyplot as plt
from src.core.quaternion import Quaternion

class IMUTrajectory:
    def __init__(self, rotators, acc_data, dt, gravity):
        self.rotators = rotators
        self.acc = acc_data
        self.dt = dt
        self.gravity = gravity
        self.pos = np.array([0.0, 0.0, 0.0])

    def get_observer_acc(self, index):
        """
        해당 시점의 로켓 가속도를 절대 좌표계 기준으로 변환
        :param index:
        :return:
        """
        q = self.rotators[index]
        a_body = self.acc[index]

        a_body_quat = Quaternion(0.0, a_body[0], a_body[1], a_body[2])
        a_obs_quat = q.conjugate() * a_body_quat * q

        return a_obs_quat.vector() - np.array([0.0, 0.0, -self.gravity])

    def calculate_trajectory(self, launch_idx, touchdown_idx):
        vel = np.array([0.0, 0.0, 0.0])
        p = np.array([0.0, 0.0, 0.0])
        positions = [p.copy()]

        for i in range(launch_idx, touchdown_idx):
            # RK4
            a1 = self.get_observer_acc(i)
            a2 = self.get_observer_acc(min(i+1, len(self.acc) - 1))
            a_mid = (a1 + a2) / 2.0

            k1_v = a1
            k2_v = a_mid
            k3_v = a_mid
            k4_v = a2

            k1_p = vel
            k2_p = vel + 0.5 * k1_v * self.dt
            k3_p = vel + 0.5 * k2_v * self.dt
            k4_p = vel + k3_v * self.dt

            vel = vel + (self.dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            p = p + (self.dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

            positions.append(p.copy())

        self.pos = np.array(positions)

    def plot_trajectory(self, ax=None, color='r', label='IMU (EKF)'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")

        ax.plot(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2], color=color, label=label)