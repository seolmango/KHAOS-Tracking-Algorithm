import numpy as np
from .imu import IMUTrajectory
from src.core.quaternion import Quaternion


class SIMPLETrajectory(IMUTrajectory):
    def __init__(self, acc_data, gyro_data, dt, gravity, q0, launch_idx, touchdown_idx):
        self.gyro = gyro_data
        self.dt = dt
        self.launch_idx = launch_idx
        self.touchdown_idx = touchdown_idx

        rotators = self._generate_rotators(q0)

        super().__init__(rotators, acc_data, dt, gravity)

    def _generate_rotators(self, q0):
        rotators = [q0] * len(self.gyro)
        current_q = q0

        for i in range(self.launch_idx, self.touchdown_idx):
            w = self.gyro[i]
            w_norm = np.linalg.norm(w)

            if w_norm > 0:
                delta_q = Quaternion.from_angle(w_norm * self.dt, w)
                current_q = delta_q * current_q

            rotators[i + 1] = current_q

        return rotators

    def calculate_trajectory(self):
        super().calculate_trajectory(self.launch_idx, self.touchdown_idx)