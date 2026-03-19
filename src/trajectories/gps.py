import numpy as np
import matplotlib.pyplot as plt

class GPSTrajectory:
    def __init__(self, lat_data, lon_data, alt_data, start_idx = 0):
        self.lat = lat_data
        self.lon = lon_data
        self.alt = alt_data
        self.lat0 = self.lat[start_idx]
        self.lon0 = self.lon[start_idx]

        self.pos = np.zeros((len(lat_data), 3))

    def calculate_trajectory(self, radius=6378137.0):
        """
        Haversine 공식으로 계산
        :param radius:
        :return:
        """
        dlat = self.lat - self.lat0
        dlon = self.lon - self.lon0

        # Haversine 거리 계산
        a = np.sin(dlat / 2) ** 2 + np.cos(self.lat0) * np.cos(self.lat) * np.sin(dlon / 2) ** 2
        distance = 2 * radius * np.arcsin(np.sqrt(a))

        # 방향각도 고려, X와 Y를 분해
        denom = np.sqrt(dlat**2 + dlon**2)
        safe_denom = np.where(denom == 0, 1e-8, denom)

        x = distance * (dlon / safe_denom)
        y = distance * (dlat / safe_denom)

        self.pos[:, 0] = x
        self.pos[:, 1] = y
        self.pos[:, 2] = self.alt

    def plot_trajectory(self, ax=None, color='g', label='GPS'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")

        ax.plot(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2], color=color, label=label)