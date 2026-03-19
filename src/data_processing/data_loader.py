import os
import pandas as pd
import numpy as np
from src.core.quaternion import Quaternion

class FlightDataLoader:
    """
    로켓 비행 데이터를 읽어와 분석합니다.
    """
    def __init__(self, file_path, dt=0.1):
        self.file_path = os.path.abspath(file_path)
        self.dt = dt
        self.indices = {}

        self._load_data()
        self._find_flight_phases()
        self._set_initial_conditions()

    def _load_data(self):
        df = pd.read_csv(self.file_path)

        self.acc = np.column_stack([
            -df['xAxisAcc'].to_numpy(),
            -df['yAxisAcc'].to_numpy(),
            -df['zAxisAcc'].to_numpy()
        ])

        self.gyro = np.column_stack([
            df['xAxisAngVal'].to_numpy(),
            df['yAxisAngVal'].to_numpy(),
            df['zAxisAngVal'].to_numpy()
        ]) * (np.pi / 180.0)

        self.mag = np.column_stack([
            df['xAxisMagF'].to_numpy(),
            df['yAxisMagF'].to_numpy(),
            df['zAxisMagF'].to_numpy()
        ])

        self.gps_lat = df['gpsLatitude'].to_numpy() * 1e-7
        self.gps_lon = df['gpsLongitude'].to_numpy() * 1e-7

        pressure = df['pressure']
        self.altitude = 44307.69396 * (1.0 - np.power(pressure / 101325.0, 0.190284))
        self.ground_altitude = np.min(self.altitude)
        self.altitude -= self.ground_altitude

        self.length = len(df)
        self.time = df['deviceTime'].to_numpy() / 1000.0 / self.dt

    def _find_flight_phases(self):
        self.indices['stand_by'] = 0

        g0_vec = self.acc[self.indices['stand_by']]
        self.g0 = np.linalg.norm(g0_vec)

        acc_norms = np.linalg.norm(self.acc, axis=1)

        # 발사 순간 탐색: 가속도가 초기 중력 대비 5% 이상 변하는 시점
        launch_idx = 0
        while launch_idx < self.length and abs(self.g0 - acc_norms[launch_idx]) < self.g0 * 0.05:
            launch_idx += 1
        self.indices['launch'] = launch_idx

        # 착지 순간 탐색: 맨 뒤부터 가속도가 안정되는 시점
        touchdown_idx = self.length - 1
        while touchdown_idx > launch_idx and abs(self.g0 - acc_norms[touchdown_idx]) < self.g0 * 0.05:
            touchdown_idx -= 1
        self.indices['touchdown'] = touchdown_idx

    def _set_initial_conditions(self):
        """
        EKF 알고리즘에 넘겨줄 초기 텐서 설정
        :return:
        """
        start = self.indices['stand_by']
        end = self.indices['launch']
        if start == end: end = start + 1

        # 초기 자세 계산
        gg0 = np.mean(self.acc[start:end], axis=0)
        ref_g = np.array([0.0, 0.0, self.g0])
        self.q0 = Quaternion.from_two_vectors(gg0, ref_g)

        # 기준 자기장 계산(북쪽 벡터 생성)
        m0_mean = np.mean(self.mag[start:end], axis=0)
        m0_quat = Quaternion(0.0, m0_mean[0], m0_mean[1], m0_mean[2])
        m_ref_quat = self.q0 * m0_quat * self.q0.conjugate()

        self.m_ref = m_ref_quat.vector()
        self.m_ref_4d = m_ref_quat.q

        # 칼만 필터 공분산 행렬
        self.P0 = np.eye(4) * 0.1
        gyro_std = np.std(self.gyro[start:end], axis=0) * self.dt
        self.Q = np.diag([0.0, gyro_std[0]**2, gyro_std[1]**2, gyro_std[2]**2])
        mag_std = np.std(self.mag[start:end], axis=0)
        self.R = np.diag([mag_std[0]**2, mag_std[1]**2, mag_std[2]**2])
        self.R_4x4 = np.diag([0.0, mag_std[0]**2, mag_std[1]**2, mag_std[2]**2])