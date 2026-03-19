import numpy as np
from src.core.quaternion import Quaternion


class AttitudeEKF:
    """확장 칼만 필터 (Extended Kalman Filter)"""

    def __init__(self, q0: Quaternion, P0, Q, R, dt, m_ref):
        self.q = q0
        self.P = np.array(P0)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.dt = dt
        self.m_ref = np.array(m_ref)

    def predict(self, gyro_data):
        """[예측 단계] 자이로스코프로 다음 상태 예측"""
        wx, wy, wz = gyro_data
        gyro_vec = np.array([wx, wy, wz])
        w_norm = np.linalg.norm(gyro_vec)

        q_gyro = Quaternion(1.0, 0.0, 0.0, 0.0) if w_norm == 0 else Quaternion.from_angle(w_norm * self.dt, gyro_vec)

        # 상태 예측(x=f(x, u, 0))
        self.q = q_gyro * self.q

        # 공분산 예측(P=APA^T+WQW^T)
        F = q_gyro.matrix_left()
        self.P = F @ self.P @ F.T + self.Q

        return self.q

    def update(self, mag_data):
        """[보정 단계] 비선형 측정 모델의 편미분(Jacobian)을 통한 보정"""
        z = np.array(mag_data)

        # 1. 예상 측정값 h(x) 계산: 기준 자기장을 현재 자세로 회전
        m_ref_quat = Quaternion(0.0, self.m_ref[0], self.m_ref[1], self.m_ref[2])
        h_q_quat = self.q * m_ref_quat * self.q.conjugate()
        h_q = h_q_quat.vector()

        # 2. 측정 행렬 자코비안 H 계산 (편미분 행렬)
        q0, q1, q2, q3 = self.q.q
        m0, m1, m2 = self.m_ref

        H = np.array([
            [2 * (q0 * m0 + q2 * m2 - q3 * m1), 2 * (q1 * m0 + q2 * m1 + q3 * m2), 2 * (-q2 * m0 + q1 * m2 - q0 * m1),
             2 * (-q3 * m0 - q0 * m2 + q1 * m1)],
            [2 * (-q3 * m2 + q0 * m1 + q1 * m0), 2 * (q0 * m0 - q1 * m2 + q2 * m1), 2 * (q3 * m0 + q2 * m2 + q1 * m1),
             2 * (-q2 * m0 - q1 * m1 - q0 * m2)],
            [2 * (q2 * m1 - q3 * m0 + q0 * m2), 2 * (q3 * m0 - q0 * m2 - q1 * m1), 2 * (q0 * m0 + q1 * m2 + q2 * m1),
             2 * (-q1 * m0 + q2 * m2 - q3 * m1)]
        ])

        # 3. 칼만 이득 K 계산
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 4. 상태 및 오차 공분산 보정
        q_updated_vec = self.q.q + K @ (z - h_q)
        self.q = Quaternion.from_list(q_updated_vec).normalize()
        self.P = (np.eye(4) - K @ H) @ self.P

        return self.q