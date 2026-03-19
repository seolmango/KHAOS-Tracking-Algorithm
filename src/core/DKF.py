import numpy as np
from src.core.quaternion import Quaternion

def _quat_from_vec(v, v_prime):
    """두 3D 벡터 사이의 회전 쿼터니언을 반환"""
    v = np.array(v, dtype=float)
    v_prime = np.array(v_prime, dtype=float)

    v_norm = np.linalg.norm(v)
    v_prime_norm = np.linalg.norm(v_prime)

    if v_norm == 0 or v_prime_norm == 0:
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    v = v / v_norm
    v_prime = v_prime / v_prime_norm
    axis = np.cross(v, v_prime)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        if np.dot(v, v_prime) > 0:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            orthogonal_axis = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(v, orthogonal_axis)
            axis = axis / np.linalg.norm(axis)
            return Quaternion(0.0, axis[0], axis[1], axis[2])

    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(v, v_prime), -1.0, 1.0))
    return Quaternion.from_angle(angle, axis)


class AttitudeDKF:
    """이산 칼만 필터 (Discrete Linear Kalman Filter)"""

    def __init__(self, q0: Quaternion, P0, Q, R, dt, m_ref):
        self.q = q0
        self.P = np.array(P0)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.dt = dt
        self.m_ref = np.array(m_ref)

    def predict(self, gyro_data):
        """[예측 단계] 자이로스코프 데이터로 다음 상태 예측"""
        wx, wy, wz = gyro_data
        gyro_vec = np.array([wx, wy, wz])
        w_norm = np.linalg.norm(gyro_vec)

        q_gyro = Quaternion(1.0, 0.0, 0.0, 0.0) if w_norm == 0 else Quaternion.from_angle(w_norm * self.dt, gyro_vec)

        # 상태 예측(x=Ax+Bu)
        self.q = q_gyro * self.q

        # 공분산 예측(P=APA^T+Q)
        F = q_gyro.matrix_left()
        self.P = F @ self.P @ F.T + self.Q

        return self.q

    def update(self, mag_data):
        """[보정 단계] 지자기 측정값 선처리 및 선형 보정"""
        mx, my, mz = mag_data

        z_quat = _quat_from_vec(self.m_ref, [mx, my, mz])
        z = z_quat.q

        H = np.eye(4)

        # Kalman Gain(PH^T(HPH^T+R)^-1)
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)

        # 상태 업데이트(x=x+K(z-Hx))
        q_updated_vec = self.q.q + K @ (z - H @ self.q.q)
        self.q = Quaternion.from_list(q_updated_vec).normalize()

        # 공분산 업데이트
        self.P = (np.eye(4) - K @ H) @ self.P

        return self.q

