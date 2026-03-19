import numpy as np

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.q = np.array([w, x, y, z], dtype=float)

    @classmethod
    def from_angle(cls, angle, axis):
        """
        특정 축을 기준으로 각도만큼 회전하는 쿼터니언을 반환합니다.
        :param angle: 각도(라디안)
        :param axis: 기준 축
        :return:
        """
        axis = np.array(axis, dtype=float)
        axis_norm = np.linalg.norm(axis)

        if axis_norm == 0:
            return cls(1.0, 0.0, 0.0, 0.0)

        axis = axis / axis_norm
        half_angle = angle / 2.0

        w = np.cos(half_angle)
        x, y, z = np.sin(half_angle) * axis
        return cls(w, x, y, z)

    @classmethod
    def from_vector(cls, w, v):
        """

        :param w: 스칼라
        :param v: 벡터
        :return:
        """
        return cls(w, v[0], v[1], v[2])

    @classmethod
    def from_list(cls, lst):
        return cls(lst[0], lst[1], lst[2], lst[3])

    @classmethod
    def from_two_vectors(cls, v_from, v_to):
        v_from = np.array(v_from, dtype=float)
        v_to = np.array(v_to, dtype=float)

        norm_from = np.linalg.norm(v_from)
        norm_to = np.linalg.norm(v_to)

        if norm_from == 0 or norm_to == 0:
            return cls(1.0, 0.0, 0.0, 0.0)

        v_from = v_from / norm_from
        v_to = v_to / norm_to

        axis = np.cross(v_from, v_to)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-8:
            if np.dot(v_from, v_to) > 0:
                return cls(1.0, 0.0, 0.0, 0.0)
            else:
                orthogonal_axis = np.array([1.0, 0.0, 0.0])
                axis = np.cross(v_from, orthogonal_axis)
                axis = axis / np.linalg.norm(axis)
                return cls(0.0, axis[0], axis[1], axis[2])

        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
        return cls.from_angle(angle, axis)

    def __repr__(self):
        return f"Quaternion({self.q[0]:.4f}, {self.q[1]:.4f}, {self.q[2]:.4f}, {self.q[3]:.4f})"

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion.from_list(self.q + other.q)
        raise TypeError("Addition is only supported between Quaternion and Quaternion")

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion.from_list(self.q - other.q)
        raise TypeError("Subtraction is only supported between Quaternion and Quaternion")

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            # 해밀턴 곱
            w1, x1, y1, z1 = self.q
            w2, x2, y2, z2 = other.q
            return Quaternion(
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            )
        elif isinstance(other, (int, float)):
            return Quaternion.from_list(self.q * other)
        raise TypeError("Multiplication is only supported between Quaternions or a scalar")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError()
            return Quaternion.from_list(self.q / other)
        raise TypeError("Division is only supported between a Quaternion and a scalar")

    def conjugate(self):
        w, x, y, z = self.q
        return Quaternion(w, -x, -y, -z)

    def normalize(self):
        n = np.linalg.norm(self.q)
        if n == 0:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion.from_list(self.q / n)

    def inverse(self):
        n_sq = np.linalg.norm(self.q) ** 2
        if n_sq == 0:
            raise ZeroDivisionError("Cannot invert a zero quaternion")
        conj = self.conjugate()
        return Quaternion.from_list(conj.q / n_sq)

    def matrix_left(self):
        q0, q1, q2, q3 = self.q
        return np.array([
            [q0, -q1, -q2, -q3],
            [q1, q0, -q3, q2],
            [q2, q3, q0, -q1],
            [q3, -q2, q1, q0]
        ])

    def matrix_right(self):
        q0, q1, q2, q3 = self.q
        return np.array([
            [q0, -q1, -q2, -q3],
            [q1, q0, q3, -q2],
            [q2, -q3, q0, q1],
            [q3, q2, -q1, q0]
        ])

    def vector(self):
        return self.q[1:]

    def copy(self):
        return Quaternion(*self.q)