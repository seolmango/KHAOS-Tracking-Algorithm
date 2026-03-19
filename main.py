import matplotlib.pyplot as plt
from src.data_processing.data_loader import FlightDataLoader
from src.core.EKF import AttitudeEKF
from src.trajectories.imu import IMUTrajectory
from src.trajectories.gps import GPSTrajectory
from src.trajectories.simple import SIMPLETrajectory


def main():
    print("🚀 로켓 비행 궤적 추적 프로그램을 시작합니다...")

    # 1. 데이터 로드 및 전처리
    print("[1/4] 센서 데이터 로딩 및 초기화 중...")
    data_path = "data/identity3-B_lora_data.csv"
    loader = FlightDataLoader(data_path, dt=0.1)

    print(loader.dt)

    launch_idx = loader.indices['launch']
    touchdown_idx = loader.indices['touchdown']
    print(f"발사 인덱스: {launch_idx}, 착지 인덱스: {touchdown_idx}")

    # 2. EKF 엔진 구동 (자세 추정)
    print("[2/4] 확장 칼만 필터(EKF) 엔진 가동 중...")
    ekf = AttitudeEKF(loader.q0, loader.P0, loader.Q, loader.R, loader.dt, loader.m_ref)

    # 모든 시점의 회전(자세)을 저장할 리스트 (발사 전까지는 q0로 유지)
    rotators = [loader.q0.copy() for _ in range(loader.length)]

    for i in range(launch_idx, touchdown_idx):
        ekf.predict(loader.gyro[i])
        ekf.update(loader.mag[i])
        rotators[i + 1] = ekf.q

    # 3. 3가지 방식의 궤적 계산
    print("[3/4] 3D 비행 궤적 연산 중...")

    # 3-1. IMU 궤적 (EKF 자세 + 진짜 RK4 적분 적용)
    imu_traj = IMUTrajectory(rotators, loader.acc, loader.dt, loader.g0)
    imu_traj.calculate_trajectory(launch_idx, touchdown_idx)
    print(imu_traj.get_observer_acc(0))

    # 3-2. GPS 궤적 (하버사인 공식을 이용한 기하학적 궤적)
    gps_traj = GPSTrajectory(loader.gps_lat, loader.gps_lon, loader.altitude, start_idx=launch_idx)
    gps_traj.calculate_trajectory()

    # 3-3. Simple 궤적 (자이로스코프 단순 적분)
    simple_traj = SIMPLETrajectory(loader.acc, loader.gyro, loader.dt, loader.g0, loader.q0, launch_idx, touchdown_idx)
    simple_traj.calculate_trajectory()

    # 4. 결과 시각화
    print("[4/4] 3D 시각화 그래프 렌더링...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    imu_traj.plot_trajectory(ax, color='red', label='IMU (EKF + RK4)')
    gps_traj.plot_trajectory(ax, color='green', label='GPS (Haversine)')
    simple_traj.plot_trajectory(ax, color='blue', label='Simple (No EKF)')

    ax.set_title("Hanaro Rocket Flight Trajectory Estimation", fontsize=16)
    ax.set_xlabel("X Distance (m)")
    ax.set_ylabel("Y Distance (m)")
    ax.set_zlabel("Altitude Z (m)")

    ax.legend(loc='upper left')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()