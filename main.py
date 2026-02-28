import numpy as np
from src.world import World
from src.sensors import GPS, IMU, RangeSensor
from src.ekf import EKF
from src.visualiser import Visualiser
from src.pygame_visualiser import PygameVisualiser


def control_fn(t):
    if t < 30.0:
        return (0.01, 0.0)
    elif t < 60.0:
        return (-0.01, 0.0)
    else:
        return (0.0, 0.0)


def run_simulation(mode: str = "pygame"):
    """
    mode: "pygame" for real-time visualisation
          "matplotlib" for static plots
    """
    dt = 0.1
    world = World(dt=dt)
    gps = GPS(noise_std=2.0)
    imu = IMU(heading_noise_std=0.05, speed_noise_std=0.2)
    landmark = np.array([0.0, 0.0])
    range_sensor = RangeSensor(landmark=landmark, noise_std=1.5)
    ekf = EKF(initial_state=world.vessel.state())

    true_positions = []
    gps_positions  = []
    ekf_positions  = []

    if mode == "pygame":
        vis = PygameVisualiser()

        for step in range(1000):
            d_heading, d_speed = control_fn(world.time)
            ekf.predict(dt, d_heading, d_speed)
            world.step(d_heading, d_speed)
            true_state = world.vessel.state()

            gps_reading   = gps.read(true_state)
            imu_reading   = imu.read(true_state)
            range_reading = range_sensor.read(true_state)

            ekf.update_gps(gps_reading)
            ekf.update_imu(imu_reading)
            ekf.update_range(range_reading, landmark)

            true_positions.append(true_state[:2].copy())
            gps_positions.append(gps_reading.copy())
            ekf_positions.append(ekf.state[:2].copy())

            running = vis.update(world.time, true_state,
                                 gps_reading, ekf.state, landmark)
            if not running:
                break

        vis.wait_for_close()

    elif mode == "matplotlib":
        visualiser = Visualiser()

        for step in range(1000):
            d_heading, d_speed = control_fn(world.time)
            ekf.predict(dt, d_heading, d_speed)
            world.step(d_heading, d_speed)
            true_state = world.vessel.state()

            gps_reading   = gps.read(true_state)
            imu_reading   = imu.read(true_state)
            range_reading = range_sensor.read(true_state)

            ekf.update_gps(gps_reading)
            ekf.update_imu(imu_reading)
            ekf.update_range(range_reading, landmark)

            true_positions.append(true_state[:2].copy())
            gps_positions.append(gps_reading.copy())
            ekf_positions.append(ekf.state[:2].copy())

        true_positions = np.array(true_positions)
        gps_positions  = np.array(gps_positions)
        ekf_positions  = np.array(ekf_positions)

        gps_error = np.linalg.norm(gps_positions[-1] - true_positions[-1])
        ekf_error = np.linalg.norm(ekf_positions[-1] - true_positions[-1])
        print(f"Final GPS Error:  {gps_error:.2f}m")
        print(f"Final EKF Error:  {ekf_error:.2f}m")

        visualiser.plot_trajectory(true_positions, gps_positions,
                                   ekf_positions, landmark)
        visualiser.plot_errors(true_positions, gps_positions,
                               ekf_positions, dt)
        visualiser.show()


if __name__ == "__main__":
    run_simulation(mode="pygame")       # switch to "matplotlib" for static plots