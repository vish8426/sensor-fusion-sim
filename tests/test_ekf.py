import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ekf import EKF
from src.world import World
from src.sensors import GPS, IMU, RangeSensor


class TestEKFPredict(unittest.TestCase):

    def test_predict_advances_position(self):
        """Predict step should move estimated position forward."""
        state = np.array([0.0, 0.0, 0.0, 5.0])
        ekf = EKF(initial_state=state)
        ekf.predict(dt=1.0, d_heading=0.0, d_speed=0.0)
        self.assertGreater(ekf.state[0], 0.0)

    def test_predict_increases_uncertainty(self):
        """Uncertainty (trace of P) should grow after predict with no update."""
        state = np.array([0.0, 0.0, 0.0, 5.0])
        ekf = EKF(initial_state=state)
        initial_uncertainty = np.trace(ekf.P)
        for _ in range(10):
            ekf.predict(dt=0.1, d_heading=0.0, d_speed=0.0)
        self.assertGreater(np.trace(ekf.P), initial_uncertainty)


class TestEKFUpdate(unittest.TestCase):

    def test_gps_update_reduces_uncertainty(self):
        """GPS update should reduce uncertainty."""
        state = np.array([0.0, 0.0, 0.0, 5.0])
        ekf = EKF(initial_state=state)
        ekf.predict(dt=0.1, d_heading=0.0, d_speed=0.0)
        uncertainty_before = np.trace(ekf.P)
        ekf.update_gps(np.array([0.1, 0.05]))
        self.assertLess(np.trace(ekf.P), uncertainty_before)

    def test_ekf_outperforms_gps_over_time(self):
        """EKF error should be lower than raw GPS error over a full run."""
        dt = 0.1
        world = World(dt=dt)
        gps = GPS(noise_std=2.0)
        imu = IMU()
        landmark = np.array([0.0, 0.0])
        range_sensor = RangeSensor(landmark=landmark, noise_std=1.5)
        ekf = EKF(initial_state=world.vessel.state())

        gps_errors = []
        ekf_errors = []

        for _ in range(500):
            ekf.predict(dt, 0.0, 0.0)
            world.step(0.0, 0.0)
            true_state = world.vessel.state()

            gps_reading = gps.read(true_state)
            ekf.update_gps(gps_reading)
            ekf.update_imu(imu.read(true_state))
            ekf.update_range(range_sensor.read(true_state), landmark)

            gps_errors.append(np.linalg.norm(gps_reading - true_state[:2]))
            ekf_errors.append(np.linalg.norm(ekf.state[:2] - true_state[:2]))

        self.assertLess(np.mean(ekf_errors), np.mean(gps_errors))


if __name__ == '__main__':
    unittest.main()