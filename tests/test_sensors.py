import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sensors import GPS, IMU, RangeSensor

class TestGPS(unittest.TestCase):

    def test_output_shape(self):
        """GPS should return a 2-element array."""
        gps = GPS()
        state = np.array([100.0, 50.0, 0.0, 5.0])
        reading = gps.read(state)
        self.assertEqual(reading.shape, (2,))

    def test_noise_is_applied(self):
        """GPS reading should differ from true position due to noise."""
        gps = GPS(noise_std=2.0)
        state = np.array([100.0, 50.0, 0.0, 5.0])
        readings = np.array([gps.read(state) for _ in range(100)])
        self.assertGreater(np.std(readings[:, 0]), 0.1)

    def test_mean_close_to_truth(self):
        """Average of many GPS readings should be close to true position."""
        gps = GPS(noise_std=2.0)
        state = np.array([100.0, 50.0, 0.0, 5.0])
        readings = np.array([gps.read(state) for _ in range(1000)])
        self.assertAlmostEqual(np.mean(readings[:, 0]), 100.0, delta=0.5)
        self.assertAlmostEqual(np.mean(readings[:, 1]), 50.0, delta=0.5)

class TestIMU(unittest.TestCase):

    def test_output_shape(self):
        """IMU should return a 2-element array."""
        imu = IMU()
        state = np.array([0.0, 0.0, 0.5, 5.0])
        reading = imu.read(state)
        self.assertEqual(reading.shape, (2,))

    def test_mean_close_to_truth(self):
        """Average of many IMU readings should be close to true heading and speed."""
        imu = IMU(heading_noise_std=0.05, speed_noise_std=0.2)
        state = np.array([0.0, 0.0, 1.0, 5.0])
        readings = np.array([imu.read(state) for _ in range(1000)])
        self.assertAlmostEqual(np.mean(readings[:, 0]), 1.0, delta=0.1)
        self.assertAlmostEqual(np.mean(readings[:, 1]), 5.0, delta=0.1)

class TestRangeSensor(unittest.TestCase):

    def test_zero_noise_exact(self):
        """With zero noise, range sensor should return exact distance."""
        sensor = RangeSensor(landmark=(0.0, 0.0), noise_std=0.0)
        state = np.array([3.0, 4.0, 0.0, 5.0])
        reading = sensor.read(state)
        self.assertAlmostEqual(reading, 5.0, places=5)  # 3-4-5 triangle

    def test_mean_close_to_truth(self):
        """Average of many range readings should be close to true distance."""
        sensor = RangeSensor(landmark=(0.0, 0.0), noise_std=1.5)
        state = np.array([3.0, 4.0, 0.0, 5.0])
        readings = [sensor.read(state) for _ in range(1000)]
        self.assertAlmostEqual(np.mean(readings), 5.0, delta=0.2)


if __name__ == '__main__':
    unittest.main()