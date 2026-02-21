import numpy as np

class GPS:
    """
    Simulates a GPS sensor that returns noisy x, y position. 
    noise_std: standard deviation of the Gaussian noise in meters
    """

    def __init__(self, noise_std=2.0):
        self.noise_std = noise_std

    def read(self, true_state: np.ndarray) -> np.ndarray:
        x, y = true_state[0], true_state[1]
        noise = np.random.normal(0, self.noise_std, size=2)
        return np.array([x, y]) + noise

class IMU:
    """
    Simulates an IMU that reurns noisy heading and speed.
    heading_noise_std: standard deviation in radians
    speed_noise_std: standard deviation in m/s
    """

    def __init__(self, heading_noise_std=0.05, speed_noise_std=0.2):
        self.heading_noise_std = heading_noise_std
        self.speed_noise_std = speed_noise_std  

    def read(self, true_state: np.ndarray) -> np.ndarray:  
        heading, speed = true_state[2], true_state[3]
        heading_noise = np.random.normal(0, self.heading_noise_std)
        speed_noise = np.random.normal(0, self.speed_noise_std)
        return np.array([heading + heading_noise, speed + speed_noise])

class RangeSensor:
    """
    Simulates a range sensor measuring distance to a fixed landmark.
    landmark: (x, y) position of the landmark
    noise_std: standard deviation of range noise in meters
    """
    def __init__(self, landmark=(0.0, 0.0), noise_std=1.5):
        self.landmark = np.array(landmark)
        self.noise_std = noise_std 

    def read(self, true_state: np.ndarray) -> float:
        x, y = true_state[0], true_state[1]
        true_range = np.sqrt((x - self.landmark[0])**2 + (y - self.landmark[1])**2)
        noise = np.random.normal(0, self.noise_std)
        return true_range + noise


        