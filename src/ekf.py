import numpy as np

class EKF:
    """
    Extended Kalman Filter for 2D vessel tracking. 
    State vector: [x, y, heading, speed]
    """

    def __init__(self, initial_state: np.ndarray, process_noise_std=0.1):
        # Initial state estimate - start with the true initial state
        self.state = initial_state.copy()

        # P: Covariance matrix - represents uncertainty in teh state estimate.
        # Starting with small values means fairly condifdent in the initial state. 
        self.P = np.eye(4) * 0.1

        # Q: Process noise ovariance - how much we expect the model to drift per step.
        # Higher values = less trust in our motion model prediction. 
        self.Q = np.eye(4) * (process_noise_std**2)

        # R matrices: Measurement noise covariance for each sensor. 
        # These should match the noise_std values we set in sensors.py. 
        self.R_gps = np.eye(2) * (2.0**2)  # GPS: 2m std
        self.R_imu = np.eye(2) * np.array([
            0.05**2,  # Heading noise std in radians
            0.2**2    # Speed noise std in m/s
        ])
        self.R_range = np.array([[1.5**2]])  # Range sensor: 1.5m std

    def predict(self, dt: float, d_heading: float, d_speed: float):
        """
        Predict step - advance state estimate using motion model. 
        """
        x, y, heading, speed = self.state

        # Apply the same motion model as Vessel.update()
        heading_new = heading + d_heading * dt
        speed_new = speed + d_speed * dt
        x_new = x + speed * np.cos(heading) * dt
        y_new = y + speed * np.sin(heading) * dt

        self.state = np.array([x_new, y_new, heading_new, speed_new])

        # F: Jacobian of the motion model (linearised around current state). 
        # This tells the filter how uncertainty propgates through the motion model.
        F = np.array([
            [1, 0, -speed * np.sin(heading) * dt, np.cos(heading) * dt],
            [0, 1,  speed * np.cos(heading) * dt, np.sin(heading) * dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Propagrate uncertainty forward
        self.P = F @ self.P @ F.T + self.Q

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, h_pred: np.ndarray):
        """
        Core update step - shared by all sensors.
        z       : actual sensor reading
        H       : Jacobian of sensor model
        R       : Sensor noise covariance
        h_pred  : Predicted sensor reading based on current state
        """

        # Innovation: difference between what sensor saw and what we expected
        innovation = z - h_pred

        # Innovation convariance
        S = H @ self.P @ H.T + R

        # Kalman Gain: how muich to trust the sensor vs our prediction
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state estimate
        self.state = self.state + K @ innovation

        # Update uncertainty - incorporating the sensor reduces our uncertainty
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_gps(self, z: np.ndarray):
        """
        Updates with GPS reading (x, y).
        """

        # H maps state to GPS observation (observe x and y directly)
        H = np.array([
            [1, 0, 0, 0],  # x depends on x
            [0, 1, 0, 0]   # y depends on y
        ])
        h_pred = np.array([self.state[0], self.state[1]])  # Predicted GPS reading based on current state
        self._update(z, H, self.R_gps, h_pred) 

    def update_imu(self, z: np.ndarray):
        """
        Update with IMU reading [heading, speed].
        """

        # H maps state to IMU observation (observe heading and speed directly)
        H = np.array([
            [0, 0, 1, 0],  # heading depends on heading
            [0, 0, 0, 1]   # speed depends on speed
        ])
        h_pred = np.array([self.state[2], self.state[3]])  # Predicted IMU reading based on current state
        self._update(z, H, self.R_imu, h_pred) 

    def update_range(self, z: np.ndarray, landmark: np.ndarray):
        """
        Update with range sensor reading - distance to a landmark. 
        This is nonlinear so linearise it here with a Jacobian.
        """

        x, y = self.state[0], self.state[1]
        lx, ly = landmark[0], landmark[1]  

        predicted_range = np.sqrt((x - lx)**2 + (y - ly)**2)

        # Avoid division by zero
        if predicted_range < 1e-6:
            return

        # H: Jacobian of the range function with respect to state
        H = np.array([[
            (x - lx) / predicted_range,  # dr/dx
            (y - ly) / predicted_range,  # dr/dy
            0,                           # dr/dheading
            0                            # dr/dspeed    
        ]])

        h_pred = np.array([predicted_range])  # Predicted range based on current state
        self._update(np.array([z]), H, self.R_range, h_pred)
