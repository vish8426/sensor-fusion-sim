import numpy as np
import matplotlib.pyplot as plt


class Visualiser:
    """
    Handles all plotting for the sensor fusion simulator.
    """

    def plot_trajectory(self, true_positions: np.ndarray, 
                        gps_positions: np.ndarray,
                        ekf_positions: np.ndarray,
                        landmark: np.ndarray):
        """
        Plot ground truth, GPS scatter, and EKF estimate on a 2D map.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', linewidth=2, label='Ground Truth')
        plt.plot(gps_positions[:, 0], gps_positions[:, 1], 'r.', markersize=2, alpha=0.4, label='GPS (Noisy)')
        plt.plot(ekf_positions[:, 0], ekf_positions[:, 1], 'b-', linewidth=1.5, label='EKF Estimate')
        plt.scatter(landmark[0], landmark[1], c='black', s=100, zorder=5, label='Landmark (Origin)')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Sensor Fusion Simulator — EKF vs GPS vs Ground Truth')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

    def plot_errors(self, true_positions: np.ndarray,
                    gps_positions: np.ndarray,
                    ekf_positions: np.ndarray,
                    dt: float):
        """
        Plot GPS error vs EKF error over time.
        """
        gps_errors = np.linalg.norm(gps_positions - true_positions, axis=1)
        ekf_errors = np.linalg.norm(ekf_positions - true_positions, axis=1)
        timesteps = np.arange(len(gps_errors)) * dt

        plt.figure(figsize=(12, 4))
        plt.plot(timesteps, gps_errors, 'r-', alpha=0.6, label='GPS Error (m)')
        plt.plot(timesteps, ekf_errors, 'b-', linewidth=1.5, label='EKF Error (m)')
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        plt.title('GPS Error vs EKF Error Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    def show(self):
        plt.show()