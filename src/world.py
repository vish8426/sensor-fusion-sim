import numpy as np

class Vessel:
    """
    Represents a 2D vessel with ground truth state.
    State vector: [x, y, heading (rad), speed (m/s)]
    """

    def __init__(self, x=0.0, y=0.0, heading=0.0, speed=5.0):
        self.x = x
        self.y = y
        self.heading = heading  # radians, 0 = east, pi/2 = north
        self.speed = speed      # m/s

    def update(self, dt: float, d_heading: float = 0.0, d_speed: float = 0.0):
        """
        Advance vessel state by one timestep. 
        dt          : time delta in seconds
        d_heading   : change in heading (rad/s)
        d_speed     : change in speed (m/s^2)
        """
        # Update heading and speed
        self.heading += d_heading * dt
        self.speed += d_speed * dt

        # Update position based on current heading and speed
        self.x += self.speed * np.cos(self.heading) * dt
        self.y += self.speed * np.sin(self.heading) * dt

    def state(self):
        return np.array([self.x, self.y, self.heading, self.speed])

class World:
    """
    Manages simulation time and records ground truth trajectory.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.time = 0.0
        self.vessel = Vessel()
        self.history = [] # list of (time, state) tuples

    def step(self, d_heading: float = 0.0, d_speed: float = 0.0):
        self.vessel.update(self.dt, d_heading, d_speed)
        self.time += self.dt
        self.history.append((self.time, self.vessel.state().copy()))   

    def run(self, steps: int, control_fn=None):
        """
        Run the simulation for a given number of steps.
        control_fn: optional callable(time) -> (d_heading, d_speed)
        """
        for _ in range(steps):
            if control_fn:
                d_heading, d_speed = control_fn(self.time)
            else:
                d_heading, d_speed = 0.0, 0.0
            self.step(d_heading, d_speed)