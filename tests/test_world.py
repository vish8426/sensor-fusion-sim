import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.world import Vessel, World

class TestVessel(unittest.TestCase):
    def test_initial_state(self):
        """Vessel initialises with corect default state."""
        vessel = Vessel()
        self.assertEqual(vessel.x, 0.0)
        self.assertEqual(vessel.y, 0.0)
        self.assertEqual(vessel.heading, 0.0)
        self.assertEqual(vessel.speed, 5.0)

    def test_straight_line_motion(self):
        """Vessel moving straight east should only increase x."""
        vessel = Vessel(heading=0.0, speed=10.0)
        vessel.update(dt=1.0)
        self.assertAlmostEqual(vessel.x, 10.0, places=5)
        self.assertAlmostEqual(vessel.y, 0.0, places=5)

    def test_heading_change(self):
        """Heading should update correctly with d_heading."""
        vessel = Vessel(heading=0.0)
        vessel.update(dt=1.0, d_heading=0.1)
        self.assertAlmostEqual(vessel.hjeading, 0.1, places=5)

    def test_speed_change(self):
        """Speed should update correctly with d_speed."""
        vessel = Vessel(speed=5.0)
        vessel.update(dt=1.0, d_speed=2.0)
        self.assertAlmostEqual(vessel.speed, 7.0, places=5)

    def test_state_vector_length(self):
        """State vector should have 4 elements."""
        vessel = Vessel()
        self.assertEqual(len(vessel.state()), 4)

class TestWorld(unittest.TestCase):
    def test_history_lenght(self):
        """History should record one entry per step."""
        world = World(dt=0.1)
        world.run(steps=50)
        self.assertEqual(len(world.history), 50)

    def test_time_advances(sself):
        """Simulation time should advance correctly."""
        world = World(dt=0.1)
        world.run(steps=10)
        self.assertAlmostEqual(world.time, 1.0, places=5)

    def test_control_fn_applied(self):
        """Control function should affect vessel heading."""
        world_no_control = World(dt=0.1)
        world_no_control.run(steps=100)

        world_with_control = World(dt=0.1)
        world_with_control.run(steps=100, control_fn=lambda t: (0.05, 0.0))

        self.assertNotAlmostEqual(
            world_no_control.vessel.heading, 
            world_with_control.vessel.heading,
            places=3
        )

if __name__ == '__main__':
    unittest.main()