import pygame
import numpy as np


class PygameVisualiser:
    """
    Real-time 2D visualiser for the sensor fusion simulator using pygame.
    """

    # Colours
    BLACK      = (0, 0, 0)
    WHITE      = (255, 255, 255)
    GREEN      = (0, 255, 0)
    RED        = (255, 80, 80)
    BLUE       = (80, 160, 255)
    YELLOW     = (255, 255, 0)
    DARK_GREY  = (30, 30, 30)
    LIGHT_GREY = (60, 60, 60)

    def __init__(self, width=1200, height=600, world_x_range=(0, 520),
                 world_y_range=(-20, 80)):
        pygame.init()
        self.width = width
        self.height = height
        self.world_x_range = world_x_range
        self.world_y_range = world_y_range
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Sensor Fusion Simulator — Real Time EKF")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("consolas", 18)
        self.font_small = pygame.font.SysFont("consolas", 14)

        # Trail storage
        self.true_trail  = []
        self.gps_trail   = []
        self.ekf_trail   = []

    def _world_to_screen(self, x: float, y: float):
        """
        Convert simulation world coordinates to screen pixel coordinates.
        Y is flipped because pygame's y axis increases downward.
        """
        sx = int((x - self.world_x_range[0]) /
                 (self.world_x_range[1] - self.world_x_range[0]) * self.width)
        sy = int((1 - (y - self.world_y_range[0]) /
                 (self.world_y_range[1] - self.world_y_range[0])) * self.height)
        return sx, sy

    def _draw_grid(self):
        """Draw a subtle background grid."""
        step_x = self.width // 10
        step_y = self.height // 5
        for x in range(0, self.width, step_x):
            pygame.draw.line(self.screen, self.LIGHT_GREY, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, step_y):
            pygame.draw.line(self.screen, self.LIGHT_GREY, (0, y), (self.width, y), 1)

    def _draw_legend(self):
        """Draw legend in top right corner."""
        items = [
            (self.GREEN,  "Ground Truth"),
            (self.RED,    "GPS (Noisy)"),
            (self.BLUE,   "EKF Estimate"),
            (self.YELLOW, "Landmark"),
        ]
        x, y = self.width - 220, 15
        for colour, label in items:
            pygame.draw.circle(self.screen, colour, (x, y + 6), 6)
            text = self.font_small.render(label, True, self.WHITE)
            self.screen.blit(text, (x + 15, y))
            y += 25

    def _draw_hud(self, sim_time: float, true_state: np.ndarray,
                  gps_error: float, ekf_error: float):
        """Draw heads-up display in top left corner."""
        lines = [
            f"Time:      {sim_time:6.1f}s",
            f"X:         {true_state[0]:6.1f}m",
            f"Y:         {true_state[1]:6.1f}m",
            f"Heading:   {np.degrees(true_state[2]):6.1f}deg",
            f"Speed:     {true_state[3]:6.1f}m/s",
            f"GPS Err:   {gps_error:6.2f}m",
            f"EKF Err:   {ekf_error:6.2f}m",
        ]
        y = 15
        for line in lines:
            text = self.font_large.render(line, True, self.WHITE)
            self.screen.blit(text, (15, y))
            y += 24

    def update(self, sim_time: float, true_state: np.ndarray,
               gps_reading: np.ndarray, ekf_state: np.ndarray,
               landmark: np.ndarray):
        """
        Call this every simulation step to update the display.
        Returns False if the user closes the window, True otherwise.
        """
        # Handle window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        # Add to trails
        self.true_trail.append(self._world_to_screen(true_state[0], true_state[1]))
        self.gps_trail.append(self._world_to_screen(gps_reading[0], gps_reading[1]))
        self.ekf_trail.append(self._world_to_screen(ekf_state[0], ekf_state[1]))

        # Clear screen
        self.screen.fill(self.DARK_GREY)
        self._draw_grid()

        # Draw trails
        for pos in self.gps_trail:
            pygame.draw.circle(self.screen, self.RED, pos, 2)
        if len(self.true_trail) > 1:
            pygame.draw.lines(self.screen, self.GREEN, False, self.true_trail, 2)
        if len(self.ekf_trail) > 1:
            pygame.draw.lines(self.screen, self.BLUE, False, self.ekf_trail, 2)

        # Draw current vessel positions as larger dots
        pygame.draw.circle(self.screen, self.GREEN,
                           self._world_to_screen(true_state[0], true_state[1]), 6)
        pygame.draw.circle(self.screen, self.BLUE,
                           self._world_to_screen(ekf_state[0], ekf_state[1]), 5)

        # Draw landmark
        lx, ly = self._world_to_screen(landmark[0], landmark[1])
        pygame.draw.circle(self.screen, self.YELLOW, (lx, ly), 8)
        pygame.draw.circle(self.screen, self.WHITE, (lx, ly), 8, 2)

        # Draw HUD and legend
        gps_error = float(np.linalg.norm(gps_reading - true_state[:2]))
        ekf_error = float(np.linalg.norm(ekf_state[:2] - true_state[:2]))
        self._draw_hud(sim_time, true_state, gps_error, ekf_error)
        self._draw_legend()

        pygame.display.flip()
        self.clock.tick(60)  # cap at 60fps
        return True

    def wait_for_close(self):
        """Hold the window open after simulation ends until user closes it."""
        waiting = True
        msg = self.font_large.render(
            "Simulation complete — press ESC or close window to exit",
            True, self.WHITE
        )
        self.screen.blit(msg, (self.width // 2 - msg.get_width() // 2,
                               self.height - 35))
        pygame.display.flip()
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    waiting = False
        pygame.quit()