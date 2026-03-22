"""
robot_simulator.py
------------------
Animated robot simulation.

The simulator places a robot on an occupancy grid, drives it along an A*
planned path step-by-step, and displays the movement as a live Matplotlib
animation.  A PID balancing signal is computed at each step and printed to
the console (and overlaid on the plot) to simulate the self-balancing loop.

Usage
-----
    sim = RobotSimulator(grid_map, path, pid_controller)
    sim.run()          # blocking – shows the animation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

from mapping.occupancy_grid import OccupancyGrid, FREE, OCCUPIED, UNKNOWN
from control.pid_controller import PIDController


# ─── ROBOT SIMULATOR CLASS ────────────────────────────────────────────────────
class RobotSimulator:
    """
    Simulates a robot moving along a planned path on an occupancy grid.

    Parameters
    ----------
    grid_map : OccupancyGrid
        The environment map.
    path : list of (row, col)
        Waypoints to follow (typically from A*).
    pid : PIDController
        Controller used to simulate the balancing loop.
    step_delay_ms : int
        Milliseconds between animation frames (default 300 ms).
    initial_tilt : float
        Simulated initial tilt angle in degrees (default 2.0°).
    noise_std : float
        Standard deviation of tilt noise in degrees (default 0.3°).
    """

    def __init__(
        self,
        grid_map: OccupancyGrid,
        path: list[tuple[int, int]],
        pid: PIDController,
        step_delay_ms: int = 300,
        initial_tilt: float = 2.0,
        noise_std: float = 0.3,
    ):
        self._map          = grid_map
        self._path         = path
        self._pid          = pid
        self._step_delay   = step_delay_ms
        self._tilt         = initial_tilt
        self._noise_std    = noise_std
        self._step_idx     = 0
        self._rng          = np.random.default_rng(seed=0)

        # PID history for overlay text
        self._pid_history: list[float] = []
        self._tilt_history: list[float] = []

    # ------------------------------------------------------------------ display
    def _build_display_array(self) -> np.ndarray:
        """Convert grid states to [0, 1] float image for imshow."""
        g = self._map.grid
        d = np.where(g == UNKNOWN, 0.5,
            np.where(g == FREE,    1.0, 0.0))
        return d.astype(np.float32)

    def run(self, title: str = "ViBot-S Simulation"):
        """
        Start the animated simulation (blocking call).

        The matplotlib FuncAnimation updates the plot every `step_delay_ms`
        milliseconds and stops when the path is fully traversed.
        """
        if not self._path:
            print("[RobotSimulator] Path is empty – nothing to simulate.")
            return

        # ── figure / axes ─────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                                 gridspec_kw={"width_ratios": [2, 1]})
        ax_map, ax_pid = axes

        fig.patch.set_facecolor("#1a1a2e")
        for ax in axes:
            ax.set_facecolor("#16213e")

        display = self._build_display_array()

        # Custom colourmap: black=obstacle, white=free, grey=unknown
        cmap = LinearSegmentedColormap.from_list(
            "occ", ["#0d1117", "#7f8c8d", "#ecf0f1"]
        )
        im = ax_map.imshow(display, cmap=cmap, vmin=0, vmax=1, origin="upper",
                           animated=True)

        # Path overlay
        pr = [p[0] for p in self._path]
        pc = [p[1] for p in self._path]
        path_line, = ax_map.plot([], [], "c-", linewidth=2, alpha=0.7, label="Path")
        path_line.set_data(pc, pr)

        # Start / goal markers
        ax_map.scatter(pc[0],  pr[0],  c="#2ecc71", s=120, zorder=5,
                       marker="o", label="Start")
        ax_map.scatter(pc[-1], pr[-1], c="#e74c3c", s=120, zorder=5,
                       marker="*", label="Goal")

        # Robot marker
        robot_dot, = ax_map.plot([], [], "r^", markersize=16,
                                 label="Robot", zorder=6)

        # Visited trail
        trail_line, = ax_map.plot([], [], "yo-", markersize=4,
                                  linewidth=1.5, alpha=0.6, label="Trail")

        ax_map.set_title(f"{title}", color="white", fontsize=13, fontweight="bold")
        ax_map.set_xlabel("Column", color="#bdc3c7")
        ax_map.set_ylabel("Row",    color="#bdc3c7")
        ax_map.tick_params(colors="#bdc3c7")
        leg = ax_map.legend(loc="upper right", fontsize=8,
                             facecolor="#2c3e50", labelcolor="white")

        # Status text overlay
        status_text = ax_map.text(
            0.02, 0.97, "", transform=ax_map.transAxes,
            fontsize=9, color="#f1c40f", verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#2c3e50", alpha=0.7),
        )

        # ── PID line chart ────────────────────────────────────────────────────
        ax_pid.set_title("PID Balancing  (tilt °)", color="white", fontsize=11)
        ax_pid.set_xlabel("Step",  color="#bdc3c7")
        ax_pid.set_ylabel("Angle (°)", color="#bdc3c7")
        ax_pid.tick_params(colors="#bdc3c7")
        ax_pid.set_xlim(0, len(self._path))
        ax_pid.set_ylim(-10, 10)
        ax_pid.axhline(0, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.7)
        ax_pid.set_facecolor("#16213e")

        tilt_line,  = ax_pid.plot([], [], color="#e74c3c",  linewidth=1.5,
                                   label="Tilt (°)")
        motor_line, = ax_pid.plot([], [], color="#3498db",  linewidth=1.5,
                                   alpha=0.6, linestyle=":", label="Motor cmd/10")
        ax_pid.legend(fontsize=8, facecolor="#2c3e50", labelcolor="white")

        # Visited trail data
        trail_rows: list[int] = []
        trail_cols: list[int] = []

        # ── animation update function ─────────────────────────────────────────
        def update(frame: int):
            if self._step_idx >= len(self._path):
                return im, robot_dot, trail_line, tilt_line, motor_line, status_text

            row, col = self._path[self._step_idx]

            # Update robot position on map
            self._map.set_robot_position(row, col)
            robot_dot.set_data([col], [row])

            # Trail
            trail_rows.append(row)
            trail_cols.append(col)
            trail_line.set_data(trail_cols, trail_rows)

            # PID balance step
            noise = self._rng.normal(0, self._noise_std)
            self._tilt += noise
            motor_cmd = self._pid.compute(self._tilt, dt=0.01)
            # Simple pendulum response: tilt decays toward 0 via motor
            self._tilt -= motor_cmd * 0.004
            self._tilt = float(np.clip(self._tilt, -15.0, 15.0))

            self._tilt_history.append(self._tilt)
            self._pid_history.append(motor_cmd)

            steps = list(range(len(self._tilt_history)))
            tilt_line.set_data(steps, self._tilt_history)
            motor_line.set_data(steps, [v / 10 for v in self._pid_history])

            status_text.set_text(
                f"Step : {self._step_idx + 1}/{len(self._path)}\n"
                f"Pos  : ({row}, {col})\n"
                f"Tilt : {self._tilt:+.2f}°\n"
                f"Motor: {motor_cmd:+.1f}"
            )

            self._step_idx += 1
            return im, robot_dot, trail_line, tilt_line, motor_line, status_text

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self._path) + 5,   # a few extra frames at the end
            interval=self._step_delay,
            blit=False,
            repeat=False,
        )

        plt.tight_layout()
        print(f"[RobotSimulator] Starting animation – {len(self._path)} steps.")
        plt.show()
        print("[RobotSimulator] Simulation complete.")

    # ------------------------------------------------------------------ static
    @staticmethod
    def preview_map(grid_map: OccupancyGrid, path=None):
        """Quick static preview of the map without animation."""
        grid_map.plot(path=path, title="Map Preview")


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from mapping.occupancy_grid import OccupancyGrid
    from navigation.astar_planner import AStarPlanner
    from control.pid_controller import PIDController

    og  = OccupancyGrid.build_sample_map(rows=20, cols=20)
    planner = AStarPlanner(og)
    path = planner.plan((1, 1), (18, 18))

    if not path:
        print("[robot_simulator] No path – aborting.")
    else:
        pid = PIDController(Kp=40.0, Ki=5.0, Kd=8.0,
                            setpoint=0.0, output_limits=(-255.0, 255.0))
        sim = RobotSimulator(og, path, pid, step_delay_ms=200)
        sim.run()
