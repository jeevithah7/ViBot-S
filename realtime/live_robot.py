"""
live_robot.py
-------------
The central real-time simulation engine for ViBot-S.

Runs a background thread that continuously:
  1. Steps the robot along the A* path
  2. Updates PID balancing state
  3. Simulates camera frame index
  4. Tracks odometry / position history

All data is exposed via thread-safe read() methods and is consumed
by the Flask dashboard's Server-Sent Events (SSE) endpoint.

The class also stores rolling history buffers for plotting:
  - tilt_history[]   : last N tilt samples
  - motor_history[]  : last N motor command samples
  - position_log[]   : (row, col) walked so far
"""

import sys, os, time, threading, math
import numpy as np
from collections import deque
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mapping.occupancy_grid import OccupancyGrid
from navigation.astar_planner import AStarPlanner
from control.pid_controller import PIDController, simple_pendulum_process
from simulation_3d.environment import Environment3D

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
MAP_ROWS     = 28
MAP_COLS     = 34
START        = (24, 21)
GOAL         = (12, 10)
HISTORY_LEN  = 120      # samples to keep in rolling buffer
STEP_PERIOD  = 0.45     # seconds between robot path steps
PID_PERIOD   = 0.02     # PID loop period (50 Hz)
# ──────────────────────────────────────────────────────────────────────────────


class LiveRobot:
    """
    Central simulation engine.  Call start() then read_state() repeatedly.

    Parameters
    ----------
    loop_steps   : int    How many times to reloop the path (0 = infinite)
    initial_tilt : float  Starting tilt angle in degrees
    noise_std    : float  Tilt noise standard deviation
    """

    def __init__(
        self,
        loop_steps: int = 0,
        initial_tilt: float = 1.8,
        noise_std: float = 0.25,
    ):
        self._loop_steps   = loop_steps
        self._initial_tilt = initial_tilt
        self._noise_std    = noise_std
        self._rng          = np.random.default_rng(seed=7)

        # Build map + path
        self._og = OccupancyGrid.build_house_map(rows=MAP_ROWS, cols=MAP_COLS)
        self.env3d = Environment3D(self._og.grid)
        planner  = AStarPlanner(self._og, diagonal=True)
        self._path = planner.plan(START, GOAL)
        if not self._path:
            raise RuntimeError("[LiveRobot] A* found no path – check map config.")

        # PID controller
        self._pid = PIDController(
            Kp=40.0, Ki=5.0, Kd=8.0,
            setpoint=0.0,
            output_limits=(-255.0, 255.0),
            integral_limit=50.0,
        )

        # State
        self._step_idx  = 0
        self._tilt      = float(initial_tilt)
        self._loops     = 0
        self._running   = False
        self._t_start   = 0.0
        self._step_t0   = 0.0
        self._last_th   = 0.0

        # History deques (fixed-length rolling windows)
        self._tilt_hist  : deque = deque(maxlen=HISTORY_LEN)
        self._motor_hist : deque = deque(maxlen=HISTORY_LEN)
        self._time_hist  : deque = deque(maxlen=HISTORY_LEN)
        self._pos_log    : list  = []       # full walked path

        # Grid state as serialisable list
        self._grid_state : list = self._build_grid_state()

        # Thread scaffolding
        self._lock = threading.Lock()
        self._path_thread  : Optional[threading.Thread] = None
        self._pid_thread   : Optional[threading.Thread] = None

    # ─── grid helpers ─────────────────────────────────────────────────────────
    def _build_grid_state(self) -> list:
        """Serialise the occupancy grid as a list-of-lists for JSON."""
        g = self._og.grid
        # 0=free  1=obstacle  -1=unknown
        return g.tolist()

    # ─── background threads ───────────────────────────────────────────────────
    def start(self):
        """Start both background simulation threads."""
        self._t_start  = time.time()
        self._running  = True

        self._pid_thread  = threading.Thread(
            target=self._pid_loop, daemon=True, name="vibot-pid")
        self._path_thread = threading.Thread(
            target=self._path_loop, daemon=True, name="vibot-path")

        self._pid_thread.start()
        self._path_thread.start()

    def stop(self):
        self._running = False

    # ─── PID loop (fast, ~50 Hz) ──────────────────────────────────────────────
    def _pid_loop(self):
        while self._running:
            t0 = time.monotonic()

            # Compute PID step
            motor_cmd = self._pid.compute(self._tilt, dt=PID_PERIOD)

            # Pendulum physics
            self._tilt -= motor_cmd * 0.004
            noise       = float(self._rng.normal(0, self._noise_std))
            self._tilt += noise
            self._tilt  = float(np.clip(self._tilt, -15.0, 15.0))

            # Update histories (thread-safe via deque.append being GIL-safe)
            elapsed = time.time() - self._t_start
            self._tilt_hist .append(round(self._tilt,    3))
            self._motor_hist.append(round(motor_cmd,     2))
            self._time_hist .append(round(elapsed,       3))

            # Pacing
            elapsed_loop = time.monotonic() - t0
            sleep_t = max(0.0, PID_PERIOD - elapsed_loop)
            time.sleep(sleep_t)

    # ─── path loop (slow, ~2 Hz) ──────────────────────────────────────────────
    def _path_loop(self):
        while self._running:
            t0 = time.monotonic()

            with self._lock:
                idx = self._step_idx
                if idx < len(self._path):
                    row, col = self._path[idx]
                    self._og.set_robot_position(row, col)
                    if not self._pos_log or self._pos_log[-1] != (row, col):
                        self._pos_log.append((row, col))
                    
                    if idx > 0 and idx < len(self._path):
                        r_prev, c_prev = self._path[idx-1]
                        self._last_th = math.atan2(row - r_prev, col - c_prev)

                    self._step_idx += 1
                else:
                    self._step_idx = 0
                    self._loops   += 1
                    self._tilt = self._initial_tilt
                    self._pid.reset()

            self._step_t0 = time.monotonic()
            elapsed_loop = time.monotonic() - t0
            time.sleep(max(0.0, STEP_PERIOD - elapsed_loop))

    # ─── public read interface ────────────────────────────────────────────────
    def read_state(self) -> dict:
        """
        Return a complete snapshot of robot state (thread-safe).

        The dict is JSON-serialisable and contains:
          - robot_pos        : [row, col]
          - path             : list of [row, col]
          - pos_log          : visited cells so far
          - grid             : 20×20 occupancy grid
          - tilt_history     : rolling list of tilt values
          - motor_history    : rolling list of motor commands
          - time_history     : timestamps matching tilt_history
          - current_tilt     : latest tilt angle
          - current_motor    : latest motor command
          - step_idx         : which path step we're on
          - total_steps      : path length
          - loops            : how many times we've looped
          - elapsed          : seconds since start
          - nav_status       : "Navigating" / "Goal Reached"
          - balance_status   : "Balancing Active"
          - mode             : always "simulation"
        """
        with self._lock:
            rp       = list(self._og.robot_pos)
            step_idx = self._step_idx
            loops    = self._loops
            pos_log  = list(self._pos_log[-100:])   # last 100 positions

        tilt_hist  = list(self._tilt_hist)
        motor_hist = list(self._motor_hist)
        time_hist  = list(self._time_hist)
        tilt_now   = tilt_hist[-1] if tilt_hist else 0.0
        motor_now  = motor_hist[-1] if motor_hist else 0.0
        elapsed    = round(time.time() - self._t_start, 1)

        nav_active = step_idx < len(self._path)
        nav_status = "Navigation Running" if nav_active else "Goal Reached"

        # Interpolate continuous 3D state
        frac = min(1.0, (time.monotonic() - self._step_t0) / STEP_PERIOD)
        smooth_x, smooth_y, smooth_theta = 1.0, 1.0, 0.0
        
        idx = max(0, min(step_idx - 1, len(self._path) - 1))
        if idx < len(self._path) - 1:
            r1, c1 = self._path[idx]
            r2, c2 = self._path[idx+1]
            smooth_x = c1 + (c2 - c1) * frac
            smooth_y = r1 + (r2 - r1) * frac
            
            target_th = math.atan2(r2 - r1, c2 - c1)
            # Shortest angle diff
            th_diff = (target_th - self._last_th + math.pi) % (2*math.pi) - math.pi
            smooth_theta = self._last_th + th_diff * frac
        elif len(self._path) > 0:
            r1, c1 = self._path[-1]
            smooth_x, smooth_y = float(c1), float(r1)
            smooth_theta = self._last_th

        return {
            "robot_pos"     : rp,
            "smooth_x"      : round(smooth_x, 3),
            "smooth_y"      : round(smooth_y, 3),
            "smooth_theta"  : round(smooth_theta, 3),
            "path"          : [list(p) for p in self._path],
            "pos_log"       : [list(p) for p in pos_log],
            "grid"          : self._grid_state,
            "tilt_history"  : tilt_hist,
            "motor_history" : motor_hist,
            "time_history"  : time_hist,
            "current_tilt"  : round(tilt_now, 3),
            "current_motor" : round(motor_now, 2),
            "step_idx"      : step_idx,
            "total_steps"   : len(self._path),
            "loops"         : loops,
            "elapsed"       : elapsed,
            "map_rows"      : MAP_ROWS,
            "map_cols"      : MAP_COLS,
            "nav_status"    : nav_status,
            "balance_status": "Balancing Active",
            "mode"          : "simulation",
            "start"         : list(START),
            "goal"          : list(GOAL),
            "env_walls"     : self.env3d.walls
        }


# ─── SINGLETON accessor (used by Flask app) ───────────────────────────────────
_robot_instance: Optional[LiveRobot] = None

def get_robot() -> LiveRobot:
    global _robot_instance
    if _robot_instance is None:
        _robot_instance = LiveRobot()
        _robot_instance.start()
    return _robot_instance

def reset_robot():
    global _robot_instance
    if _robot_instance:
        _robot_instance.stop()
    _robot_instance = LiveRobot()
    _robot_instance.start()


# ─── SELF-TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[live_robot] Starting self-test …")
    robot = LiveRobot()
    robot.start()
    for i in range(8):
        time.sleep(0.5)
        s = robot.read_state()
        print(f"  step={s['step_idx']}/{s['total_steps']}  "
              f"pos={s['robot_pos']}  tilt={s['current_tilt']:+.2f}°  "
              f"motor={s['current_motor']:+.0f}")
    robot.stop()
    print("[live_robot] Self-test complete.")
