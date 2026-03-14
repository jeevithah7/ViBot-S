"""
main.py
-------
ViBot-S: Self-Balancing Vision-Based Indoor Navigation Robot
============================================================

Main simulation pipeline.  Run this file directly:

    python main.py

What it does
------------
1.  Builds a synthetic indoor occupancy-grid map.
2.  Runs ORB feature detection on two simulated camera frames.
3.  Matches features between the frames (visual odometry prep).
4.  Estimates camera motion (Essential Matrix + pose recovery).
5.  Plans an A* path from start to goal on the map.
6.  Launches an animated robot simulation:
    • The robot follows the A* path step by step.
    • A live PID balancing chart is displayed alongside.
7.  Plots the final visual-odometry trajectory.
8.  Shows a PID step-response curve for self-balancing.
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# ─── Add project root to Python path ──────────────────────────────────────────
# This lets Python find the vision/, mapping/, etc. packages regardless of
# where you run the script from.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─── ViBot-S modules ──────────────────────────────────────────────────────────
from vision.camera_capture      import SimulatedCamera
from vision.feature_detection_orb import ORBDetector
from vision.feature_matching    import FeatureMatcher
from vision.visual_odometry     import VisualOdometry
from mapping.occupancy_grid     import OccupancyGrid
from navigation.astar_planner   import AStarPlanner
from control.pid_controller     import PIDController, simple_pendulum_process
from simulation.robot_simulator import RobotSimulator
from utils.visualization        import (
    plot_grid_with_path,
    plot_orb_features,
    plot_feature_matches,
    plot_pid_response,
    plot_vo_trajectory,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
MAP_ROWS    = 20
MAP_COLS    = 20
START       = (1,  1)
GOAL        = (18, 18)

ORB_FEATURES = 800
MATCH_RATIO  = 0.75

PID_KP = 40.0
PID_KI =  5.0
PID_KD =  8.0

ANIM_DELAY_MS = 250   # milliseconds between animation frames


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: divider banner
# ══════════════════════════════════════════════════════════════════════════════
def banner(text: str):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {text}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Build occupancy grid map
# ══════════════════════════════════════════════════════════════════════════════
def step_build_map() -> OccupancyGrid:
    banner("STEP 1 – Building Occupancy Grid Map")
    og = OccupancyGrid.build_sample_map(rows=MAP_ROWS, cols=MAP_COLS)
    print(f"  Grid size  : {og.rows} × {og.cols} cells")
    print(f"  Cell size  : {og.cell_size} m  →  room ≈ {og.rows*og.cell_size:.1f}×{og.cols*og.cell_size:.1f} m")
    print(f"  Robot start: {og.robot_pos}")
    return og


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Simulate camera capture + ORB feature detection
# ══════════════════════════════════════════════════════════════════════════════
def step_vision():
    banner("STEP 2 – Camera Capture & ORB Feature Detection")
    cam = SimulatedCamera(num_frames=3)

    frame1 = cam.get_frame()
    frame2 = cam.get_frame()
    cam.release()

    detector = ORBDetector(n_features=ORB_FEATURES)
    kp1, des1 = detector.detect_and_compute(frame1)
    kp2, des2 = detector.detect_and_compute(frame2)
    print(f"  Frame 1  : {len(kp1)} keypoints  | descriptor shape {des1.shape if des1 is not None else 'None'}")
    print(f"  Frame 2  : {len(kp2)} keypoints  | descriptor shape {des2.shape if des2 is not None else 'None'}")

    # Show ORB detections (non-blocking)
    fig_kp = plot_orb_features(frame1, kp1, title="ORB Keypoints – Frame 1")
    fig_kp.suptitle("Close this window to continue…", color="#f39c12", fontsize=9)
    plt.show(block=False)
    plt.pause(0.5)

    return frame1, frame2, kp1, kp2, des1, des2


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Feature matching
# ══════════════════════════════════════════════════════════════════════════════
def step_matching(frame1, frame2, kp1, kp2, des1, des2):
    banner("STEP 3 – Feature Matching (BF + Lowe's ratio test)")
    matcher = FeatureMatcher(ratio_threshold=MATCH_RATIO)
    good = matcher.match(des1, des2)
    print(f"  Good matches : {len(good)}")

    fig_m = plot_feature_matches(
        frame1, kp1, frame2, kp2, good,
        title="ORB Feature Matches – Frame 1 → Frame 2",
    )
    fig_m.suptitle("Close this window to continue…", color="#f39c12", fontsize=9)
    plt.show(block=False)
    plt.pause(0.5)

    return good


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Visual odometry
# ══════════════════════════════════════════════════════════════════════════════
def step_visual_odometry():
    banner("STEP 4 – Visual Odometry (Essential Matrix → Pose)")
    cam = SimulatedCamera(num_frames=8)
    vo  = VisualOdometry(n_features=ORB_FEATURES)

    prev = cam.get_frame()
    for i in range(7):
        curr = cam.get_frame()
        if curr is None:
            break
        result = vo.process_frame_pair(prev, curr)
        if result["success"]:
            R = result["R"]
            t = result["t"].ravel()
            print(f"  step {i+1}: inliers={result['inliers']:3d}  "
                  f"Δt=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})")
        prev = curr
    cam.release()

    print(f"\n  Final position estimate: {vo.trajectory[-1]}")
    fig_vo = plot_vo_trajectory(vo.trajectory, title="Visual Odometry Trajectory")
    fig_vo.suptitle("Close this window to continue…", color="#f39c12", fontsize=9)
    plt.show(block=False)
    plt.pause(0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: A* path planning
# ══════════════════════════════════════════════════════════════════════════════
def step_path_planning(og: OccupancyGrid) -> list[tuple[int, int]]:
    banner("STEP 5 – A* Path Planning")
    planner = AStarPlanner(og, diagonal=True)
    path = planner.plan(START, GOAL)

    if not path:
        print("  [!] A* found no path.  Check start/goal are in free cells.")
        return []

    print(f"  Start : {START}   Goal : {GOAL}")
    print(f"  Path length : {len(path)} steps")
    print(f"  Path preview (first 6 cells): {path[:6]} …")

    # Static map preview
    fig_map = plot_grid_with_path(og.grid, path=path, robot_pos=START,
                                   title="Occupancy Grid + A* Path")
    fig_map.suptitle("Close this window to start animation…",
                     color="#f39c12", fontsize=9)
    plt.show(block=False)
    plt.pause(0.5)

    return path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: PID balancing demo (standalone plot)
# ══════════════════════════════════════════════════════════════════════════════
def step_pid_demo():
    banner("STEP 6 – PID Self-Balancing Demonstration")
    pid = PIDController(
        Kp=PID_KP, Ki=PID_KI, Kd=PID_KD,
        setpoint=0.0,
        output_limits=(-255.0, 255.0),
        integral_limit=50.0,
    )

    DT    = 0.01
    STEPS = 300
    state = 5.0   # initial tilt 5°
    tilt_history: list[float]    = []
    motor_history: list[float]   = []

    for _ in range(STEPS):
        u = pid.compute(state, dt=DT)
        state = simple_pendulum_process(state, u, DT)
        tilt_history.append(state)
        motor_history.append(u)

    print(f"  Initial tilt : 5.0°")
    print(f"  Final   tilt : {tilt_history[-1]:+.4f}°")
    settled = abs(tilt_history[-1]) < 0.5
    print(f"  Settled (±0.5°): {settled}")

    fig_pid = plot_pid_response(tilt_history, motor_history, dt=DT,
                                 title="PID Balancing – Step Response")
    fig_pid.suptitle("Close this window to start robot animation…",
                     color="#f39c12", fontsize=9)
    plt.show(block=False)
    plt.pause(0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: Robot simulation (animated)
# ══════════════════════════════════════════════════════════════════════════════
def step_robot_simulation(og: OccupancyGrid, path: list[tuple[int, int]]):
    banner("STEP 7 – Animated Robot Simulation + Live PID Balancing")
    if not path:
        print("  [!] No path to simulate.")
        return

    pid = PIDController(
        Kp=PID_KP, Ki=PID_KI, Kd=PID_KD,
        setpoint=0.0,
        output_limits=(-255.0, 255.0),
        integral_limit=50.0,
    )
    sim = RobotSimulator(
        og, path, pid,
        step_delay_ms=ANIM_DELAY_MS,
        initial_tilt=2.0,
        noise_std=0.3,
    )
    sim.run()   # ← blocking – shows the animation window


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print("  ViBot-S: Self-Balancing Vision-Based Indoor Navigation Robot")
    print("  Simulation Pipeline")
    print("=" * 62)

    # 1. Map
    og = step_build_map()

    # 2. Vision
    frame1, frame2, kp1, kp2, des1, des2 = step_vision()

    # 3. Matching
    matches = step_matching(frame1, frame2, kp1, kp2, des1, des2)

    # 4. Visual odometry
    step_visual_odometry()

    # 5. Path planning
    path = step_path_planning(og)

    # 6. PID demo
    step_pid_demo()

    # 7. Animated simulation (blocking)
    step_robot_simulation(og, path)

    print("\n[main] All steps complete.  Goodbye!\n")
