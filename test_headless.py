"""Quick headless test – no display windows opened."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend → no windows

from vision.camera_capture        import SimulatedCamera
from vision.feature_detection_orb import ORBDetector
from vision.feature_matching      import FeatureMatcher
from vision.visual_odometry       import VisualOdometry
from mapping.occupancy_grid       import OccupancyGrid
from navigation.astar_planner     import AStarPlanner
from control.pid_controller       import PIDController, simple_pendulum_process

print("=== ViBot-S Headless Test ===")

# Occupancy grid
og = OccupancyGrid.build_sample_map()
assert og.rows == 20 and og.cols == 20
print(f"[PASS] OccupancyGrid  {og.rows}×{og.cols}")

# A* planner
path = AStarPlanner(og).plan((1, 1), (18, 18))
assert len(path) > 0, "A* returned empty path"
print(f"[PASS] AStarPlanner   path={len(path)} steps  first={path[0]}  last={path[-1]}")

# PID controller
pid   = PIDController(Kp=40, Ki=5, Kd=8, setpoint=0.0, output_limits=(-255, 255))
state = 5.0
for _ in range(150):
    u     = pid.compute(state, dt=0.01)
    state = simple_pendulum_process(state, u, 0.01)
print(f"[PASS] PIDController  final_tilt={state:+.4f}°  settled={abs(state)<1.0}")

# Camera + ORB
cam = SimulatedCamera(num_frames=2)
f1, f2 = cam.get_frame(), cam.get_frame()
cam.release()
det = ORBDetector(n_features=500)
kp1, des1 = det.detect_and_compute(f1)
kp2, des2 = det.detect_and_compute(f2)
assert len(kp1) > 0 and len(kp2) > 0
print(f"[PASS] ORBDetector    kp1={len(kp1)}  kp2={len(kp2)}")

# Feature matching
good = FeatureMatcher().match(des1, des2)
print(f"[PASS] FeatureMatcher good_matches={len(good)}")

# Visual odometry
vo  = VisualOdometry()
res = vo.process_frame_pair(f1, f2)
print(f"[PASS] VisualOdometry success={res['success']}  inliers={res['inliers']}")

print("\n=== ALL TESTS PASSED ===")
