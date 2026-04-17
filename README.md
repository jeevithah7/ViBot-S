# ViBot-S 🤖 — Self-Balancing Vision-Based Indoor Navigation Robot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-latest-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-latest-red)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Raspberry%20Pi-lightgrey)

A complete Python project that simulates a **self-balancing indoor navigation robot** using computer
vision (ORB features, monocular visual odometry), grid-based mapping, A\* path planning, and PID
balancing control.  Runs fully in simulation — hardware modules for Raspberry Pi 4, MPU-6050,
and L298N motor driver are included as ready-to-enable stubs.

---

## 📁 Project Structure

```
ViBot-S/
│
├── vision/                     # Computer-vision pipeline
│   ├── camera_capture.py       # Simulated / Pi camera capture
│   ├── feature_detection_orb.py# ORB keypoint detection
│   ├── feature_matching.py     # BFMatcher + Lowe's ratio test
│   └── visual_odometry.py      # Essential Matrix → R, t estimation
│
├── mapping/
│   └── occupancy_grid.py       # 2-D grid map (free / obstacle / unknown)
│
├── navigation/
│   └── astar_planner.py        # A* shortest-path planner
│
├── control/
│   └── pid_controller.py       # Discrete PID with anti-windup
│
├── simulation/
│   └── robot_simulator.py      # Matplotlib animated simulation
│
├── utils/
│   └── visualization.py        # Reusable dark-themed plot helpers
│
├── raspberry_pi/               # Hardware stub modules
│   ├── rpi_camera.py           # PiCamera2 stub
│   ├── mpu6050.py              # MPU-6050 IMU stub
│   └── motor_driver.py         # L298N motor driver stub
│
├── datasets/                   # Place your own images here (optional)
├── main.py                     # ← Run this
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install opencv-python numpy matplotlib scipy
```

### 2. Run the Real-Time 3D Simulation Web Dashboard

To view the interactive 3D simulation of the robot, you need to serve the frontend files locally.

Navigate to the `frontend` directory and start a local HTTP server:

```bash
cd frontend
npx http-server -p 8000
```

Then, open your web browser and navigate to:
`http://localhost:8000/vibot_3d_simulation.html`

This will display the working project output with the real-time 3D simulation.

### 3. (Optional) Run the Original Offline Simulation

To see the step-by-step algorithms (OpenCV windows, Plotly animations):
```bash
python main.py
```
A series of matplotlib windows will appear. Close each window to advance to the next step.

---

## 📖 How It Works

### 🔍 Visual Odometry

```
Frame N  ──ORB──▶  keypoints + descriptors
Frame N+1 ──ORB──▶  keypoints + descriptors
                        │
                   BF Match + Lowe ratio
                        │
                   Good 2-D correspondences
                        │
               findEssentialMat (RANSAC)
                        │
                   Essential Matrix  E
                        │
                   recoverPose  →  R, t
                        │
                Accumulate trajectory
```

Monocular visual odometry estimates the **relative rotation R** and **translation direction t**
between two camera frames by:

1. Detecting ORB keypoints in both frames.
2. Matching their binary descriptors with Hamming distance.
3. Applying Lowe's ratio test (k=2, threshold 0.75) to reject ambiguous matches.
4. Computing the Essential Matrix E via RANSAC (5-point algorithm internally used by `findEssentialMat`).
5. Decomposing E → (R, t) with `recoverPose`.

> ⚠️ **Scale ambiguity**: monocular VO gives unit-length translations.  Real scale requires stereo
> cameras or IMU fusion.

---

### 🗺️ Occupancy Grid Mapping

The `OccupancyGrid` class stores a `(rows × cols)` integer array with three states:
| Value | Meaning |
|------:|---------|
|  `0`  | FREE — navigable |
|  `1`  | OCCUPIED — obstacle |
| `-1`  | UNKNOWN — not yet observed |

`build_sample_map()` creates a 20 × 20 simulated indoor room with perimeter walls and three
interior obstacles.  In a full SLAM system, cells would be updated probabilistically from sensor
readings.

---

### 🧭 A\* Path Planning

A\* is a best-first graph search that minimises `f(n) = g(n) + h(n)`:

| Term | Meaning |
|------|---------|
| `g(n)` | Exact cost from start to node n |
| `h(n)` | Admissible heuristic — **octile distance** (default) |
| `f(n)` | Estimated total path cost through n |

The planner uses an 8-connected grid (diagonal moves allowed, cost √2) and a binary min-heap for
O(log N) priority-queue operations.  Only `FREE` cells are added to the open set.

```python
planner = AStarPlanner(og, diagonal=True)
path = planner.plan(start=(1,1), goal=(18,18))
```

---

### ⚖️ PID Self-Balancing

A **discrete-time PID controller** corrects the robot's tilt to maintain vertical balance:

```
u(t) = Kp·e(t) + Ki·∑e·dt + Kd·(Δe/dt)
```

| Gain | Role |
|------|------|
| **Kp** | Proportional — responds instantly to current tilt |
| **Ki** | Integral — eliminates steady-state offset |
| **Kd** | Derivative — damps oscillations |

**Anti-windup**: the integral accumulator is clamped symmetrically so it cannot saturate the output
even when the robot is stuck against a wall.

**Derivative filter**: a low-pass exponential filter (`alpha=0.1`) suppresses noise on the D-term.

Default tuning: `Kp=40, Ki=5, Kd=8` — tested on the built-in inverted-pendulum model.

---

## 🖥️ Simulation Output

| Window | Content |
|--------|---------|
| ORB Keypoints | Green circles over detected features (rich keypoint mode) |
| Feature Matches | Colour lines connecting matched points across frames |
| VO Trajectory | Red/yellow 2-D scatter trail of estimated camera positions |
| Grid + Path | Black/white map, cyan path, green start, yellow goal, red robot marker |
| PID Response | Top: tilt angle converging to 0°; Bottom: motor command history |
| Animation | Left: robot (red ▲) moving on map; Right: live PID chart per step |

---

## 🔌 Raspberry Pi Deployment

All hardware modules live in `raspberry_pi/`.  To activate them on real hardware:

### Pi Camera (PiCamera2)

```bash
pip install picamera2
```
In `raspberry_pi/rpi_camera.py`, uncomment the `picamera2` import block and comment out the stub.

### MPU-6050 IMU

```bash
sudo apt install -y i2c-tools
pip install mpu6050-raspberrypi
```
Enable I²C: `sudo raspi-config → Interfaces → I2C → Enable`  
In `raspberry_pi/mpu6050.py`, set `stub=False` and uncomment the sensor block.

### L298N Motor Driver

```bash
pip install RPi.GPIO
```
In `raspberry_pi/motor_driver.py`, set `stub=False` and uncomment the GPIO block.  
Update the BCM pin numbers to match your wiring.

### Wiring Quick-Reference

```
Pi GPIO       L298N
─────────────────────
GPIO 17  →  IN1   (Left  motor A)
GPIO 27  →  IN2   (Left  motor B)
GPIO 22  →  IN3   (Right motor A)
GPIO 23  →  IN4   (Right motor B)
GPIO 18  →  ENA   (Left  PWM)
GPIO 24  →  ENB   (Right PWM)

Pi I²C        MPU-6050
──────────────────────
GPIO 2 (SDA) →  SDA
GPIO 3 (SCL) →  SCL
3.3 V        →  VCC
GND          →  GND
```

---

## 🛠️ Configuration (main.py)

```python
MAP_ROWS    = 20       # Grid height
MAP_COLS    = 20       # Grid width
START       = (1,  1)  # Robot start cell (row, col)
GOAL        = (18, 18) # Robot goal  cell (row, col)

ORB_FEATURES = 800     # Max ORB keypoints per frame
MATCH_RATIO  = 0.75    # Lowe's ratio threshold

PID_KP = 40.0          # Proportional gain
PID_KI =  5.0          # Integral gain
PID_KD =  8.0          # Derivative gain

ANIM_DELAY_MS = 250    # ms between animation frames
```

---

## 📚 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥ 4.5 | Feature detection, matching, essential matrix |
| `numpy` | ≥ 1.22 | Array operations, camera math |
| `matplotlib` | ≥ 3.5 | Visualization, animation |
| `scipy` | ≥ 1.8 | (Available for SLAM extensions) |

Install all at once:
```bash
pip install opencv-python numpy matplotlib scipy
```

---

## 🗺️ Roadmap

- [x] ORB feature detection & matching
- [x] Monocular visual odometry
- [x] Occupancy grid mapping
- [x] A\* path planning
- [x] PID balancing controller
- [x] Animated simulation
- [x] Raspberry Pi hardware stubs
- [ ] Stereo visual odometry (to resolve scale ambiguity)
- [ ] EKF/UKF sensor fusion (camera + IMU)
- [ ] ROS 2 integration
- [ ] Real-time SLAM with occupancy probability updates

---

## 📄 License

See [LICENSE](LICENSE) in the project root.

---

*Built with ❤️ for robotics enthusiasts — from simulation to hardware in one codebase.*
