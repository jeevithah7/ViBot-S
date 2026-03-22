"""
live_imu.py
-----------
Simulated IMU (MPU6050-style) sensor module for ViBot-S.

In SIMULATION mode  : generates realistic tilt, acceleration, and gyro data
                      with sensor noise, drift, and dynamic motion profiles.
In HARDWARE mode    : reads from an actual MPU6050 via I2C (Raspberry Pi).

The data is consistent with a two-wheeled self-balancing robot in motion.

Mode switch
-----------
    MODE = "simulation"   ← always use this without hardware
    MODE = "hardware"     ← requires smbus2 + MPU6050 hardware
"""

import time
import math
import threading
import numpy as np
from typing import Optional

# ─── MODE SWITCH ──────────────────────────────────────────────────────────────
MODE = "simulation"   # "simulation"  |  "hardware"
# ──────────────────────────────────────────────────────────────────────────────


# ─── SIMULATED IMU ────────────────────────────────────────────────────────────
class SimulatedIMU:
    """
    Generates realistic simulated IMU readings for a self-balancing robot.

    Mimics an MPU6050 with:
    - Tilt angle (pitch) that dynamically changes as robot balances
    - Acceleration (ax, ay, az) in m/s²
    - Gyroscope (gx, gy, gz) in °/s
    - Gaussian sensor noise
    - Simulated vibration from motors
    - Battery voltage droop over time

    Parameters
    ----------
    noise_std      : float   Std-dev of Gaussian noise (degrees / m/s²)
    update_rate_hz : float   Simulated sensor sample rate in Hz
    initial_tilt   : float   Starting tilt angle in degrees
    """

    def __init__(
        self,
        noise_std: float = 0.15,
        update_rate_hz: float = 50.0,
        initial_tilt: float = 1.5,
    ):
        self._noise_std   = noise_std
        self._dt          = 1.0 / update_rate_hz
        self._tilt        = initial_tilt
        self._velocity    = 0.0          # robot linear velocity (m/s)
        self._t           = 0.0          # internal time counter
        self._battery_pct = 100.0       # battery percentage
        self._rng         = np.random.default_rng(seed=13)

        # PID simulation state (simple model)
        self._integral    = 0.0
        self._prev_err    = 0.0
        self._KP, self._KI, self._KD = 38.0, 4.5, 7.5

        # Motor speeds (left, right) in RPM
        self._motor_left  = 0.0
        self._motor_right = 0.0

        # Lock for thread-safe reads
        self._lock = threading.Lock()
        self._latest: dict = self._compute()

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ─── background update loop ───────────────────────────────────────────────
    def start(self):
        """Start the background sensor update thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background update thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self):
        while self._running:
            sample = self._compute()
            with self._lock:
                self._latest = sample
            time.sleep(self._dt)

    # ─── physics model ────────────────────────────────────────────────────────
    def _compute(self) -> dict:
        self._t += self._dt

        # --- PID control on tilt ---
        error        = 0.0 - self._tilt      # setpoint = 0°
        self._integral += error * self._dt
        self._integral   = float(np.clip(self._integral, -50, 50))
        derivative    = (error - self._prev_err) / self._dt
        motor_cmd     = (self._KP * error
                         + self._KI * self._integral
                         + self._KD * derivative)
        motor_cmd     = float(np.clip(motor_cmd, -255, 255))
        self._prev_err = error

        # --- pendulum physics ---
        g_torque   = 9.81 * math.sin(math.radians(self._tilt))
        m_torque   = motor_cmd * 0.003
        theta_dot  = g_torque - m_torque
        self._tilt += theta_dot * self._dt
        self._tilt  = float(np.clip(self._tilt, -20, 20))

        # --- add sensor noise ---
        noise = lambda s=None: float(self._rng.normal(0, s or self._noise_std))

        tilt_reading = self._tilt + noise()

        # --- acceleration (robot in motion) ---
        ax = self._velocity * 0.1 + noise(0.05)
        ay = math.sin(self._t * 0.3) * 0.2 + noise(0.03)    # lateral sway
        az = 9.81 + noise(0.05)                               # gravity component

        # --- gyroscope ---
        gx = theta_dot * (180 / math.pi) + noise(0.5)
        gy = math.cos(self._t * 0.2) * 2.0 + noise(0.3)
        gz = noise(0.2)

        # --- robot velocity: loops between 0 and 0.5 m/s ---
        self._velocity = 0.25 * (1 + math.sin(self._t * 0.4))

        # --- motor RPM from command ---
        rpm_scale = abs(motor_cmd) * 2.0
        # slight differential for curves
        self._motor_left  = float(np.clip(
            rpm_scale + math.sin(self._t * 0.15) * 20, -600, 600))
        self._motor_right = float(np.clip(
            rpm_scale - math.sin(self._t * 0.15) * 20, -600, 600))

        # --- battery slow drain ---
        self._battery_pct = max(0.0, self._battery_pct - 0.0003)

        return {
            # IMU readings
            "tilt_angle"    : round(tilt_reading, 3),    # degrees
            "accel_x"       : round(ax, 4),              # m/s²
            "accel_y"       : round(ay, 4),
            "accel_z"       : round(az, 4),
            "gyro_x"        : round(gx, 3),              # °/s
            "gyro_y"        : round(gy, 3),
            "gyro_z"        : round(gz, 3),
            # robot state
            "velocity"      : round(self._velocity, 3),  # m/s
            "motor_left"    : round(self._motor_left,  1),
            "motor_right"   : round(self._motor_right, 1),
            "motor_cmd"     : round(motor_cmd, 2),
            "battery_pct"   : round(self._battery_pct, 2),
            # meta
            "timestamp"     : round(self._t, 3),
            "mode"          : "simulation",
        }

    # ─── public ───────────────────────────────────────────────────────────────
    def read(self) -> dict:
        """Return the most-recent sensor snapshot (thread-safe)."""
        with self._lock:
            return dict(self._latest)

    def read_once(self) -> dict:
        """Compute and return a single snapshot without threading."""
        s = self._compute()
        with self._lock:
            self._latest = s
        return s


# ─── HARDWARE IMU STUB (MPU6050 via I2C) ──────────────────────────────────────
class HardwareIMU:
    """
    Thin wrapper for the MPU6050 IMU on a Raspberry Pi.

    Requires: pip install smbus2 mpu6050-raspberrypi
    """

    def __init__(self, i2c_address: int = 0x68):
        try:
            from mpu6050 import mpu6050
            self._sensor = mpu6050(i2c_address)
            print(f"[HardwareIMU] MPU6050 at 0x{i2c_address:02X} opened.")
        except ImportError:
            raise RuntimeError(
                "[HardwareIMU] mpu6050 package not installed. "
                "Run: pip install mpu6050-raspberrypi"
            )

    def read(self) -> dict:
        accel = self._sensor.get_accel_data()
        gyro  = self._sensor.get_gyro_data()
        ax, ay, az = accel["x"], accel["y"], accel["z"]
        pitch = math.degrees(math.atan2(ax, math.sqrt(ay**2 + az**2)))
        return {
            "tilt_angle" : round(pitch,    3),
            "accel_x"    : round(ax,       4),
            "accel_y"    : round(ay,       4),
            "accel_z"    : round(az,       4),
            "gyro_x"     : round(gyro["x"], 3),
            "gyro_y"     : round(gyro["y"], 3),
            "gyro_z"     : round(gyro["z"], 3),
            "velocity"   : 0.0,
            "motor_left" : 0.0,
            "motor_right": 0.0,
            "motor_cmd"  : 0.0,
            "battery_pct": 99.9,
            "timestamp"  : time.time(),
            "mode"       : "hardware",
        }


# ─── FACTORY ──────────────────────────────────────────────────────────────────
def get_imu(**kwargs) -> SimulatedIMU | HardwareIMU:
    if MODE == "simulation":
        return SimulatedIMU(**kwargs)
    elif MODE == "hardware":
        return HardwareIMU(**kwargs)
    else:
        raise ValueError(f"[get_imu] Unknown MODE: '{MODE}'")


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    imu = SimulatedIMU()
    print("[live_imu] Self-test – 5 readings:")
    for i in range(5):
        d = imu.read_once()
        print(f"  {i}: tilt={d['tilt_angle']:+.2f}°  "
              f"az={d['accel_z']:.2f}  bat={d['battery_pct']:.1f}%")
        time.sleep(0.02)
