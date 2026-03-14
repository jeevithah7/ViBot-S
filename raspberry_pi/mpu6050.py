"""
mpu6050.py
----------
MPU-6050 IMU (accelerometer + gyroscope) placeholder module.

The MPU-6050 is a 6-DOF IMU commonly paired with the Raspberry Pi via I²C.
On ViBot-S it provides tilt-angle readings for the PID balancing controller.

Stub mode
---------
When NOT on a Raspberry Pi the class returns simulated tilt data so the rest
of the system can still run.

Hardware deployment
-------------------
1. Wire MPU-6050 SDA → Pi GPIO 2 (SDA), SCL → Pi GPIO 3 (SCL).
2. Enable I2C on the Pi:  sudo raspi-config → Interfaces → I2C.
3. Install mpu6050 library:  pip install mpu6050-raspberrypi
4. Uncomment the hardware block below and remove the stub block.

Requires:
    pip install mpu6050-raspberrypi
"""

import math
import numpy as np


GRAVITY = 9.81   # m s⁻²


class MPU6050:
    """
    Wrapper for the MPU-6050 IMU on Raspberry Pi.

    Parameters
    ----------
    address  : int   I²C device address (default 0x68).
    stub     : bool  If True, use simulated readings (default True).
    """

    def __init__(self, address: int = 0x68, stub: bool = True):
        self.address = address
        self._stub   = stub
        self._sensor = None
        self._rng    = np.random.default_rng(seed=42)
        self._sim_tilt = 0.5   # simulated static tilt

        if not stub:
            # ── Uncomment on Raspberry Pi ──────────────────────────────────
            # from mpu6050 import mpu6050 as MPU
            # self._sensor = MPU(address)
            # print(f"[MPU6050] Sensor initialised at I2C address 0x{address:02X}")
            # ──────────────────────────────────────────────────────────────
            print("[MPU6050] Hardware mode selected but import is commented out.")
            print("  Falling back to stub mode.")
            self._stub = True
        else:
            print("[MPU6050] Running in STUB (simulated) mode.")

    # ------------------------------------------------------------------ public
    def get_tilt_angle(self) -> float:
        """
        Return the tilt (pitch) angle in degrees.

        Stub: returns a slowly drifting simulated value with Gaussian noise.
        Hardware: computes pitch from accelerometer x, y, z readings.

        Returns
        -------
        float  Pitch angle in degrees (+forward, −backward).
        """
        if self._stub:
            # Simulate a slight oscillation + noise
            noise = float(self._rng.normal(0, 0.2))
            self._sim_tilt = self._sim_tilt * 0.98 + noise
            return round(self._sim_tilt, 3)

        # ── Hardware implementation (uncomment on Pi) ──────────────────────
        # data = self._sensor.get_accel_data()
        # ax, ay, az = data["x"], data["y"], data["z"]
        # pitch = math.degrees(math.atan2(ax, math.sqrt(ay**2 + az**2)))
        # return round(pitch, 3)
        # ──────────────────────────────────────────────────────────────────
        return 0.0   # fallback

    def get_accel_data(self) -> dict[str, float]:
        """
        Return raw accelerometer readings (m s⁻²).

        Returns
        -------
        dict with keys 'x', 'y', 'z'
        """
        if self._stub:
            return {
                "x": float(self._rng.normal(0.0, 0.05)),
                "y": float(self._rng.normal(0.0, 0.05)),
                "z": float(self._rng.normal(GRAVITY, 0.1)),
            }
        # return self._sensor.get_accel_data()   # ← uncomment on Pi
        return {"x": 0.0, "y": 0.0, "z": GRAVITY}

    def get_gyro_data(self) -> dict[str, float]:
        """
        Return raw gyroscope readings (degrees per second).

        Returns
        -------
        dict with keys 'x', 'y', 'z'
        """
        if self._stub:
            return {
                "x": float(self._rng.normal(0.0, 0.5)),
                "y": float(self._rng.normal(0.0, 0.5)),
                "z": float(self._rng.normal(0.0, 0.5)),
            }
        # return self._sensor.get_gyro_data()   # ← uncomment on Pi
        return {"x": 0.0, "y": 0.0, "z": 0.0}


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    imu = MPU6050(stub=True)
    print("[mpu6050] Simulated readings:")
    for i in range(5):
        tilt  = imu.get_tilt_angle()
        accel = imu.get_accel_data()
        gyro  = imu.get_gyro_data()
        print(f"  [{i}] tilt={tilt:+.3f}°  az={accel['z']:.3f} m/s²  "
              f"gyro_x={gyro['x']:+.3f} °/s")
