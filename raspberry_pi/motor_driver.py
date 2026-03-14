"""
motor_driver.py
---------------
L298N dual H-bridge motor driver placeholder module.

On ViBot-S the L298N drives two DC motors (left and right) for both
locomotion and self-balancing.  This stub provides the identical API as the
real driver so that the rest of the codebase can run unmodified in simulation.

Pinout (Raspberry Pi GPIO BCM numbers – change to match your wiring):
  IN1  = 17    →  Left motor direction A
  IN2  = 27    →  Left motor direction B
  IN3  = 22    →  Right motor direction A
  IN4  = 23    →  Right motor direction B
  ENA  = 18    →  Left motor enable  (PWM)
  ENB  = 24    →  Right motor enable (PWM)

Hardware deployment:
    pip install RPi.GPIO
Then uncomment the GPIO block inside the class.
"""

import numpy as np


# PWM frequency for motor speed control (Hz)
_PWM_FREQ = 1000


class L298NDriver:
    """
    L298N motor driver controller.

    Parameters
    ----------
    stub : bool
        True  → print commands only (simulation/stub mode).
        False → use real GPIO (Raspberry Pi, requires RPi.GPIO).
    """

    # GPIO pin numbers (BCM) ──────────────────────────────────────────────────
    _IN1 = 17;  _IN2 = 27   # Left motor
    _IN3 = 22;  _IN4 = 23   # Right motor
    _ENA = 18;  _ENB = 24   # PWM enable pins

    def __init__(self, stub: bool = True):
        self._stub = stub
        self._left_speed  = 0
        self._right_speed = 0

        if not stub:
            # ── Uncomment on Raspberry Pi ──────────────────────────────────
            # import RPi.GPIO as GPIO
            # self._GPIO = GPIO
            # GPIO.setmode(GPIO.BCM)
            # for pin in [self._IN1, self._IN2, self._IN3, self._IN4,
            #             self._ENA, self._ENB]:
            #     GPIO.setup(pin, GPIO.OUT)
            # self._pwm_l = GPIO.PWM(self._ENA, _PWM_FREQ)
            # self._pwm_r = GPIO.PWM(self._ENB, _PWM_FREQ)
            # self._pwm_l.start(0)
            # self._pwm_r.start(0)
            # print("[L298NDriver] GPIO initialised.")
            # ──────────────────────────────────────────────────────────────
            print("[L298NDriver] Hardware mode requested but GPIO code is commented.")
            print("  Falling back to stub mode.")
            self._stub = True
        else:
            print("[L298NDriver] Running in STUB mode (no GPIO).")

    # ------------------------------------------------------------------ public
    def set_speed(self, left: float, right: float):
        """
        Set motor speeds in the range [−255, +255].

        Positive → forward, Negative → backward, 0 → stop.

        Parameters
        ----------
        left  : float  Left motor command.
        right : float  Right motor command.
        """
        left  = float(np.clip(left,  -255, 255))
        right = float(np.clip(right, -255, 255))
        self._left_speed  = left
        self._right_speed = right

        if self._stub:
            print(f"[L298NDriver] STUB  left={left:+.1f}  right={right:+.1f}")
            return

        # ── Hardware implementation ────────────────────────────────────────
        # duty_l = abs(left)  / 255 * 100
        # duty_r = abs(right) / 255 * 100
        # self._set_direction(self._IN1, self._IN2, left)
        # self._set_direction(self._IN3, self._IN4, right)
        # self._pwm_l.ChangeDutyCycle(duty_l)
        # self._pwm_r.ChangeDutyCycle(duty_r)
        # ──────────────────────────────────────────────────────────────────

    def apply_pid(self, correction: float):
        """
        Apply a PID correction to both motors symmetrically (balancing mode).

        When the robot leans forward (positive tilt), increase forward speed.
        When it leans back (negative tilt), reverse.

        Parameters
        ----------
        correction : float  Output of PIDController.compute().
        """
        self.set_speed(left=correction, right=correction)

    def stop(self):
        """Halt both motors."""
        self.set_speed(0, 0)
        if self._stub:
            print("[L298NDriver] Motors stopped.")

    def cleanup(self):
        """Release GPIO resources (no-op in stub mode)."""
        if not self._stub:
            pass
            # self._pwm_l.stop()
            # self._pwm_r.stop()
            # self._GPIO.cleanup()
        print("[L298NDriver] Cleanup done.")

    # ------------------------------------------------------------------ private
    # def _set_direction(self, pin_a, pin_b, value):
    #     if value >= 0:      # forward
    #         self._GPIO.output(pin_a, GPIO.HIGH)
    #         self._GPIO.output(pin_b, GPIO.LOW)
    #     else:               # backward
    #         self._GPIO.output(pin_a, GPIO.LOW)
    #         self._GPIO.output(pin_b, GPIO.HIGH)


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    driver = L298NDriver(stub=True)
    driver.set_speed(left=150, right=150)
    driver.apply_pid(correction=-80.0)
    driver.stop()
    driver.cleanup()
    print("[motor_driver] Self-test complete.")
