"""
pid_controller.py
-----------------
Generic discrete-time PID controller.

Used by ViBot-S to balance the robot (like a two-wheeled self-balancing robot)
by computing motor correction values from tilt-angle error.

Control law
-----------
  u(t) = Kp·e(t) + Ki·∑e(t)·dt + Kd·(de/dt)

Anti-windup: the integral term is clamped to [-integral_limit, +integral_limit]
to prevent runaway when the error is persistently non-zero (e.g., hard wall).

Hardware note
-------------
On a Raspberry Pi, plug the output u directly into your L298N or similar
motor driver as a PWM duty-cycle value (clipped to [0, 255] for GPIO).

Simulation
----------
The self-test at the bottom simulates a 2-kg inverted-pendulum being
stabilised from a +5° initial tilt.
"""

import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


# ─── PID CONTROLLER ───────────────────────────────────────────────────────────
class PIDController:
    """
    Discrete-time PID controller.

    Parameters
    ----------
    Kp : float   Proportional gain.
    Ki : float   Integral gain.
    Kd : float   Derivative gain.
    setpoint : float
        Desired value (e.g., 0.0° tilt for balancing).
    sample_time : float
        Expected time between compute() calls in seconds (default 0.01 s).
    output_limits : tuple[float, float]
        (min, max) clamp applied to the controller output.
    integral_limit : float
        Symmetric clamp on the integral accumulator (anti-windup).
    derivative_filter_alpha : float
        Low-pass coefficient [0, 1] on the derivative term.
        0 → pure derivative, 1 → no derivative.  Default 0.1 (mild filter).
    """

    def __init__(
        self,
        Kp: float = 1.0,
        Ki: float = 0.0,
        Kd: float = 0.0,
        setpoint: float = 0.0,
        sample_time: float = 0.01,
        output_limits: tuple[float, float] = (-255.0, 255.0),
        integral_limit: float = 100.0,
        derivative_filter_alpha: float = 0.1,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint   = setpoint
        self.sample_time = sample_time
        self.output_limits  = output_limits
        self.integral_limit = integral_limit
        self._alpha = derivative_filter_alpha

        # Internal state
        self._integral     = 0.0
        self._prev_error   = 0.0
        self._filtered_d   = 0.0
        self._last_time: float | None = None

    # ------------------------------------------------------------------ public
    def compute(self, measurement: float, dt: float | None = None) -> float:
        """
        Compute the PID control output for the current measurement.

        Parameters
        ----------
        measurement : float
            Current process variable (e.g., current tilt angle in degrees).
        dt : float | None
            Time delta in **seconds** since last call.
            If None, the sample_time given at construction is used.

        Returns
        -------
        float
            Control output (e.g., motor PWM command).
        """
        if dt is None:
            dt = self.sample_time
        if dt <= 0:
            dt = self.sample_time

        error = self.setpoint - measurement

        # --- Proportional ---
        p_term = self.Kp * error

        # --- Integral (with anti-windup clamping) ---
        self._integral += error * dt
        self._integral = float(
            np.clip(self._integral, -self.integral_limit, self.integral_limit)
        )
        i_term = self.Ki * self._integral

        # --- Derivative (with low-pass filter) ---
        raw_d = (error - self._prev_error) / dt
        self._filtered_d = (
            (1 - self._alpha) * self._filtered_d + self._alpha * raw_d
        )
        d_term = self.Kd * self._filtered_d

        # --- Total output ---
        output = p_term + i_term + d_term
        output = float(np.clip(output, *self.output_limits))

        # Persist for next iteration
        self._prev_error = error

        return output

    def reset(self):
        """Reset all internal state (use after major disturbances)."""
        self._integral   = 0.0
        self._prev_error = 0.0
        self._filtered_d = 0.0
        self._last_time  = None

    # ------------------------------------------------------------------ display
    def tune_plot(
        self,
        process_fn,
        steps: int = 200,
        dt: float = 0.01,
        title: str = "PID Response",
    ) -> list[float]:
        """
        Simulate and plot the closed-loop response.

        Parameters
        ----------
        process_fn : callable
            f(prev_state, control_output, dt) → new_state
            Models the physical process being controlled.
        steps  : int    Number of simulation steps.
        dt     : float  Time step in seconds.
        title  : str    Plot title.

        Returns
        -------
        list[float]  History of process outputs (measurements).
        """
        self.reset()
        outputs: list[float] = []
        state = 5.0   # start 5° tilted

        for _ in range(steps):
            u = self.compute(state, dt=dt)
            state = process_fn(state, u, dt)
            outputs.append(state)

        t_axis = [i * dt for i in range(steps)]
        plt.figure(figsize=(10, 4))
        plt.plot(t_axis, outputs,  "b-",  label="Tilt angle (°)")
        plt.axhline(self.setpoint, color="r", linestyle="--", label="Setpoint (0°)")
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Tilt (°)")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

        return outputs


# ─── SIMPLE INVERTED-PENDULUM MODEL ───────────────────────────────────────────
def simple_pendulum_process(state: float, u: float, dt: float) -> float:
    """
    Approximate discrete-time model of a self-balancing robot.

    θ(t+dt) = θ(t) + θ̇·dt
    θ̇      = gravity_torque − motor_torque(u)

    Parameters
    ----------
    state : float  Current tilt angle θ in degrees.
    u     : float  Motor command (positive → correct lean).
    dt    : float  Time step in seconds.

    Returns
    -------
    float  New tilt angle θ in degrees.
    """
    g_torque = 9.81 * math.sin(math.radians(state))   # gravity pulling the robot
    m_torque = u * 0.04                                # motor counter-torque
    theta_dot = g_torque - m_torque
    new_state = state + theta_dot * dt
    # Clamp to ±45° (robot falls over beyond this)
    return float(np.clip(new_state, -45.0, 45.0))


# lazy import (only needed in self-test)
import math as math


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    pid = PIDController(
        Kp=40.0,
        Ki=5.0,
        Kd=8.0,
        setpoint=0.0,
        output_limits=(-255.0, 255.0),
        integral_limit=50.0,
    )

    print("[pid_controller] Simulating balancing from 5° initial tilt …")
    history = pid.tune_plot(
        process_fn=simple_pendulum_process,
        steps=300,
        dt=0.01,
        title="PID Balancing – Self-Test",
    )
    final_angle = history[-1]
    print(f"  Final tilt angle : {final_angle:.4f}°")
    print(f"  Settled within ±0.5°: {abs(final_angle) < 0.5}")
