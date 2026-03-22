"""
components.py
-------------
Reusable UI component helpers for the ViBot-S dashboard.

These are Python-side utilities that support the Flask backend:

  • graph_config()      : Returns Chart.js dataset config dicts
  • format_sensor()     : Pretty-print a sensor reading
  • status_color()      : Map a status string to a CSS colour
  • battery_color()     : Battery level → green/yellow/red CSS colour
  • make_grid_json()    : Serialise OccupancyGrid to compact JSON
  • make_path_json()    : Serialise the A* path for the frontend

These are NOT HTML components – the HTML is embedded in app.py for
portability. This module keeps the backend helper logic clean and testable.
"""

from __future__ import annotations
import json
import numpy as np

# ─── Sensor formatter ─────────────────────────────────────────────────────────

def format_sensor(value: float, unit: str = "", precision: int = 3) -> str:
    """Return a formatted sensor reading string with sign."""
    return f"{value:+.{precision}f} {unit}".strip()


# ─── Status helpers ───────────────────────────────────────────────────────────

def status_color(status: str) -> str:
    """Map a status string to a CSS hex colour."""
    s = status.lower()
    if "running" in s or "active" in s or "balancing" in s:
        return "#00e676"   # green
    if "reached" in s or "complete" in s:
        return "#7c4dff"   # purple
    if "error" in s or "fail" in s:
        return "#ff1744"   # red
    if "init" in s or "starting" in s:
        return "#ffd740"   # yellow
    return "#7a90b0"       # dim grey


def battery_color(pct: float) -> str:
    """Return CSS colour based on battery percentage."""
    if pct > 60:
        return "#00e676"   # green
    if pct > 25:
        return "#ffd740"   # yellow
    return "#ff5252"       # red


# ─── Grid / path serialisers ──────────────────────────────────────────────────

def make_grid_json(grid: np.ndarray) -> list[list[int]]:
    """
    Convert the occupancy grid numpy array to a JSON-serialisable list.

    Values:
        -1 → unknown  → displayed as dark grey
         0 → free     → displayed as dark blue-grey
         1 → occupied → displayed as near-black (obstacle)
    """
    return grid.tolist()


def make_path_json(path: list[tuple[int, int]]) -> list[list[int]]:
    """Convert A* path list to JSON-serialisable list of [row, col]."""
    return [list(p) for p in path]


# ─── Chart.js dataset configs ─────────────────────────────────────────────────

def tilt_dataset_config(data: list[float]) -> dict:
    """Chart.js dataset config for the PID tilt chart."""
    return {
        "label"          : "Tilt (°)",
        "data"           : data,
        "borderColor"    : "#ff5252",
        "backgroundColor": "rgba(255,82,82,0.08)",
        "borderWidth"    : 2,
        "tension"        : 0.4,
        "pointRadius"    : 0,
        "fill"           : True,
    }


def motor_dataset_config(data: list[float]) -> dict:
    """Chart.js dataset config for the motor command chart."""
    return {
        "label"          : "Motor CMD",
        "data"           : data,
        "borderColor"    : "#00e5ff",
        "backgroundColor": "rgba(0,229,255,0.06)",
        "borderWidth"    : 2,
        "tension"        : 0.4,
        "pointRadius"    : 0,
        "fill"           : True,
    }


# ─── State summary ────────────────────────────────────────────────────────────

def summarise_state(state: dict) -> str:
    """
    Return a human-readable multiline summary of the robot state dict.
    Useful for console logging or debugging.
    """
    lines = [
        "─" * 50,
        "  ViBot-S State Snapshot",
        "─" * 50,
        f"  Position   : {state.get('robot_pos', '?')}",
        f"  Step       : {state.get('step_idx', '?')} / {state.get('total_steps', '?')}",
        f"  Tilt       : {state.get('current_tilt', 0):+.3f}°",
        f"  Motor CMD  : {state.get('current_motor', 0):+.1f}",
        f"  Elapsed    : {state.get('elapsed', 0):.1f}s",
        f"  Loops      : {state.get('loops', 0)}",
        f"  Nav Status : {state.get('nav_status', '?')}",
        f"  Balance    : {state.get('balance_status', '?')}",
        f"  Mode       : {state.get('mode', '?')}",
    ]
    if "imu" in state:
        imu = state["imu"]
        lines += [
            "  ── IMU ──",
            f"  Battery   : {imu.get('battery_pct', 0):.1f}%",
            f"  Velocity  : {imu.get('velocity', 0):.3f} m/s",
            f"  Motor L   : {imu.get('motor_left', 0):.0f} rpm",
            f"  Motor R   : {imu.get('motor_right', 0):.0f} rpm",
        ]
    lines.append("─" * 50)
    return "\n".join(lines)


# ─── SELF-TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(format_sensor(3.141592, "m/s²", 4))
    print(status_color("Navigation Running"))
    print(battery_color(18.5))
    test_state = {
        "robot_pos"     : [5, 7],
        "step_idx"      : 10,
        "total_steps"   : 34,
        "current_tilt"  : -1.23,
        "current_motor" : 42.0,
        "elapsed"       : 15.3,
        "loops"         : 1,
        "nav_status"    : "Navigation Running",
        "balance_status": "Balancing Active",
        "mode"          : "simulation",
        "imu"           : {"battery_pct": 97.5, "velocity": 0.21,
                           "motor_left": 320.0, "motor_right": 310.0},
    }
    print(summarise_state(test_state))
