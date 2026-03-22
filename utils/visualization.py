"""
visualization.py
----------------
Shared visualization utilities for ViBot-S.

All functions return Matplotlib Figure objects so they can be embedded in
larger dashboards, saved to file, or shown interactively.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2


# ─── COLOUR PALETTE ───────────────────────────────────────────────────────────
DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT    = "#e94560"
CYAN      = "#0f3460"
WHITE     = "#ecf0f1"
GREEN     = "#2ecc71"
YELLOW    = "#f39c12"


# ─── MAP VISUALISATION ────────────────────────────────────────────────────────
def plot_grid_with_path(
    grid: np.ndarray,
    path: list[tuple[int, int]] | None = None,
    robot_pos: tuple[int, int] | None = None,
    title: str = "Occupancy Grid + Path",
) -> plt.Figure:
    """
    Render an occupancy grid with an optional planned path and robot marker.

    Parameters
    ----------
    grid      : (H, W) int8 array  (0=free, 1=occupied, -1=unknown)
    path      : list of (row, col) waypoints
    robot_pos : (row, col) of current robot position
    title     : figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    # Convert grid to float image
    display = np.where(grid == -1, 0.5,
              np.where(grid == 0,  1.0, 0.0)).astype(np.float32)

    ax.imshow(display, cmap="gray", vmin=0, vmax=1, origin="upper")

    if path:
        pr = [p[0] for p in path]
        pc = [p[1] for p in path]
        ax.plot(pc, pr, color=ACCENT, linewidth=2.5, label="Path", zorder=4)
        ax.scatter(pc[0],  pr[0],  c=GREEN,  s=100, zorder=5, label="Start")
        ax.scatter(pc[-1], pr[-1], c=YELLOW, s=100, zorder=5, label="Goal")

    if robot_pos:
        ax.scatter(robot_pos[1], robot_pos[0], c="red",
                   s=140, marker="^", zorder=6, label="Robot")

    ax.set_title(title, color=WHITE, fontsize=13)
    ax.set_xlabel("Column", color=WHITE)
    ax.set_ylabel("Row",    color=WHITE)
    ax.tick_params(colors=WHITE)

    legend = ax.legend(facecolor=CYAN, labelcolor=WHITE, fontsize=9)
    plt.tight_layout()
    return fig


# ─── ORB FEATURE VISUALISATION ────────────────────────────────────────────────
def plot_orb_features(
    image: np.ndarray,
    keypoints,   # list[cv2.KeyPoint]
    title: str = "ORB Keypoints",
) -> plt.Figure:
    """
    Draw ORB keypoints on the image and return a Figure.

    Parameters
    ----------
    image     : BGR numpy array
    keypoints : list of cv2.KeyPoint

    Returns
    -------
    matplotlib.figure.Figure
    """
    annotated = cv2.drawKeypoints(
        image, keypoints, None,
        color=(0, 255, 100),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.imshow(rgb)
    ax.set_title(f"{title}  ({len(keypoints)} features)", color=WHITE, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ─── FEATURE MATCH VISUALISATION ──────────────────────────────────────────────
def plot_feature_matches(
    img1: np.ndarray,
    kp1,
    img2: np.ndarray,
    kp2,
    matches,
    max_matches: int = 60,
    title: str = "Feature Matches",
) -> plt.Figure:
    """
    Side-by-side view of two frames with match lines drawn.

    Parameters
    ----------
    img1, img2    : BGR frames
    kp1, kp2      : keypoints from each frame
    matches       : list[cv2.DMatch]
    max_matches   : how many to draw
    title         : figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    drawn = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:max_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.imshow(rgb)
    ax.set_title(f"{title}  ({len(matches)} good matches)", color=WHITE, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ─── PID SIGNAL PLOT ──────────────────────────────────────────────────────────
def plot_pid_response(
    tilt_history: list[float],
    command_history: list[float],
    dt: float = 0.01,
    title: str = "PID Balancing Response",
) -> plt.Figure:
    """
    Plot tilt angle and motor command over time.

    Parameters
    ----------
    tilt_history    : sequence of tilt measurements (degrees)
    command_history : sequence of PID output (motor PWM)
    dt              : time step in seconds
    title           : figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    t_axis = [i * dt for i in range(len(tilt_history))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=WHITE)

    ax1.plot(t_axis, tilt_history, color=ACCENT, linewidth=2, label="Tilt (°)")
    ax1.axhline(0, color=GREEN, linestyle="--", linewidth=1, label="Setpoint")
    ax1.set_ylabel("Tilt (°)", color=WHITE)
    ax1.legend(facecolor=CYAN, labelcolor=WHITE, fontsize=9)
    ax1.set_title(title, color=WHITE, fontsize=12)

    ax2.plot(t_axis, command_history, color="#3498db", linewidth=1.5, label="Motor command")
    ax2.set_ylabel("Motor command", color=WHITE)
    ax2.set_xlabel("Time (s)",      color=WHITE)
    ax2.legend(facecolor=CYAN, labelcolor=WHITE, fontsize=9)

    plt.tight_layout()
    return fig


# ─── VO TRAJECTORY PLOT ───────────────────────────────────────────────────────
def plot_vo_trajectory(
    trajectory: list[tuple[float, float]],
    title: str = "Visual Odometry Trajectory",
) -> plt.Figure:
    """
    Plot the 2-D trajectory estimated by VisualOdometry.

    Parameters
    ----------
    trajectory : list of (x, z) world-frame positions
    title      : figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    xs = [p[0] for p in trajectory]
    zs = [p[1] for p in trajectory]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    ax.plot(xs, zs, color=ACCENT, linewidth=2, marker=".", markersize=8, label="Path")
    ax.scatter(xs[0],  zs[0],  c=GREEN,  s=120, zorder=5, label="Start")
    ax.scatter(xs[-1], zs[-1], c=YELLOW, s=120, zorder=5, label="End")

    ax.set_xlabel("X (normalised m)", color=WHITE)
    ax.set_ylabel("Z (normalised m)", color=WHITE)
    ax.set_title(title, color=WHITE, fontsize=12)
    ax.tick_params(colors=WHITE)
    ax.legend(facecolor=CYAN, labelcolor=WHITE, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()
    return fig


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from mapping.occupancy_grid import OccupancyGrid

    og   = OccupancyGrid.build_sample_map()
    path = [(1, 1), (3, 3), (5, 5), (7, 7), (10, 10)]
    fig  = plot_grid_with_path(og.grid, path=path, robot_pos=(1, 1))
    plt.show()
    print("[visualization] Self-test complete.")
