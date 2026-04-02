"""
occupancy_grid.py
-----------------
2-D occupancy grid map for indoor robot navigation.

The grid stores one of three possible cell states:
  FREE     = 0   – navigable space
  OCCUPIED = 1   – obstacle
  UNKNOWN  = -1  – not yet observed (grey in visualisations)

The robot pose is tracked separately as (row, col) index into the grid.

This module is self-contained and does not depend on any vision module,
so it can be used standalone or swapped with a SLAM-based map in the future.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
FREE     =  0
OCCUPIED =  1
UNKNOWN  = -1

# Object specific IDs (all > 0 are obstacles to A*)
DESK        = 2
CHAIR       = 3
BED_CYAN    = 4
WEIGHT_BNCH = 5
WATER_COOL  = 6
WARDROBE    = 7
LAMP        = 8
COAT_RACK   = 9


# ─── OCCUPANCY GRID CLASS ─────────────────────────────────────────────────────
class OccupancyGrid:
    """
    2-D occupancy grid.

    Parameters
    ----------
    rows : int      Number of grid rows  (height).
    cols : int      Number of grid cols  (width).
    cell_size : float  Physical size of each cell in metres (default 0.2 m).
    """

    def __init__(self, rows: int = 20, cols: int = 20, cell_size: float = 0.2):
        self.rows      = rows
        self.cols      = cols
        self.cell_size = cell_size
        # Initialise all cells as UNKNOWN
        self._grid: np.ndarray = np.full((rows, cols), UNKNOWN, dtype=np.int8)
        # Robot position in grid coordinates (row, col)
        self.robot_pos: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------ access
    @property
    def grid(self) -> np.ndarray:
        """Read-only view of the internal grid array."""
        return self._grid.copy()

    def set_cell(self, row: int, col: int, state: int):
        """Set a single cell state.  Bounds-checked."""
        if self._in_bounds(row, col):
            self._grid[row, col] = state
        else:
            print(f"[OccupancyGrid] set_cell: ({row},{col}) is out of bounds.")

    def get_cell(self, row: int, col: int) -> int:
        """Return cell state or UNKNOWN if out of bounds."""
        if self._in_bounds(row, col):
            return int(self._grid[row, col])
        return UNKNOWN

    def is_free(self, row: int, col: int) -> bool:
        """True if the cell is navigable."""
        return self.get_cell(row, col) == FREE

    # ------------------------------------------------------------------ bulk
    def mark_free_area(self, r_start: int, r_end: int, c_start: int, c_end: int):
        """Mark a rectangular region as FREE."""
        rs = max(r_start, 0);  re = min(r_end, self.rows)
        cs = max(c_start, 0);  ce = min(c_end, self.cols)
        self._grid[rs:re, cs:ce] = FREE

    def mark_obstacle(self, r_start: int, r_end: int, c_start: int, c_end: int, val: int = OCCUPIED):
        """Mark a rectangular region as OCCUPIED (or a specific object ID)."""
        rs = max(r_start, 0);  re = min(r_end, self.rows)
        cs = max(c_start, 0);  ce = min(c_end, self.cols)
        self._grid[rs:re, cs:ce] = val

    # ------------------------------------------------------------------ robot
    def set_robot_position(self, row: int, col: int):
        """Update the robot's grid position (does not modify the map)."""
        if self._in_bounds(row, col):
            self.robot_pos = (row, col)
        else:
            print(f"[OccupancyGrid] set_robot_position: ({row},{col}) out of bounds.")

    # ------------------------------------------------------------------ helpers
    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    # ------------------------------------------------------------------ factory
    @classmethod
    def build_sample_map(cls, rows: int = 30, cols: int = 14) -> "OccupancyGrid":
        og = cls(rows=rows, cols=cols, cell_size=0.15)
        og.mark_free_area(0, rows, 0, cols)

        # Hallway perimeter walls
        og.mark_obstacle(0, rows, 0, 1)
        og.mark_obstacle(0, rows, cols-1, cols)
        og.mark_obstacle(0, 1, 0, cols)
        og.mark_obstacle(rows-1, rows, 0, cols)

        # Obstacles (Chairs in the hallway)
        og.mark_obstacle(12, 14, 5, 9)
        og.mark_obstacle(20, 22, 2, 6)

        og.set_robot_position(2, 7)
        return og

    @classmethod
    def build_house_map(cls, rows: int = 28, cols: int = 34) -> "OccupancyGrid":
        """Builds a map matching the small gym/office room layout."""
        og = cls(rows=rows, cols=cols, cell_size=0.15)
        og.mark_free_area(0, rows, 0, cols)

        # 1. Outer Perimeter Walls
        og.mark_obstacle(0, 1, 0, cols)        # Top wall
        og.mark_obstacle(rows-2, rows, 0, cols) # Bottom wall
        og.mark_obstacle(0, rows, 0, 2)        # Left wall
        og.mark_obstacle(0, rows, cols-2, cols) # Right wall

        # 2. Objects
        # Desk (Left wall)
        og.mark_obstacle(14, 19, 2, 13, DESK)
        # Office Chair
        og.mark_obstacle(18, 22, 12, 16, CHAIR)
        # Cyan Bed (Bottom left)
        og.mark_obstacle(23, 27, 2, 16, BED_CYAN)
        
        # Weight Bench & Barbell (Center-top right)
        og.mark_obstacle(12, 17, 18, 26, WEIGHT_BNCH)

        # Wardrobe (Right wall)
        og.mark_obstacle(8, 20, 28, 32, WARDROBE)
        # Water Cooler
        og.mark_obstacle(20, 23, 29, 32, WATER_COOL)
        # Floor Lamp
        og.mark_obstacle(24, 26, 26, 29, LAMP)

        # Coat Rack (Top left corner)
        og.mark_obstacle(2, 5, 2, 5, COAT_RACK)

        og.set_robot_position(24, 21) # Safe start zone
        return og

    # ------------------------------------------------------------------ display
    def plot(
        self,
        path: list[tuple[int, int]] | None = None,
        title: str = "Occupancy Grid",
        ax: plt.Axes | None = None,
    ):
        """
        Render the map with optional path overlay.

        Parameters
        ----------
        path  : list of (row, col) tuples from A* or similar planner.
        title : figure / axes title.
        ax    : Matplotlib Axes to draw on; a new figure is created if None.
        """
        show = ax is None
        if show:
            fig, ax = plt.subplots(figsize=(7, 7))

        # Build display array: unknown=-1 → 0.5, free=0 → 1.0, occupied=1 → 0.0
        display = np.where(self._grid == UNKNOWN, 0.5,
                  np.where(self._grid == FREE,    1.0, 0.0))

        ax.imshow(display, cmap="gray", vmin=0, vmax=1, origin="upper")

        # Draw planned path
        if path:
            pr = [p[0] for p in path]
            pc = [p[1] for p in path]
            ax.plot(pc, pr, "b-", linewidth=2, label="Path")
            ax.scatter(pc[0],  pr[0],  c="lime",   s=80, zorder=5, label="Start")
            ax.scatter(pc[-1], pr[-1], c="orange",  s=80, zorder=5, label="Goal")

        # Draw robot
        rr, rc = self.robot_pos
        ax.scatter(rc, rr, c="red", s=120, marker="^", zorder=6, label="Robot")

        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Legend patches
        patches = [
            mpatches.Patch(color="white", label="Free"),
            mpatches.Patch(color="black", label="Obstacle"),
            mpatches.Patch(color="gray",  label="Unknown"),
        ]
        ax.legend(handles=patches + ([ax.lines[0]] if path else []),
                  loc="upper right", fontsize=8)

        if show:
            plt.tight_layout()
            plt.show()


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    og = OccupancyGrid.build_sample_map()
    print(f"[occupancy_grid] Grid shape: {og.grid.shape}")
    print(f"  Robot at: {og.robot_pos}")
    print(f"  Cell (1,1) is free: {og.is_free(1, 1)}")
    print(f"  Cell (4,4) is free: {og.is_free(4, 4)}")
    og.plot(title="Sample Indoor Map")
