"""
astar_planner.py
----------------
A* (A-star) shortest-path planner on a 2-D occupancy grid.

A* combines:
  • g(n) – actual cost from start to node n
  • h(n) – admissible heuristic (here: octile distance or Euclidean distance)

Both 4-connected (NESW) and 8-connected (diagonal) grids are supported.
Only cells marked FREE (0) are treated as traversable.

Reference: Hart, Nilsson & Raphael (1968).
"""

import heapq
import numpy as np
import math
from mapping.occupancy_grid import OccupancyGrid, FREE


# ─── HEURISTICS ───────────────────────────────────────────────────────────────

def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Straight-line distance between two grid cells."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    """
    Octile distance — admissible for 8-connected grids.
    Cheaper to compute than Euclidean while remaining admissible.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Manhattan distance — admissible for 4-connected grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ─── A* PLANNER ───────────────────────────────────────────────────────────────
class AStarPlanner:
    """
    A* path planner for a 2-D occupancy grid.

    Parameters
    ----------
    grid_map : OccupancyGrid
        The map to plan on.
    diagonal : bool
        Allow diagonal moves (8-connected) if True, else 4-connected.
    heuristic : callable
        Distance heuristic h(a, b) → float.  Defaults to octile (diagonal)
        or manhattan (4-connected).
    """

    def __init__(
        self,
        grid_map: OccupancyGrid,
        diagonal: bool = True,
        heuristic=None,
    ):
        self._map      = grid_map
        self._diagonal = diagonal

        if heuristic is not None:
            self._h = heuristic
        else:
            self._h = octile if diagonal else manhattan

        # Build neighbour offsets
        self._dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self._dirs8 = self._dirs4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # ------------------------------------------------------------------ public
    def plan(
        self,
        start: tuple[int, int],
        goal:  tuple[int, int],
    ) -> list[tuple[int, int]]:
        """
        Find the shortest FREE path from start to goal using A*.

        Parameters
        ----------
        start : (row, col)
        goal  : (row, col)

        Returns
        -------
        list of (row, col) tuples from start (inclusive) to goal (inclusive),
        or an empty list if no path exists.
        """
        if not self._is_free(start):
            print(f"[AStarPlanner] Start {start} is not a free cell.")
            return []
        if not self._is_free(goal):
            print(f"[AStarPlanner] Goal {goal} is not a free cell.")
            return []

        # open_set: min-heap of (f_score, g_score, node)
        open_set: list[tuple[float, float, tuple]] = []
        heapq.heappush(open_set, (0.0, 0.0, start))

        came_from: dict[tuple, tuple] = {}
        g_score:   dict[tuple, float] = {start: 0.0}
        # f_score = g + h (only needed for heap priority; stored inside heap)

        dirs = self._dirs8 if self._diagonal else self._dirs4

        while open_set:
            f_cur, g_cur, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct(came_from, current)

            # Skip if we've already found a cheaper path to this node
            if g_cur > g_score.get(current, float("inf")):
                continue

            for dr, dc in dirs:
                nxt = (current[0] + dr, current[1] + dc)

                if not self._is_free(nxt):
                    continue

                # Diagonal steps cost √2; cardinal steps cost 1
                step_cost = math.sqrt(2) if abs(dr) == abs(dc) == 1 else 1.0
                tentative_g = g_cur + step_cost

                if tentative_g < g_score.get(nxt, float("inf")):
                    g_score[nxt]  = tentative_g
                    came_from[nxt] = current
                    f_nxt = tentative_g + self._h(nxt, goal)
                    heapq.heappush(open_set, (f_nxt, tentative_g, nxt))

        print("[AStarPlanner] No path found.")
        return []

    # ------------------------------------------------------------------ helpers
    def _is_free(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return (
            0 <= r < self._map.rows
            and 0 <= c < self._map.cols
            and self._map.get_cell(r, c) == FREE
        )

    @staticmethod
    def _reconstruct(
        came_from: dict,
        current:   tuple,
    ) -> list[tuple[int, int]]:
        """Walk back through came_from to build the path."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------ display
    def plot_plan(
        self,
        start: tuple[int, int],
        goal:  tuple[int, int],
        path:  list[tuple[int, int]],
    ):
        """Visualise the grid, start, goal, and the computed path."""
        import matplotlib.pyplot as plt

        self._map.set_robot_position(*start)
        fig, ax = plt.subplots(figsize=(8, 8))
        self._map.plot(path=path, title="A* Path Plan", ax=ax)
        # Override start/goal labels for clarity
        ax.scatter(goal[1],  goal[0],  c="orange", s=120, marker="*",
                   zorder=7, label="Goal")
        ax.scatter(start[1], start[0], c="lime",   s=120, marker="o",
                   zorder=7, label="Start")
        plt.tight_layout()
        plt.show()


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from mapping.occupancy_grid import OccupancyGrid

    og    = OccupancyGrid.build_sample_map(rows=20, cols=20)
    planner = AStarPlanner(og, diagonal=True)

    start = (1,  1)
    goal  = (18, 18)

    print(f"[astar_planner] Planning from {start} to {goal} …")
    path = planner.plan(start, goal)

    if path:
        print(f"  Path length : {len(path)} steps")
        print(f"  First 5 steps : {path[:5]}")
        planner.plot_plan(start, goal, path)
    else:
        print("  [!] No path found.")
