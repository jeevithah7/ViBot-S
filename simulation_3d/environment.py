import numpy as np

class Environment3D:
    def __init__(self, grid):
        """grid is a 2D numpy array where 1 == obstacle."""
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.walls = self._extract_vector_walls()
        
    def _extract_vector_walls(self):
        """
        Convert obstacle cells into LONG merged wall segments (vectorized).
        Instead of 1x1 boxes, we extract horizontal and vertical spans into 
        continuous line segments (x1, y1, x2, y2).
        """
        walls = []
        
        # Horizontal spans
        for r in range(self.rows):
            c = 0
            while c < self.cols:
                if self.grid[r, c] == 1:
                    start_c = c
                    while c < self.cols and self.grid[r, c] == 1:
                        c += 1
                    # Found a horizontal span of length (c - start_c)
                    if c - start_c > 1:
                        # Only add as horizontal wall if it's a span, not just a point
                        walls.append((start_c, r, c, r))
                        walls.append((start_c, r+1, c, r+1))
                else:
                    c += 1
        
        # Vertical spans
        for c in range(self.cols):
            r = 0
            while r < self.rows:
                if self.grid[r, c] == 1:
                    start_r = r
                    while r < self.rows and self.grid[r, c] == 1:
                        r += 1
                    # Found a vertical span of length (r - start_r)
                    if r - start_r > 0:
                        walls.append((c, start_r, c, r))
                        walls.append((c+1, start_r, c+1, r))
                else:
                    r += 1
                    
        # Add perimeter boundary
        walls.append((0, 0, self.cols, 0))
        walls.append((self.cols, 0, self.cols, self.rows))
        walls.append((self.cols, self.rows, 0, self.rows))
        walls.append((0, self.rows, 0, 0))
                    
        return walls
