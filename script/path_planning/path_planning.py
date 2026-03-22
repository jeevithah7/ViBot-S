import numpy as np
import matplotlib.pyplot as plt
import heapq

def astar(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)

            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor] == 1:
                    continue

                temp_g = g_score[current] + 1

                if neighbor not in g_score or temp_g < g_score[neighbor]:
                    g_score[neighbor] = temp_g
                    f = temp_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current

    return []

def demo_path():
    grid = np.zeros((50,50))
    grid[20:30, 25] = 1

    start = (10,10)
    goal = (40,40)

    path = astar(grid, start, goal)

    plt.imshow(grid, cmap='gray')

    for p in path:
        plt.scatter(p[1], p[0], c='red', s=5)

    plt.title("A* Path")
    plt.show()

if __name__ == "__main__":
    demo_path()