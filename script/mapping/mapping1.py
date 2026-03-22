import numpy as np
import matplotlib.pyplot as plt

def create_map(trajectory):
    # Create map
    map_img = np.zeros((500, 500))

    x_vals = []
    z_vals = []

    for x, z in trajectory:
        # Scale and center
        px = int(x * 50 + 250)
        pz = int(z * 50 + 250)

        if 0 <= px < 500 and 0 <= pz < 500:
            map_img[pz, px] = 255
            x_vals.append(px)
            z_vals.append(pz)

    # Plot map
    plt.figure(figsize=(6, 6))
    plt.imshow(map_img, cmap='gray')

    # Plot trajectory line
    plt.plot(x_vals, z_vals, color='blue', linewidth=2, label="Trajectory")

    # Mark start point
    if len(x_vals) > 0:
        plt.scatter(x_vals[0], z_vals[0], c='green', s=50, label="Start")

    # Mark end point
    if len(x_vals) > 0:
        plt.scatter(x_vals[-1], z_vals[-1], c='red', s=50, label="End")

    plt.title("2D Map from Visual Odometry")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.grid()

    plt.show()
    