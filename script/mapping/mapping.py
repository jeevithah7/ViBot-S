import numpy as np
import matplotlib.pyplot as plt

def create_map(trajectory):
    map_img = np.zeros((500, 500))

    for x, z in trajectory:
        px = int(x * 50 + 250)
        pz = int(z * 50 + 250)

        if 0 <= px < 500 and 0 <= pz < 500:
            map_img[pz, px] = 255

    plt.imshow(map_img, cmap='gray')
    plt.title("Map")
    plt.show()