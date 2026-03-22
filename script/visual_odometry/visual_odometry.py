import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time

start_time = time.time()



def run_visual_odometry(data_path):
    images = sorted(os.listdir(data_path))
    images = [os.path.join(data_path, img) for img in images]

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    pose = np.eye(4)
    trajectory = []

    for i in range(len(images) - 1):
        img1 = cv2.imread(images[i], 0)
        img2 = cv2.imread(images[i+1], 0)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            continue

        matches = bf.match(des1, des2)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, _ = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.))

        if E is None:
            continue

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        pose = pose @ T
        trajectory.append((pose[0, 3], pose[2, 3]))

    trajectory = np.array(trajectory)

    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
    plt.title("Trajectory")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid()
    plt.show()

    return trajectory

if __name__ == "__main__":
    run_visual_odometry("Simulation\data\kitti")
    
    # inside loop (end)
fps = 1.0 / (time.time() - start_time)
start_time = time.time()

print(f"FPS: {fps:.2f}")