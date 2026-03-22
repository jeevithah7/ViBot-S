import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time

def run_visual_odometry(data_path):
    images = sorted(os.listdir(data_path))
    images = [os.path.join(data_path, img) for img in images]

    if len(images) < 2:
        print("Not enough images")
        return []

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    pose = np.eye(4)
    trajectory = []

    prev_time = time.time()

    print("Running Visual Odometry...")

    for i in range(len(images) - 1):
        img1 = cv2.imread(images[i], 0)
        img2 = cv2.imread(images[i+1], 0)

        if img1 is None or img2 is None:
            print(f"Skipping frame {i}")
            continue

        # Detect features
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            continue

        # Match features
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:100]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Essential matrix
        E, _ = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.))

        if E is None:
            continue

        # Recover pose
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2)

        # Transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        # Update pose
        pose = pose @ T

        # Save trajectory (X, Z)
        trajectory.append((pose[0, 3], pose[2, 3]))

        # -------- FPS FIX --------
        current_time = time.time()
        delta = current_time - prev_time

        if delta > 0:
            fps = 1.0 / delta
        else:
            fps = 0

        prev_time = current_time

        print(f"Frame {i} | FPS: {fps:.2f}")

    # Convert to numpy
    trajectory = np.array(trajectory)

    # -------- PLOT --------
    plt.figure(figsize=(6, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='blue')

    # Add direction arrows
    for i in range(1, len(trajectory)):
        plt.arrow(
            trajectory[i-1, 0], trajectory[i-1, 1],
            trajectory[i, 0] - trajectory[i-1, 0],
            trajectory[i, 1] - trajectory[i-1, 1],
            head_width=0.05,
            color='blue'
        )

    plt.title("Visual Odometry Trajectory")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid()
    plt.show()

    return trajectory


if __name__ == "__main__":
    run_visual_odometry("Simulation/data/kitti")