"""
visual_odometry.py
------------------
Monocular Visual Odometry (VO) — estimates camera motion between two frames.

Pipeline
--------
1. Detect ORB keypoints in both frames (via ORBDetector).
2. Match descriptors with ratio-test filtering (via FeatureMatcher).
3. Extract 2-D point correspondences from the good matches.
4. Compute the Essential Matrix (E) using RANSAC + the known (or assumed)
   camera intrinsics K.
5. Decompose E into rotation R and translation t.
6. Accumulate pose as a trajectory.

Notes on Scale
--------------
Monocular VO recovers motion up to an *unknown scale factor*.  For a real
robot you need either:
  • A stereo camera, or
  • Known average scene depth / IMU integration.
In this simulation we normalise the translation to unit length and treat it
as a *direction*.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from vision.feature_detection_orb import ORBDetector
from vision.feature_matching import FeatureMatcher


# ─── CAMERA INTRINSICS ────────────────────────────────────────────────────────
# Simulated camera matrix for a 640 × 480 sensor.
# On real hardware, replace these with values from cv2.calibrateCamera().
DEFAULT_K = np.array(
    [
        [525.0,   0.0, 320.0],
        [  0.0, 525.0, 240.0],
        [  0.0,   0.0,   1.0],
    ],
    dtype=np.float64,
)


# ─── VISUAL ODOMETRY CLASS ────────────────────────────────────────────────────
class VisualOdometry:
    """
    Frame-to-frame monocular visual odometry.

    Parameters
    ----------
    camera_matrix : np.ndarray, shape (3, 3)
        Camera intrinsics K.  Defaults to a synthetic 640×480 camera.
    n_features : int
        ORB feature budget per frame.
    ratio : float
        Lowe's ratio-test threshold.
    ransac_prob : float
        RANSAC confidence for Essential Matrix estimation (default 0.999).
    ransac_threshold : float
        Epipolar distance threshold for RANSAC (default 1.0 px).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray = DEFAULT_K,
        n_features: int = 1000,
        ratio: float = 0.75,
        ransac_prob: float = 0.999,
        ransac_threshold: float = 1.0,
    ):
        self.K = camera_matrix
        self._detector = ORBDetector(n_features=n_features)
        self._matcher  = FeatureMatcher(ratio_threshold=ratio)
        self._ransac_prob      = ransac_prob
        self._ransac_threshold = ransac_threshold

        # Accumulated pose (world frame)
        self.R_total = np.eye(3, dtype=np.float64)
        self.t_total = np.zeros((3, 1), dtype=np.float64)

        # History for trajectory display
        self.trajectory: list[tuple[float, float]] = [(0.0, 0.0)]

    # ------------------------------------------------------------------ public
    def process_frame_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> dict:
        """
        Estimate relative motion from frame1 → frame2.

        Parameters
        ----------
        frame1 : np.ndarray  (BGR, H×W×3)
        frame2 : np.ndarray  (BGR, H×W×3)

        Returns
        -------
        dict with keys:
          'R'         – 3×3 rotation matrix
          't'         – 3×1 translation vector (unit length)
          'inliers'   – number of RANSAC inliers
          'pts1','pts2' – matched 2-D points (float32 Nx2)
          'success'   – bool
        """
        # Step 1 & 2: detect + match
        kp1, des1 = self._detector.detect_and_compute(frame1)
        kp2, des2 = self._detector.detect_and_compute(frame2)
        matches    = self._matcher.match(des1, des2)

        result: dict = {
            "R": None, "t": None, "inliers": 0,
            "pts1": None, "pts2": None, "success": False,
        }

        if len(matches) < 8:
            print(f"[VisualOdometry] Not enough matches: {len(matches)} < 8")
            return result

        # Step 3: extract matched 2-D coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Step 4: Essential Matrix via RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            cameraMatrix=self.K,
            method=cv2.RANSAC,
            prob=self._ransac_prob,
            threshold=self._ransac_threshold,
        )

        if E is None or E.shape != (3, 3):
            print("[VisualOdometry] Essential Matrix estimation failed.")
            return result

        # Step 5: Recover R, t
        inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, cameraMatrix=self.K, mask=mask)

        # Accumulate pose
        self.t_total = self.t_total + self.R_total @ t
        self.R_total = R @ self.R_total

        x = float(self.t_total[0])
        z = float(self.t_total[2])   # z = depth axis
        self.trajectory.append((x, z))

        result.update({
            "R": R,
            "t": t,
            "inliers": int(inliers),
            "pts1": pts1,
            "pts2": pts2,
            "success": True,
        })

        print(
            f"[VisualOdometry] R=\n{np.round(R, 3)}  "
            f"t={t.ravel().round(3)}  inliers={inliers}"
        )
        return result

    def reset_pose(self):
        """Reset accumulated pose to the origin."""
        self.R_total = np.eye(3, dtype=np.float64)
        self.t_total = np.zeros((3, 1), dtype=np.float64)
        self.trajectory = [(0.0, 0.0)]

    # ------------------------------------------------------------------ display
    def plot_trajectory(self, title: str = "Visual Odometry Trajectory"):
        """Plot the 2-D (X-Z) trajectory estimated so far."""
        xs = [p[0] for p in self.trajectory]
        zs = [p[1] for p in self.trajectory]

        plt.figure(figsize=(7, 7))
        plt.plot(xs, zs, "b.-", markersize=8, label="Estimated path")
        plt.scatter(xs[0],  zs[0],  c="green", s=100, zorder=5, label="Start")
        plt.scatter(xs[-1], zs[-1], c="red",   s=100, zorder=5, label="End")
        plt.xlabel("X (m, normalised)")
        plt.ylabel("Z (m, normalised)")
        plt.title(title)
        plt.legend()
        plt.axis("equal")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_flow(
        frame1: np.ndarray,
        frame2: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        max_pts: int = 80,
    ) -> np.ndarray:
        """
        Draw optical-flow-style arrows on frame2 showing point movement.

        Returns
        -------
        np.ndarray  BGR image with arrows
        """
        canvas = frame2.copy()
        for i in range(min(len(pts1), max_pts)):
            p1 = tuple(map(int, pts1[i]))
            p2 = tuple(map(int, pts2[i]))
            cv2.arrowedLine(canvas, p1, p2, (0, 255, 0), 1, tipLength=0.3)
            cv2.circle(canvas, p2, 3, (0, 0, 255), -1)
        return canvas


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from vision.camera_capture import SimulatedCamera

    cam = SimulatedCamera(num_frames=6)
    vo  = VisualOdometry()

    prev = cam.get_frame()
    for step in range(5):
        curr = cam.get_frame()
        if curr is None:
            break
        res = vo.process_frame_pair(prev, curr)
        if res["success"] and res["pts1"] is not None:
            flow_img = VisualOdometry.draw_flow(prev, curr, res["pts1"], res["pts2"])
            cv2.imshow(f"Flow step {step}", flow_img)
            cv2.waitKey(400)
        prev = curr

    cv2.destroyAllWindows()
    vo.plot_trajectory()
    print("[visual_odometry] Self-test done.")
