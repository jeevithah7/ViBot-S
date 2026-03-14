"""
feature_detection_orb.py
------------------------
ORB (Oriented FAST and Rotated BRIEF) keypoint detection.

ORB is a fast, patent-free feature detector+descriptor well-suited for
real-time robotics.  This module wraps cv2.ORB with sensible defaults and
provides helpers for displaying the detected features.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ─── DETECTOR CLASS ───────────────────────────────────────────────────────────
class ORBDetector:
    """
    Detects ORB keypoints and computes descriptors for a single image.

    Parameters
    ----------
    n_features : int
        Maximum number of features to retain (default 1000).
    scale_factor : float
        Pyramid decimation ratio, > 1 (default 1.2).
    n_levels : int
        Number of pyramid levels (default 8).
    edge_threshold : int
        Border size where features are not detected (default 31).
    """

    def __init__(
        self,
        n_features: int = 1000,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
    ):
        self._orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
        )
        self.n_features = n_features

    # ------------------------------------------------------------------ public
    def detect_and_compute(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """
        Detect ORB keypoints and compute descriptors.

        Parameters
        ----------
        image : np.ndarray
            Input BGR or grayscale image.

        Returns
        -------
        keypoints : list[cv2.KeyPoint]
            Detected keypoint objects.
        descriptors : np.ndarray
            Binary descriptors, shape (N, 32), dtype uint8.
            Returns ([], None) if no features found.
        """
        gray = self._to_gray(image)
        keypoints, descriptors = self._orb.detectAndCompute(gray, mask=None)
        return keypoints, descriptors

    def detect_only(self, image: np.ndarray) -> list[cv2.KeyPoint]:
        """Return only keypoints (no descriptors)."""
        gray = self._to_gray(image)
        return self._orb.detect(gray, mask=None)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert BGR to grayscale if necessary."""
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    # ------------------------------------------------------------------ display
    @staticmethod
    def draw_keypoints(
        image: np.ndarray,
        keypoints: list[cv2.KeyPoint],
        colour: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Overlay detected keypoints on a copy of the image.

        Parameters
        ----------
        image     : BGR source image
        keypoints : list of KeyPoint objects
        colour    : BGR colour for the keypoint circles

        Returns
        -------
        np.ndarray  annotated BGR image
        """
        return cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=colour,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    def display(
        self,
        image: np.ndarray,
        keypoints: list[cv2.KeyPoint],
        title: str = "ORB Features",
        use_matplotlib: bool = True,
    ):
        """
        Show the annotated image using Matplotlib or an OpenCV window.

        Parameters
        ----------
        image           : source BGR image
        keypoints       : detected KeyPoint objects
        title           : window / figure title
        use_matplotlib  : True → Matplotlib  |  False → cv2.imshow
        """
        annotated = self.draw_keypoints(image, keypoints)

        if use_matplotlib:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 5))
            plt.imshow(rgb)
            plt.title(f"{title}  ({len(keypoints)} keypoints)")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            cv2.imshow(title, annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate a tiny synthetic test image
    from vision.camera_capture import SimulatedCamera

    cam = SimulatedCamera(num_frames=1)
    frame = cam.get_frame()

    detector = ORBDetector(n_features=500)
    kp, des = detector.detect_and_compute(frame)
    print(f"[feature_detection_orb] Detected {len(kp)} keypoints.")
    if des is not None:
        print(f"  Descriptor shape : {des.shape}")

    detector.display(frame, kp, title="ORB Self-Test")
