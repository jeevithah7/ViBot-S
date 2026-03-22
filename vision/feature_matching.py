"""
feature_matching.py
-------------------
Matches ORB descriptors between two frames using a Brute-Force Matcher.

Brute-Force (BF) matching with Hamming distance is the standard choice for
binary descriptors such as ORB / BRIEF.  Lowe's ratio test filters out
ambiguous matches, producing a cleaner and more reliable correspondence set.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from vision.feature_detection_orb import ORBDetector


# ─── MATCHER CLASS ────────────────────────────────────────────────────────────
class FeatureMatcher:
    """
    Matches ORB descriptors between two consecutive frames.

    Parameters
    ----------
    ratio_threshold : float
        Lowe's ratio-test threshold.  Matches where the best distance is less
        than `ratio_threshold × second-best distance` are kept.
        Typical values: 0.65 – 0.75 (default 0.75).
    """

    def __init__(self, ratio_threshold: float = 0.75):
        # NORM_HAMMING is correct for binary (ORB / BRIEF) descriptors.
        # crossCheck=False because we use knnMatch + ratio test instead.
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_threshold = ratio_threshold

    # ------------------------------------------------------------------ public
    def match(
        self,
        des1: np.ndarray,
        des2: np.ndarray,
    ) -> list[cv2.DMatch]:
        """
        Run KNN matching and apply Lowe's ratio test.

        Parameters
        ----------
        des1 : np.ndarray
            Descriptors from frame 1  (N × 32, uint8).
        des2 : np.ndarray
            Descriptors from frame 2  (M × 32, uint8).

        Returns
        -------
        list[cv2.DMatch]
            Filtered list of good matches, sorted by distance (ascending).
        """
        if des1 is None or des2 is None:
            print("[FeatureMatcher] One or both descriptor arrays are None.")
            return []

        if len(des1) < 2 or len(des2) < 2:
            print("[FeatureMatcher] Not enough descriptors to match.")
            return []

        # k=2 → keep the two nearest neighbours for ratio test
        knn_matches = self._bf.knnMatch(des1, des2, k=2)

        good: list[cv2.DMatch] = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                good.append(m)

        # Sort by Hamming distance (ascending)
        good.sort(key=lambda x: x.distance)
        return good

    # ------------------------------------------------------------------ display
    @staticmethod
    def draw_matches(
        img1: np.ndarray,
        kp1: list[cv2.KeyPoint],
        img2: np.ndarray,
        kp2: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
        max_matches: int = 50,
        use_matplotlib: bool = True,
        title: str = "Feature Matches",
    ) -> np.ndarray:
        """
        Draw matching lines between two frames.

        Parameters
        ----------
        img1, img2    : source BGR frames
        kp1, kp2      : keypoints from each frame
        matches       : list of DMatch objects
        max_matches   : maximum number of matches to draw
        use_matplotlib: True → show with Matplotlib; False → cv2.imshow
        title         : window / figure title

        Returns
        -------
        np.ndarray  side-by-side annotated image (BGR)
        """
        drawn = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            matches[:max_matches],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        if use_matplotlib:
            rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(14, 5))
            plt.imshow(rgb)
            plt.title(f"{title}  ({len(matches)} good matches)")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            cv2.imshow(title, drawn)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return drawn


# ─── PIPELINE HELPER ──────────────────────────────────────────────────────────
def match_frames(
    frame1: np.ndarray,
    frame2: np.ndarray,
    n_features: int = 1000,
    ratio: float = 0.75,
    display: bool = True,
) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch]]:
    """
    Convenience function: detect → match → optionally display.

    Parameters
    ----------
    frame1, frame2 : BGR images to compare
    n_features     : ORB feature budget
    ratio          : Lowe's ratio threshold
    display        : whether to show the matched image

    Returns
    -------
    (kp1, kp2, good_matches)
    """
    detector = ORBDetector(n_features=n_features)
    kp1, des1 = detector.detect_and_compute(frame1)
    kp2, des2 = detector.detect_and_compute(frame2)

    matcher = FeatureMatcher(ratio_threshold=ratio)
    good = matcher.match(des1, des2)

    print(
        f"[match_frames] kp1={len(kp1)}  kp2={len(kp2)}  good_matches={len(good)}"
    )

    if display and len(good) > 0:
        FeatureMatcher.draw_matches(frame1, kp1, frame2, kp2, good)

    return kp1, kp2, good


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from vision.camera_capture import SimulatedCamera

    cam = SimulatedCamera(num_frames=2)
    f1 = cam.get_frame()
    f2 = cam.get_frame()

    kp1, kp2, matches = match_frames(f1, f2, display=True)
    print(f"[feature_matching] Self-test complete. {len(matches)} matches found.")
