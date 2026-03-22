"""
live_camera.py
--------------
Real-time camera frame generator for ViBot-S dashboard.

Produces JPEG-encoded frames (as bytes) suitable for:
  • Flask streaming (multipart/x-mixed-replace)
  • Base64 embedding in JSON API responses

Each frame includes:
  - A synthetic indoor-scene corridor
  - Random dynamic elements (simulated moving objects)
  - ORB keypoint overlay
  - Frame counter, timestamp, and FPS badge

Mode switch
-----------
    MODE = "simulation"  – generate frames with NumPy + OpenCV
    MODE = "hardware"    – read from Raspberry Pi / USB camera
"""

import cv2
import numpy as np
import time
import base64
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vision.camera_capture import SimulatedCamera
from vision.feature_detection_orb import ORBDetector
from vision.feature_matching import FeatureMatcher

# ─── MODE SWITCH ──────────────────────────────────────────────────────────────
MODE = "simulation"   # "simulation"  |  "hardware"
# ──────────────────────────────────────────────────────────────────────────────

_FONT     = cv2.FONT_HERSHEY_SIMPLEX
_CYAN     = (0, 255, 255)
_GREEN    = (0, 255, 80)
_YELLOW   = (0, 215, 255)
_RED      = (0, 60, 255)
_DARK_BG  = (14, 17, 23)


class LiveCamera:
    """
    Continuously produces annotated camera frames with ORB overlay.

    Parameters
    ----------
    width      : int    Frame width in pixels  (default 640)
    height     : int    Frame height in pixels (default 360)
    fps_target : float  Target frame rate       (default 15)
    n_features : int    Max ORB keypoints       (default 400)
    jpeg_quality : int  JPEG compression quality 1-100 (default 80)
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        fps_target: float = 15.0,
        n_features: int = 400,
        jpeg_quality: int = 80,
    ):
        self._w  = width
        self._h  = height
        self._dt = 1.0 / fps_target
        self._q  = jpeg_quality
        self._frame_idx = 0
        self._t0 = time.time()
        self._rng = np.random.default_rng(seed=99)
        self._detector = ORBDetector(n_features=n_features)
        self._matcher = FeatureMatcher(ratio_threshold=0.8)
        self._prev_kp = None
        self._prev_des = None

        # Shared state for last keypoints (for API)
        self._last_kp_count = 0
        self._last_match_count = 0

    # ─── frame generation ─────────────────────────────────────────────────────
    def _make_corridor_frame(self, t: float) -> np.ndarray:
        """Render a dynamic synthetic corridor with perspective and motion."""
        frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)

        # ── background gradient (floor→ceiling) ──────────────────────────────
        for y in range(self._h):
            ratio = y / self._h
            r  = int(10 + 25  * ratio)
            g  = int(10 + 30  * ratio)
            b  = int(20 + 60  * ratio)
            frame[y, :] = (b, g, r)

        # ── perspective corridor lines ────────────────────────────────────────
        cx, cy = self._w // 2, self._h // 2
        speed  = t * 40                         # simulated forward motion offset

        for i in range(-4, 5):
            # vertical vanishing lines
            x_end   = cx + i * 60
            x_start = cx + i * 8
            if 0 <= x_start < self._w and 0 <= x_end < self._w:
                cv2.line(frame, (x_start, cy - 10), (x_end, 0),
                         (60, 60, 80), 1)
                cv2.line(frame, (x_start, cy + 10), (x_end, self._h),
                         (60, 60, 80), 1)

        # ── animated horizontal scan lines (corridor depth layers) ────────────
        for k in range(3):
            y_off = int((t * 30 + k * 40) % self._h)
            alpha_row = max(0, min(255, int(60 - k * 15)))
            if 0 <= y_off < self._h:
                frame[y_off, :] = np.clip(
                    frame[y_off, :].astype(int) + alpha_row, 0, 255
                ).astype(np.uint8)

        # ── texture patches (repeatable features for ORB) ─────────────────────
        rng2 = np.random.default_rng(seed=7)
        shift = int(t * 18) % self._w
        for _ in range(60):
            cx_ = int(rng2.integers(10, self._w  - 10))
            cy_ = int(rng2.integers(10, self._h - 10))
            cx_ = (cx_ + shift) % self._w
            rad = int(rng2.integers(3, 10))
            col = tuple(int(c) for c in rng2.integers(80, 230, 3))
            cv2.circle(frame, (cx_, cy_), rad, col, -1)

        # ── moving "obstacle" blobs ───────────────────────────────────────────
        for k in range(2):
            bx = int(self._w * (0.25 + 0.5 * k) +
                     self._w * 0.04 * np.sin(t * 0.7 + k * 1.5))
            by = int(self._h * 0.55 + self._h * 0.03 * np.cos(t * 0.9 + k))
            cv2.ellipse(frame, (bx, by), (28, 38), 0, 0, 360,
                        (40, 80, 40), -1)

        # ── Gaussian noise for realism ────────────────────────────────────────
        noise = self._rng.integers(-12, 12, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame

    def _draw_orb_overlay(
        self, frame: np.ndarray, keypoints: list, prev_kp: list | None = None, matches: list | None = None
    ) -> np.ndarray:
        """Draw ORB keypoints with styled circles and motion trails on the frame."""
        out = frame.copy()
        
        # Draw motion trails (optical flow visualization)
        if prev_kp and matches:
            for m in matches:
                pt1 = keypoints[m.trainIdx].pt
                pt2 = prev_kp[m.queryIdx].pt
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                cv2.line(out, (x2, y2), (x1, y1), _YELLOW, 1, cv2.LINE_AA)

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r     = max(3, int(kp.size / 2))
            # outer glow
            cv2.circle(out, (x, y), r + 2, (0, 200, 100), 1, cv2.LINE_AA)
            # centre dot
            cv2.circle(out, (x, y), 2, _CYAN, -1, cv2.LINE_AA)
        return out

    def _draw_hud(
        self, frame: np.ndarray, kp_count: int, fps: float, t: float, match_count: int = 0
    ) -> np.ndarray:
        """Draw HUD overlay with stats, mode badge, and scan-line effect."""
        out = frame.copy()

        # ── top-left info panel ───────────────────────────────────────────────
        panel_lines = [
            f"ViBot-S  |  SIM CAM",
            f"Frame : {self._frame_idx:05d}",
            f"ORB   : {kp_count} kp",
            f"Tracks: {match_count}",
            f"FPS   : {fps:.1f}",
        ]
        for i, line in enumerate(panel_lines):
            y_pos = 18 + i * 18
            cv2.putText(out, line, (8, y_pos), _FONT, 0.40,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, line, (8, y_pos), _FONT, 0.40,
                        _CYAN, 1, cv2.LINE_AA)

        # ── bottom-right timestamp ────────────────────────────────────────────
        ts = f"T={t:.1f}s"
        cv2.putText(out, ts,
                    (self._w - 90, self._h - 10),
                    _FONT, 0.38, _YELLOW, 1, cv2.LINE_AA)

        # ── "LIVE" badge ──────────────────────────────────────────────────────
        blink = int(t * 2) % 2 == 0
        if blink:
            cv2.rectangle(out, (self._w - 62, 5), (self._w - 5, 24),
                          (0, 0, 200), -1)
            cv2.putText(out, "● REC",
                        (self._w - 58, 19), _FONT, 0.40,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # ── border frame ──────────────────────────────────────────────────────
        cv2.rectangle(out, (0, 0), (self._w - 1, self._h - 1),
                      _CYAN, 1)

        return out

    # ─── public API ───────────────────────────────────────────────────────────
    def get_frame_jpeg(self) -> bytes:
        """Return a single annotated JPEG frame as raw bytes."""
        t   = time.time() - self._t0
        fps = self._frame_idx / max(t, 0.001)

        raw = self._make_corridor_frame(t)

        # ORB detection on this frame
        kp, des = self._detector.detect_and_compute(raw)
        self._last_kp_count = len(kp)

        matches = None
        if self._prev_des is not None and des is not None:
            matches = self._matcher.match(self._prev_des, des)
            self._last_match_count = len(matches)

        annotated = self._draw_orb_overlay(raw, kp, self._prev_kp, matches)
        annotated = self._draw_hud(annotated, len(kp), fps, t, self._last_match_count)

        self._prev_kp = kp
        self._prev_des = des

        self._frame_idx += 1

        ok, buf = cv2.imencode(
            ".jpg", annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._q]
        )
        return bytes(buf) if ok else b""

    def get_frame_b64(self) -> str:
        """Return a Base64-encoded JPEG frame string (for JSON embedding)."""
        return base64.b64encode(self.get_frame_jpeg()).decode("utf-8")

    def get_feature_stream(self) -> dict:
        """Return latest ORB stats without a full frame encode."""
        return {
            "kp_count"   : self._last_kp_count,
            "frame_idx"  : self._frame_idx,
            "match_count": self._last_match_count,
        }


# ─── HARDWARE CAMERA WRAPPER ──────────────────────────────────────────────────
class HardwareLiveCamera:
    """
    Wraps a real USB / Pi camera with ORB overlay.

    Replace SimulatedCamera with picamera2 on Raspberry Pi.
    """

    def __init__(self, device_index: int = 0, n_features: int = 400):
        self._cap      = cv2.VideoCapture(device_index)
        self._detector = ORBDetector(n_features=n_features)

    def get_frame_jpeg(self) -> bytes:
        ret, frame = self._cap.read()
        if not ret:
            return b""
        kp, _     = self._detector.detect_and_compute(frame)
        annotated = cv2.drawKeypoints(
            frame, kp, None,
            color=(0, 255, 100),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        _, buf = cv2.imencode(".jpg", annotated)
        return bytes(buf)

    def get_frame_b64(self) -> str:
        return base64.b64encode(self.get_frame_jpeg()).decode("utf-8")

    def release(self):
        self._cap.release()


# ─── FACTORY ──────────────────────────────────────────────────────────────────
def get_live_camera(**kwargs):
    if MODE == "simulation":
        return LiveCamera(**kwargs)
    elif MODE == "hardware":
        return HardwareLiveCamera(**kwargs)
    else:
        raise ValueError(f"[get_live_camera] Unknown MODE: '{MODE}'")


# ─── SELF-TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cam = LiveCamera()
    for i in range(5):
        frame_bytes = cam.get_frame_jpeg()
        print(f"Frame {i}: {len(frame_bytes)} bytes, kp={cam._last_kp_count}")
        time.sleep(0.07)
    print("[live_camera] Self-test OK.")
