"""
camera_capture.py
-----------------
Handles camera frame capture.

In SIMULATION mode  : reads frames from a folder of images or generates
                      synthetic frames using NumPy / OpenCV.
In HARDWARE mode    : grabs frames from a Raspberry Pi camera (PiCamera2)
                      or a USB webcam via cv2.VideoCapture.

Toggle the MODE constant to switch between both modes.
"""

import cv2
import numpy as np
import os

# ─── MODE SWITCH ──────────────────────────────────────────────────────────────
MODE = "simulation"   # "simulation"  |  "hardware"
# ──────────────────────────────────────────────────────────────────────────────


# ─── SIMULATION CAMERA ────────────────────────────────────────────────────────
class SimulatedCamera:
    """
    Generates a sequence of synthetic indoor-scene frames.

    Each frame looks like a simple corridor with random 'texture' noise so that
    ORB can find repeatable keypoints across successive frames.
    """

    def __init__(self, width: int = 640, height: int = 480, num_frames: int = 10):
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self._frame_idx = 0
        self._rng = np.random.default_rng(seed=42)   # reproducible noise

    # ------------------------------------------------------------------ helpers
    def _make_frame(self, shift_x: int = 0) -> np.ndarray:
        """
        Render a simple synthetic corridor frame.

        Parameters
        ----------
        shift_x : int
            Horizontal pixel shift to simulate forward robot motion.

        Returns
        -------
        np.ndarray  BGR image  (height × width × 3, uint8)
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # --- background gradient (ceiling / floor) ---
        for row in range(self.height):
            intensity = int(40 + 120 * row / self.height)
            frame[row, :] = (intensity // 2, intensity // 2, intensity)

        # --- corridor walls (vertical stripes) ---
        for col in range(0, self.width, 60):
            shifted = (col + shift_x) % self.width
            cv2.line(frame, (shifted, 0), (shifted, self.height), (180, 180, 180), 2)

        # --- add repeatable texture patches (acts like "features") ---
        rng_state = np.random.default_rng(seed=7)
        for _ in range(80):
            cx = int(rng_state.integers(10, self.width  - 10))
            cy = int(rng_state.integers(10, self.height - 10))
            cx = (cx + shift_x) % self.width
            rad = int(rng_state.integers(4, 14))
            colour = tuple(int(c) for c in rng_state.integers(60, 255, 3))
            cv2.circle(frame, (cx, cy), rad, colour, -1)

        # --- add mild Gaussian noise so ORB gets dense keypoints everywhere ---
        noise = self._rng.integers(-15, 15, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame

    # ------------------------------------------------------------------ public
    def get_frame(self) -> np.ndarray | None:
        """
        Return the next simulated frame, or None when the sequence is over.
        """
        if self._frame_idx >= self.num_frames:
            return None

        shift = self._frame_idx * 15          # 15-px shift per frame
        frame = self._make_frame(shift_x=shift)
        self._frame_idx += 1
        return frame

    def reset(self):
        """Restart the frame sequence from the beginning."""
        self._frame_idx = 0

    def release(self):
        """No-op for simulation; mirrors the cv2.VideoCapture API."""
        pass


# ─── HARDWARE CAMERA (Raspberry Pi / USB) ─────────────────────────────────────
class HardwareCamera:
    """
    Thin wrapper around cv2.VideoCapture for a webcam or Pi camera.

    On a Raspberry Pi, install `picamera2` and replace VideoCapture with the
    PiCamera2 API (see commented block below).
    """

    def __init__(self, device_index: int = 0):
        """
        Parameters
        ----------
        device_index : int
            Camera device ID (0 = first camera).
        """
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[HardwareCamera] Could not open camera at index {device_index}."
            )
        print(f"[HardwareCamera] Camera {device_index} opened successfully.")

    def get_frame(self) -> np.ndarray | None:
        """Read one frame from the camera. Returns None on failure."""
        ret, frame = self._cap.read()
        if not ret:
            print("[HardwareCamera] Failed to capture frame.")
            return None
        return frame

    def release(self):
        """Release the camera device."""
        self._cap.release()
        print("[HardwareCamera] Camera released.")

    # ---- Raspberry Pi Camera stub (PiCamera2) ---------------------------------
    # Uncomment the block below and comment out the cv2.VideoCapture block
    # above to use a native Pi camera.
    #
    # from picamera2 import Picamera2
    #
    # def __init__(self):
    #     self._cam = Picamera2()
    #     self._cam.configure(
    #         self._cam.create_preview_configuration(
    #             main={"format": "BGR888", "size": (640, 480)}
    #         )
    #     )
    #     self._cam.start()
    #
    # def get_frame(self):
    #     return self._cam.capture_array()
    #
    # def release(self):
    #     self._cam.stop()
    # ---------------------------------------------------------------------------


# ─── FACTORY ──────────────────────────────────────────────────────────────────
def get_camera(**kwargs):
    """
    Return the appropriate camera object based on the global MODE switch.

    Parameters
    ----------
    **kwargs
        Forwarded to the constructor of the chosen camera class.

    Returns
    -------
    SimulatedCamera | HardwareCamera
    """
    if MODE == "simulation":
        return SimulatedCamera(**kwargs)
    elif MODE == "hardware":
        return HardwareCamera(**kwargs)
    else:
        raise ValueError(f"[get_camera] Unknown MODE: '{MODE}'. Use 'simulation' or 'hardware'.")


# ─── QUICK SELF-TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    cam = get_camera(num_frames=5)
    print("[camera_capture] Running quick self-test …")
    for i in range(5):
        frame = cam.get_frame()
        if frame is None:
            print("  No frame returned.")
            break
        print(f"  Frame {i}: shape={frame.shape}, dtype={frame.dtype}")
        cv2.imshow(f"Frame {i}", frame)
        cv2.waitKey(400)
    cam.release()
    cv2.destroyAllWindows()
    print("[camera_capture] Self-test complete.")
