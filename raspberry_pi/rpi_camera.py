"""
rpi_camera.py
-------------
Raspberry Pi Camera placeholder module.

Swap this in place of camera_capture.SimulatedCamera when running on
real hardware.  Requires:
    pip install picamera2

Usage
-----
    from raspberry_pi.rpi_camera import PiCamera
    cam = PiCamera(width=640, height=480)
    frame = cam.get_frame()
    cam.release()
"""

# NOTE: Uncomment after installing picamera2 on the Raspberry Pi.

class PiCamera:
    """
    Wrapper around PiCamera2 for capturing BGR frames.

    Parameters
    ----------
    width  : int  Frame width  in pixels (default 640).
    height : int  Frame height in pixels (default 480).
    fps    : int  Target frames per second (default 30).
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._cam   = None

        # ── Uncomment on Raspberry Pi ──────────────────────────────────────
        # from picamera2 import Picamera2
        # self._cam = Picamera2()
        # config = self._cam.create_preview_configuration(
        #     main={"format": "BGR888", "size": (width, height)},
        #     controls={"FrameRate": fps},
        # )
        # self._cam.configure(config)
        # self._cam.start()
        # print(f"[PiCamera] Started: {width}×{height} @ {fps} fps")
        # ──────────────────────────────────────────────────────────────────

        print("[PiCamera] PLACEHOLDER – running in stub mode.")
        print("  Uncomment the picamera2 block above when on real hardware.")

    def get_frame(self):
        """
        Capture and return one BGR frame.

        Returns
        -------
        np.ndarray | None
            BGR image array, or None if stub / error.
        """
        if self._cam is None:
            print("[PiCamera] get_frame() called in stub mode – returning None.")
            return None

        # return self._cam.capture_array()   # ← uncomment on Pi

    def release(self):
        """Stop the camera and free resources."""
        if self._cam is not None:
            pass  # self._cam.stop()    ← uncomment on Pi
        print("[PiCamera] Camera released.")
