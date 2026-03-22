import sys, os, traceback
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from realtime.live_camera import LiveCamera
import time

print("Testing full camera integration...")
cam = LiveCamera()
try:
    frame_bytes = cam.get_frame_jpeg()
    print("Success. Received frame bytes:", len(frame_bytes))
except Exception as e:
    traceback.print_exc()
