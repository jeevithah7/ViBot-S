import sys, traceback
from camera_view.robot_camera import RobotCamera

print("Starting custom camera test...")
cam = RobotCamera()
walls = [(1,1,2,1), (2,1,2,2)]
try:
    frame = cam.render(5.0, 5.0, 0.0, walls)
    print("Success rendered size:", frame.shape)
except Exception as e:
    traceback.print_exc()
