import sys, os, time, threading, webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dashboard.app import app, get_robot, _get_camera, _get_imu
from camera_view.robot_camera import RobotCamera
from mapping_live.slam_visualizer import SlamVisualizer
from simulation_3d.renderer import Renderer3D

def main():
    print("=====================================================")
    print("  ViBot-S FULL VISUAL ROBOT SIMULATION (3D SYSTEM)   ")
    print("=====================================================")
    print(" 1. Spawning robot in 3D environment...")
    print(" 2. Starting 1st-person camera simulation...")
    print(" 3. Running ORB feature detection & tracking...")
    print(" 4. Building real-time SLAM map visualization...")
    print(" 5. Executing A* path planning...")
    print(" 6. Launching Web Dashboard...")
    
    # Pre-warm
    robot = get_robot()
    _get_camera()
    _get_imu()
    
    # Optional console status output
    def status_logger():
        while True:
            time.sleep(2)
            state = robot.read_state()
            x, y, th = state.get('smooth_x', 0), state.get('smooth_y', 0), state.get('smooth_theta', 0)
            print(f" [SIM] Robot moving ... Pos: ({x:.2f}, {y:.2f}) Yaw: {math.degrees(th):.1f}°")
            
    import math
    threading.Thread(target=status_logger, daemon=True).start()

    def open_browser():
        time.sleep(1.5)
        print("\n -> Opening Multi-Panel Dashboard at http://localhost:5000\n")
        webbrowser.open("http://localhost:5000")
        
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Use standard Flask dev server
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
