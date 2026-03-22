import sys, os, time, threading, webbrowser, math

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dashboard.app import app, get_robot, _get_camera, _get_imu

def main():
    print("=====================================================")
    print("  VIBOT-S: REAL-TIME ROBOTICS SIMULATION SYSTEM      ")
    print("=====================================================")
    print(" [*] Initializing Visual Odometry (Essential Matrix)...")
    print(" [*] Starting Real-time SLAM (Occupancy Grid)...")
    print(" [*] Running A* Path Planner...")
    print(" [*] Launching Multi-Panel Dash (Camera/3D/Map/PID)...")
    
    # Pre-initialize singletons
    robot = get_robot()
    _get_camera()
    _get_imu()
    
    # Robot Pos Logger
    def status_logger():
        while True:
            time.sleep(1.5)
            state = robot.read_state()
            x = state.get('smooth_x', 0)
            y = state.get('smooth_y', 0)
            th = state.get('smooth_theta', 0)
            print(f" [SIM] X={x:.2f}, Y={y:.2f}, TH={math.degrees(th):.1f}° | VO: ACTIVE | SLAM: UPDATING")
            
    threading.Thread(target=status_logger, daemon=True).start()

    def open_browser():
        time.sleep(2)
        print("\n -> Dashboard ready at http://127.0.0.1:5000\n")
        webbrowser.open("http://127.0.0.1:5000")
        
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start Flask
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
