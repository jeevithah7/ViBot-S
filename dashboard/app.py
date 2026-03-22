"""
app.py
------
ViBot-S Real-Time Dashboard – Flask backend.

Endpoints
---------
GET  /               → Serves the main dashboard HTML page
GET  /api/state      → JSON snapshot of full robot + sensor state
GET  /api/camera     → JSON with base64-encoded camera frame + ORB stats
GET  /stream/video   → MJPEG camera stream  (multipart/x-mixed-replace)
GET  /stream/events  → Server-Sent Events for real-time charts / map
POST /api/reset      → Reset the robot to start position
GET  /api/status     → Health-check endpoint

Run
---
    python dashboard/app.py
    # or via demo_mode.py
"""

import sys, os, time, json, threading
from flask import Flask, Response, jsonify, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from realtime.live_robot  import get_robot, reset_robot
from realtime.live_camera import LiveCamera
from realtime.live_imu    import SimulatedIMU
from simulation_3d.top_down_renderer import TopDownRenderer

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Global singletons (initialised on first request or early start)
_camera : LiveCamera   = None
_imu    : SimulatedIMU = None
_sim_view : TopDownRenderer = None
_cam_lock = threading.Lock()
_sim_lock = threading.Lock()


def _get_camera() -> LiveCamera:
    global _camera
    if _camera is None:
        _camera = LiveCamera(width=640, height=360, fps_target=15, n_features=350)
    return _camera


def _get_imu() -> SimulatedIMU:
    global _imu
    if _imu is None:
        _imu = SimulatedIMU()
        _imu.start()
    return _imu


def _get_sim_view() -> TopDownRenderer:
    global _sim_view
    if _sim_view is None:
        _sim_view = TopDownRenderer(width=640, height=360, scale=18)
    return _sim_view


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the 3D dashboard HTML."""
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    """Full robot state JSON."""
    robot = get_robot()
    imu   = _get_imu()
    state = robot.read_state()
    imu_d = imu.read_once()
    state.update({
        "imu": imu_d,
    })
    return jsonify(state)


@app.route("/api/camera")
def api_camera():
    """Single annotated frame as base64 JPEG + ORB stats."""
    with _cam_lock:
        cam  = _get_camera()
        b64  = cam.get_frame_b64()
        stats = cam.get_feature_stream()
    return jsonify({
        "frame_b64": b64,
        "kp_count" : stats["kp_count"],
        "frame_idx": stats["frame_idx"],
    })


@app.route("/stream/video")
def stream_video():
    """MJPEG live stream of the simulated camera."""
    def generate():
        cam = _get_camera()
        while True:
            with _cam_lock:
                frame_bytes = cam.get_frame_jpeg()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes +
                b"\r\n"
            )
            time.sleep(1.0 / 15.0)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/sim_view")
def stream_sim_view():
    """MJPEG live stream of the top-down simulation view."""
    def generate():
        render_obj = _get_sim_view()
        while True:
            robot = get_robot()
            d = robot.read_state()
            rx, ry, rth = d.get('smooth_x', 1.0), d.get('smooth_y', 1.0), d.get('smooth_theta', 0.0)
            grid, path, goal = d.get('grid', []), d.get('path', []), d.get('goal', [27, 7])
            nav_status = d.get('nav_status', 'NAVIGATING')

            with _sim_lock:
                frame = render_obj.render(rx, ry, rth, grid, path, goal, nav_status=nav_status)
                frame_bytes = render_obj.get_jpeg(frame)
                
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes +
                b"\r\n"
            )
            time.sleep(1.0 / 12.0)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/events")
def stream_events():
    """
    Server-Sent Events stream.

    Sends a JSON event every ~200 ms containing robot state + IMU.
    The frontend JavaScript listens to this and updates charts/map.
    """
    def generate():
        while True:
            robot = get_robot()
            imu   = _get_imu()
            state = robot.read_state()
            imu_d = imu.read_once()
            payload = {
                "robot_pos"     : state["robot_pos"],
                "grid"          : state["grid"],
                "path"          : state["path"],
                "pos_log"       : state["pos_log"],
                "current_tilt"  : state["current_tilt"],
                "current_motor" : state["current_motor"],
                "smooth_x"      : state.get("smooth_x", 1.0),
                "smooth_y"      : state.get("smooth_y", 1.0),
                "smooth_theta"  : state.get("smooth_theta", 0.0),
                "tilt_history"  : state["tilt_history"][-60:],
                "motor_history" : state["motor_history"][-60:],
                "time_history"  : state["time_history"][-60:],
                "step_idx"      : state["step_idx"],
                "total_steps"   : state["total_steps"],
                "elapsed"       : state["elapsed"],
                "nav_status"    : state["nav_status"],
                "balance_status": state["balance_status"],
                "loops"         : state["loops"],
                # IMU
                "tilt_angle"    : imu_d["tilt_angle"],
                "accel_x"       : imu_d["accel_x"],
                "accel_y"       : imu_d["accel_y"],
                "accel_z"       : imu_d["accel_z"],
                "gyro_x"        : imu_d["gyro_x"],
                "velocity"      : imu_d["velocity"],
                "motor_left"    : imu_d["motor_left"],
                "motor_right"   : imu_d["motor_right"],
                "battery_pct"   : imu_d["battery_pct"],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.20)   # 5 updates/second

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control" : "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset the robot to START position."""
    reset_robot()
    global _imu
    if _imu:
        _imu.stop()
    _imu = None
    _get_imu()
    return jsonify({"status": "reset", "message": "Robot reset to start."})


@app.route("/api/status")
def api_status():
    return jsonify({
        "status"  : "running",
        "mode"    : "simulation",
        "version" : "2.0",
        "robot"   : "ViBot-S",
    })


@app.route("/api/layers", methods=["POST"])
def api_layers():
    """Toggle renderer layers sent as JSON: {layer: 'map', visible: true}"""
    data = request.get_json(force=True, silent=True) or {}
    layer   = data.get("layer",   "")
    visible = bool(data.get("visible", True))
    r = _get_sim_view()
    attr_map = {
        "map":        "show_map",
        "path":       "show_path",
        "robot":      "show_robot",
        "trajectory": "show_trajectory",
        "fov":        "show_fov",
        "axes":       "show_axes",
        "legend":     "show_legend",
    }
    if layer in attr_map:
        setattr(r, attr_map[layer], visible)
        return jsonify({"ok": True, "layer": layer, "visible": visible})
    return jsonify({"ok": False, "error": "unknown layer"}), 400


# ─── Dashboard HTML (single-file, no external template needed) ────────────────
