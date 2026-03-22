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
from flask import Flask, Response, jsonify, render_template_string, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from realtime.live_robot  import get_robot, reset_robot
from realtime.live_camera import LiveCamera
from realtime.live_imu    import SimulatedIMU

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Global singletons (initialised on first request or early start)
_camera : LiveCamera   = None
_imu    : SimulatedIMU = None
_cam_lock = threading.Lock()


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


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard HTML (embedded in this file for portability)."""
    return render_template_string(DASHBOARD_HTML)


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
                "pos_log"       : state["pos_log"],
                "current_tilt"  : state["current_tilt"],
                "current_motor" : state["current_motor"],
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


# ─── Dashboard HTML (single-file, no external template needed) ────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ViBot-S | Real-Time Robotics Dashboard</title>
  <meta name="description" content="ViBot-S real-time interactive dashboard – self-balancing vision-based indoor navigation robot simulation."/>

  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;900&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>

  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

  <style>
    /* ── Reset & base ─────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg0       : #050810;
      --bg1       : #0a0d1a;
      --bg2       : #0f1629;
      --bg3       : #162040;
      --accent    : #00e5ff;
      --accent2   : #7c4dff;
      --green     : #00e676;
      --orange    : #ff9100;
      --red       : #ff1744;
      --yellow    : #ffd740;
      --text      : #e0e8f8;
      --text-dim  : #7a90b0;
      --border    : rgba(0,229,255,0.15);
      --glow      : 0 0 18px rgba(0,229,255,0.25);
      --radius    : 12px;
      --font-mono : 'JetBrains Mono', monospace;
      --font-hud  : 'Orbitron', sans-serif;
      --font-body : 'Inter', sans-serif;
    }

    html, body {
      width: 100%; height: 100%;
      background: var(--bg0);
      color: var(--text);
      font-family: var(--font-body);
      font-size: 13px;
      overflow: hidden;
    }

    /* ── Layout ──────────────────────────────────────────────── */
    #app {
      display: grid;
      grid-template-rows: 52px 1fr 180px;
      grid-template-columns: 260px 1fr 1fr 220px;
      height: 100vh;
      gap: 6px;
      padding: 6px;
    }

    /* ── Header ──────────────────────────────────────────────── */
    #header {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      background: linear-gradient(90deg, var(--bg2), var(--bg1));
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0 18px;
      gap: 18px;
    }
    #header h1 {
      font-family: var(--font-hud);
      font-size: 16px;
      font-weight: 900;
      color: var(--accent);
      text-shadow: var(--glow);
      letter-spacing: 2px;
    }
    #header .subtitle {
      font-size: 10px;
      color: var(--text-dim);
      letter-spacing: 1.5px;
      text-transform: uppercase;
    }
    .header-spacer { flex: 1; }
    .status-badge {
      display: flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 4px 10px;
      border-radius: 20px;
      border: 1px solid;
    }
    .badge-sim  { border-color: var(--accent2); color: var(--accent2); }
    .badge-live { border-color: var(--green);   color: var(--green);   }
    .dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      animation: blink 1.1s ease-in-out infinite;
    }
    .dot-green  { background: var(--green); }
    .dot-purple { background: var(--accent2); }
    @keyframes blink {
      0%,100% { opacity: 1; }
      50%      { opacity: 0.2; }
    }
    #elapsed-badge {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--yellow);
    }

    /* ── Panels ──────────────────────────────────────────────── */
    .panel {
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 8px;
      position: relative;
    }
    .panel::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: var(--radius);
      background: linear-gradient(135deg,
        rgba(0,229,255,0.03) 0%,
        transparent 60%);
      pointer-events: none;
    }
    .panel-title {
      font-family: var(--font-hud);
      font-size: 9px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--accent);
      opacity: 0.8;
      border-bottom: 1px solid var(--border);
      padding-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .panel-title .icon { font-size: 11px; }

    /* ── Left sidebar ─────────────────────────────────────────── */
    #sidebar-left {
      grid-row: 2 / 3;
      grid-column: 1 / 2;
    }

    /* Sensor values */
    .sensor-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 5px;
    }
    .sensor-card {
      background: var(--bg3);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px 8px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .sensor-label {
      font-size: 9px;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .sensor-value {
      font-family: var(--font-mono);
      font-size: 13px;
      font-weight: 500;
      color: var(--accent);
    }
    .sensor-unit {
      font-size: 9px;
      color: var(--text-dim);
    }

    /* Status section */
    .status-block {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .status-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 8px;
      background: var(--bg3);
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 10px;
    }
    .status-label { color: var(--text-dim); }
    .status-val {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--green);
    }
    .status-val.warn { color: var(--orange); }

    /* Battery bar */
    .battery-wrap { display: flex; flex-direction: column; gap: 3px; }
    .battery-bar-bg {
      height: 8px;
      background: var(--bg3);
      border-radius: 4px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    #battery-bar {
      height: 100%;
      background: linear-gradient(90deg, var(--green), var(--yellow));
      border-radius: 4px;
      transition: width 0.5s ease;
      width: 100%;
    }

    /* Motor gauges */
    .motor-row {
      display: flex;
      gap: 5px;
    }
    .motor-gauge {
      flex: 1;
      background: var(--bg3);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 5px 7px;
    }
    .motor-name {
      font-size: 9px;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .motor-val {
      font-family: var(--font-mono);
      font-size: 13px;
      color: var(--orange);
    }
    .motor-bar-bg {
      height: 4px;
      background: var(--bg0);
      border-radius: 2px;
      overflow: hidden;
      margin-top: 3px;
    }
    .motor-bar {
      height: 100%;
      border-radius: 2px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      transition: width 0.2s;
    }

    /* ── Camera feed ──────────────────────────────────────────── */
    #panel-camera {
      grid-row: 2 / 3;
      grid-column: 2 / 3;
    }
    #cam-img-wrap {
      flex: 1;
      background: #000;
      border-radius: 8px;
      overflow: hidden;
      position: relative;
    }
    #cam-img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .cam-badge {
      position: absolute;
      bottom: 6px;
      right: 6px;
      background: rgba(0,0,0,0.7);
      border: 1px solid var(--accent);
      border-radius: 4px;
      padding: 2px 7px;
      font-family: var(--font-mono);
      font-size: 9px;
      color: var(--accent);
    }
    .corner-bracket {
      position: absolute;
      width: 16px; height: 16px;
      border-color: var(--accent);
      border-style: solid;
      opacity: 0.6;
    }
    .tl { top: 4px; left: 4px;  border-width: 2px 0 0 2px; }
    .tr { top: 4px; right: 4px; border-width: 2px 2px 0 0; }
    .bl { bottom: 4px; left: 4px;  border-width: 0 0 2px 2px; }
    .br { bottom: 4px; right: 4px; border-width: 0 2px 2px 0; }
    .orb-stats {
      display: flex;
      gap: 10px;
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-dim);
    }
    .orb-stat-val { color: var(--green); font-weight: 500; }

    /* ── Map panel ────────────────────────────────────────────── */
    #panel-map {
      grid-row: 2 / 3;
      grid-column: 3 / 4;
    }
    #map-canvas-wrap {
      flex: 1;
      position: relative;
    }
    #map-canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }

    /* ── Right sidebar ────────────────────────────────────────── */
    #sidebar-right {
      grid-row: 2 / 3;
      grid-column: 4 / 5;
    }

    /* Progress */
    .progress-ring-wrap {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 4px 0;
    }
    .ring-label {
      font-family: var(--font-hud);
      font-size: 18px;
      font-weight: 900;
      fill: var(--accent);
    }
    .ring-sub {
      font-size: 8px;
      fill: var(--text-dim);
      font-family: var(--font-body);
    }

    .info-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .info-item {
      display: flex;
      justify-content: space-between;
      padding: 5px 7px;
      background: var(--bg3);
      border-radius: 5px;
      font-size: 10px;
    }
    .info-key  { color: var(--text-dim); }
    .info-val  { font-family: var(--font-mono); color: var(--yellow); }

    /* ── Bottom charts ────────────────────────────────────────── */
    #bottom-charts {
      grid-row: 3 / 4;
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    #panel-pid, #panel-motor {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .chart-wrap {
      flex: 1;
      position: relative;
    }
    .chart-wrap canvas {
      position: absolute;
      inset: 0;
      width: 100% !important;
      height: 100% !important;
    }

    /* ── Scan-line overlay ────────────────────────────────────── */
    .scanlines {
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.04) 2px,
        rgba(0,0,0,0.04) 4px
      );
      z-index: 9999;
    }

    /* ── Shimmer animation ────────────────────────────────────── */
    @keyframes shimmer {
      0%   { background-position: -200% center; }
      100% { background-position:  200% center; }
    }
    .shimmer {
      background: linear-gradient(90deg,
        transparent 25%,
        rgba(0,229,255,0.08) 50%,
        transparent 75%
      );
      background-size: 200% 100%;
      animation: shimmer 3s linear infinite;
    }

    /* ── Pulse animation ──────────────────────────────────────── */
    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 6px rgba(0,229,255,0.2); }
      50%       { box-shadow: 0 0 22px rgba(0,229,255,0.5); }
    }
    .panel:hover { animation: pulse 2s ease-in-out infinite; }

    /* ── Tilt gauge ───────────────────────────────────────────── */
    #tilt-gauge-wrap { display: flex; align-items: center; justify-content: center; }
    #tilt-canvas { border-radius: 50%; }
  </style>
</head>
<body>
<div class="scanlines"></div>

<div id="app">

  <!-- ── HEADER ─────────────────────────────────────────────────────────── -->
  <header id="header">
    <div>
      <h1>⚡ VIBOT-S</h1>
      <div class="subtitle">Self-Balancing Vision-Based Navigation Robot</div>
    </div>
    <div class="header-spacer"></div>
    <div class="status-badge badge-sim">
      <div class="dot dot-purple"></div> SIMULATED HARDWARE
    </div>
    <div class="status-badge badge-live">
      <div class="dot dot-green"></div> LIVE SIMULATION
    </div>
    <div id="elapsed-badge">T = 0.0s</div>
  </header>

  <!-- ── LEFT SIDEBAR ───────────────────────────────────────────────────── -->
  <aside class="panel" id="sidebar-left">
    <div class="panel-title"><span class="icon">📡</span> SENSOR DATA</div>

    <div class="sensor-grid">
      <div class="sensor-card">
        <div class="sensor-label">Tilt Angle</div>
        <div class="sensor-value" id="sv-tilt">0.00</div>
        <div class="sensor-unit">degrees</div>
      </div>
      <div class="sensor-card">
        <div class="sensor-label">Velocity</div>
        <div class="sensor-value" id="sv-vel">0.00</div>
        <div class="sensor-unit">m/s</div>
      </div>
      <div class="sensor-card">
        <div class="sensor-label">Accel X</div>
        <div class="sensor-value" id="sv-ax">0.00</div>
        <div class="sensor-unit">m/s²</div>
      </div>
      <div class="sensor-card">
        <div class="sensor-label">Accel Z</div>
        <div class="sensor-value" id="sv-az">9.81</div>
        <div class="sensor-unit">m/s²</div>
      </div>
      <div class="sensor-card">
        <div class="sensor-label">Gyro X</div>
        <div class="sensor-value" id="sv-gx">0.00</div>
        <div class="sensor-unit">°/s</div>
      </div>
      <div class="sensor-card">
        <div class="sensor-label">Motor Cmd</div>
        <div class="sensor-value" id="sv-mcmd">0.0</div>
        <div class="sensor-unit">PWM</div>
      </div>
    </div>

    <!-- Battery -->
    <div class="battery-wrap">
      <div class="status-row" style="padding:4px 8px;">
        <span class="status-label">🔋 Battery</span>
        <span class="status-val" id="bat-txt">100.0%</span>
      </div>
      <div class="battery-bar-bg">
        <div id="battery-bar"></div>
      </div>
    </div>

    <!-- Motor gauges -->
    <div class="motor-row">
      <div class="motor-gauge">
        <div class="motor-name">Left Motor</div>
        <div class="motor-val" id="ml-val">0</div>
        <div class="motor-bar-bg">
          <div class="motor-bar" id="ml-bar" style="width:0%"></div>
        </div>
      </div>
      <div class="motor-gauge">
        <div class="motor-name">Right Motor</div>
        <div class="motor-val" id="mr-val">0</div>
        <div class="motor-bar-bg">
          <div class="motor-bar" id="mr-bar" style="width:0%"></div>
        </div>
      </div>
    </div>

    <!-- System status -->
    <div class="panel-title" style="margin-top:4px;"><span class="icon">⚙️</span> SYSTEM STATUS</div>
    <div class="status-block">
      <div class="status-row">
        <span class="status-label">Navigation</span>
        <span class="status-val" id="nav-status">Initializing…</span>
      </div>
      <div class="status-row">
        <span class="status-label">Balancing</span>
        <span class="status-val" id="bal-status">Active</span>
      </div>
      <div class="status-row">
        <span class="status-label">IMU Mode</span>
        <span class="status-val">MPU6050 (SIM)</span>
      </div>
      <div class="status-row">
        <span class="status-label">Camera</span>
        <span class="status-val">Pi-CAM (SIM)</span>
      </div>
    </div>
  </aside>

  <!-- ── CAMERA FEED ─────────────────────────────────────────────────────── -->
  <div class="panel" id="panel-camera">
    <div class="panel-title"><span class="icon">📷</span> LIVE CAMERA FEED + ORB FEATURES</div>
    <div id="cam-img-wrap" style="flex:1; min-height:0;">
      <img id="cam-img" src="/stream/video" alt="Live Camera"/>
      <div class="corner-bracket tl"></div>
      <div class="corner-bracket tr"></div>
      <div class="corner-bracket bl"></div>
      <div class="corner-bracket br"></div>
      <div class="cam-badge" id="orb-badge">ORB: 0 kp</div>
    </div>
    <div class="orb-stats">
      <span>Keypoints: <span class="orb-stat-val" id="orb-kp">—</span></span>
      <span>Frame: <span class="orb-stat-val" id="orb-frame">—</span></span>
      <span>Detector: <span class="orb-stat-val">ORB-FAST</span></span>
    </div>
  </div>

  <!-- ── GRID MAP ────────────────────────────────────────────────────────── -->
  <div class="panel" id="panel-map">
    <div class="panel-title"><span class="icon">🗺️</span> ROBOT NAVIGATION MAP (A*)</div>
    <div id="map-canvas-wrap">
      <canvas id="map-canvas"></canvas>
    </div>
  </div>

  <!-- ── RIGHT SIDEBAR ──────────────────────────────────────────────────── -->
  <aside class="panel" id="sidebar-right">
    <div class="panel-title"><span class="icon">🤖</span> ROBOT STATUS</div>

    <!-- Progress ring -->
    <div class="progress-ring-wrap">
      <svg width="110" height="110" viewBox="0 0 110 110">
        <circle cx="55" cy="55" r="47" fill="none"
                stroke="rgba(0,229,255,0.08)" stroke-width="8"/>
        <circle cx="55" cy="55" r="47" fill="none"
                stroke="url(#ringGrad)" stroke-width="8"
                stroke-linecap="round"
                stroke-dasharray="295.3"
                stroke-dashoffset="295.3"
                id="ring-progress"
                transform="rotate(-90 55 55)"/>
        <defs>
          <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stop-color="#00e5ff"/>
            <stop offset="100%" stop-color="#7c4dff"/>
          </linearGradient>
        </defs>
        <text x="55" y="50" text-anchor="middle" class="ring-label" id="ring-pct">0%</text>
        <text x="55" y="64" text-anchor="middle" class="ring-sub">PATH DONE</text>
      </svg>
    </div>

    <div class="info-list">
      <div class="info-item">
        <span class="info-key">Position</span>
        <span class="info-val" id="info-pos">—</span>
      </div>
      <div class="info-item">
        <span class="info-key">Step</span>
        <span class="info-val" id="info-step">—</span>
      </div>
      <div class="info-item">
        <span class="info-key">Loops</span>
        <span class="info-val" id="info-loops">0</span>
      </div>
      <div class="info-item">
        <span class="info-key">Path len</span>
        <span class="info-val" id="info-pathlen">—</span>
      </div>
      <div class="info-item">
        <span class="info-key">Elapsed</span>
        <span class="info-val" id="info-elapsed">0s</span>
      </div>
    </div>

    <!-- Tilt gauge (canvas arc) -->
    <div class="panel-title" style="margin-top:6px;"><span class="icon">⚖️</span> TILT GAUGE</div>
    <div id="tilt-gauge-wrap">
      <canvas id="tilt-canvas" width="120" height="70"></canvas>
    </div>
  </aside>

  <!-- ── BOTTOM CHARTS ──────────────────────────────────────────────────── -->
  <div id="bottom-charts">
    <div class="panel" id="panel-pid">
      <div class="panel-title"><span class="icon">📊</span> PID BALANCING – TILT ANGLE (°)</div>
      <div class="chart-wrap">
        <canvas id="pid-chart"></canvas>
      </div>
    </div>
    <div class="panel" id="panel-motor">
      <div class="panel-title"><span class="icon">⚡</span> MOTOR COMMAND OUTPUT (PWM)</div>
      <div class="chart-wrap">
        <canvas id="motor-chart"></canvas>
      </div>
    </div>
  </div>

</div><!-- /#app -->

<script>
// ── Chart setup ─────────────────────────────────────────────────────────────
const chartDefaults = {
  animation: false,
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: {
      display: false,
    },
    y: {
      grid: { color: 'rgba(0,229,255,0.06)' },
      ticks: { color: '#7a90b0', font: { family: 'JetBrains Mono', size: 9 } },
    }
  },
};

// PID / Tilt chart
const pidCtx  = document.getElementById('pid-chart').getContext('2d');
const pidChart = new Chart(pidCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Tilt (°)',
      data: [],
      borderColor: '#ff5252',
      borderWidth: 2,
      fill: true,
      backgroundColor: 'rgba(255,82,82,0.08)',
      tension: 0.4,
      pointRadius: 0,
    }]
  },
  options: {
    ...chartDefaults,
    scales: {
      ...chartDefaults.scales,
      y: { ...chartDefaults.scales.y, min: -20, max: 20,
           title: { display: false } }
    }
  }
});

// Motor chart
const motorCtx  = document.getElementById('motor-chart').getContext('2d');
const motorChart = new Chart(motorCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Motor CMD',
      data: [],
      borderColor: '#00e5ff',
      borderWidth: 2,
      fill: true,
      backgroundColor: 'rgba(0,229,255,0.06)',
      tension: 0.4,
      pointRadius: 0,
    }]
  },
  options: {
    ...chartDefaults,
    scales: {
      ...chartDefaults.scales,
      y: { ...chartDefaults.scales.y, min: -260, max: 260 }
    }
  }
});

// ── Grid map renderer ────────────────────────────────────────────────────────
const mapCanvas = document.getElementById('map-canvas');
const mapCtx    = mapCanvas.getContext('2d');

let gridData = null;
let pathData = [], posLog = [], robotPos = [1,1], startPos = [1,1], goalPos = [18,18];
const GRID_ROWS = 20, GRID_COLS = 20;

function renderMap() {
  if (!gridData) return;
  const W = mapCanvas.parentElement.clientWidth;
  const H = mapCanvas.parentElement.clientHeight;
  mapCanvas.width  = W;
  mapCanvas.height = H;

  const cellW = W / GRID_COLS;
  const cellH = H / GRID_ROWS;

  // Draw grid
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const v = gridData[r][c];
      if      (v === 1)  mapCtx.fillStyle = '#0d1117';   // obstacle
      else if (v === 0)  mapCtx.fillStyle = '#0f1e35';   // free
      else               mapCtx.fillStyle = '#0a1228';   // unknown
      mapCtx.fillRect(c * cellW, r * cellH, cellW, cellH);
      // Grid lines
      mapCtx.strokeStyle = 'rgba(0,229,255,0.04)';
      mapCtx.lineWidth   = 0.5;
      mapCtx.strokeRect(c * cellW, r * cellH, cellW, cellH);
    }
  }

  // Draw planned path
  if (pathData.length > 1) {
    mapCtx.beginPath();
    mapCtx.moveTo(pathData[0][1]*cellW + cellW/2, pathData[0][0]*cellH + cellH/2);
    for (let i = 1; i < pathData.length; i++) {
      mapCtx.lineTo(pathData[i][1]*cellW + cellW/2, pathData[i][0]*cellH + cellH/2);
    }
    mapCtx.strokeStyle = 'rgba(0,229,255,0.35)';
    mapCtx.lineWidth   = 1.5;
    mapCtx.setLineDash([4, 3]);
    mapCtx.stroke();
    mapCtx.setLineDash([]);
  }

  // Draw visited trail
  if (posLog.length > 1) {
    mapCtx.beginPath();
    mapCtx.moveTo(posLog[0][1]*cellW + cellW/2, posLog[0][0]*cellH + cellH/2);
    for (let i = 1; i < posLog.length; i++) {
      mapCtx.lineTo(posLog[i][1]*cellW + cellW/2, posLog[i][0]*cellH + cellH/2);
    }
    mapCtx.strokeStyle = 'rgba(0,230,118,0.55)';
    mapCtx.lineWidth   = 2;
    mapCtx.stroke();
  }

  // Start marker
  mapCtx.beginPath();
  mapCtx.arc(startPos[1]*cellW+cellW/2, startPos[0]*cellH+cellH/2, 5, 0, Math.PI*2);
  mapCtx.fillStyle  = '#00e676';
  mapCtx.fill();

  // Goal marker
  mapCtx.beginPath();
  mapCtx.arc(goalPos[1]*cellW+cellW/2, goalPos[0]*cellH+cellH/2, 5, 0, Math.PI*2);
  mapCtx.fillStyle = '#ff1744';
  mapCtx.fill();

  // Robot marker (triangle)
  const rx = robotPos[1]*cellW + cellW/2;
  const ry = robotPos[0]*cellH + cellH/2;
  const sz = Math.max(cellW, 8) * 0.65;
  mapCtx.save();
  mapCtx.translate(rx, ry);
  mapCtx.beginPath();
  mapCtx.moveTo(0, -sz);
  mapCtx.lineTo(sz * 0.7, sz * 0.7);
  mapCtx.lineTo(-sz * 0.7, sz * 0.7);
  mapCtx.closePath();
  mapCtx.fillStyle   = '#ffd740';
  mapCtx.shadowBlur  = 12;
  mapCtx.shadowColor = '#ffd740';
  mapCtx.fill();
  mapCtx.restore();

  // Legend
  mapCtx.font = '9px JetBrains Mono';
  const legends = [
    { col: '#00e676', txt: '▶ Start' },
    { col: '#ff1744', txt: '⚑ Goal'  },
    { col: '#ffd740', txt: '▲ Robot' },
    { col: 'rgba(0,229,255,0.5)', txt: '─ Path' },
    { col: 'rgba(0,230,118,0.8)', txt: '─ Trail' },
  ];
  legends.forEach((l, i) => {
    mapCtx.fillStyle = l.col;
    mapCtx.fillText(l.txt, 6, 12 + i * 13);
  });
}

// ── Tilt gauge ───────────────────────────────────────────────────────────────
const tiltCanvas = document.getElementById('tilt-canvas');
const tiltCtx    = tiltCanvas.getContext('2d');
const TILT_MAX   = 10;  // degrees max shown

function renderTiltGauge(tilt) {
  const W = tiltCanvas.width, H = tiltCanvas.height;
  tiltCtx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H - 8;
  const r  = H - 16;
  const startA = Math.PI, endA = 2 * Math.PI;

  // Background arc
  tiltCtx.beginPath();
  tiltCtx.arc(cx, cy, r, startA, endA);
  tiltCtx.strokeStyle = 'rgba(0,229,255,0.1)';
  tiltCtx.lineWidth   = 8;
  tiltCtx.stroke();

  // Coloured arc (tilt value)
  const norm  = Math.max(-1, Math.min(1, tilt / TILT_MAX));  // -1..1
  const angle = Math.PI + norm * (Math.PI / 2);               // π … 3π/2 … 2π
  tiltCtx.beginPath();
  tiltCtx.arc(cx, cy, r, Math.PI, angle,  norm < 0);
  tiltCtx.strokeStyle = Math.abs(tilt) < 2 ? '#00e676' :
                        Math.abs(tilt) < 5 ? '#ffd740' : '#ff5252';
  tiltCtx.lineWidth  = 8;
  tiltCtx.lineCap    = 'round';
  tiltCtx.stroke();

  // Needle
  const needleAngle = Math.PI + norm * (Math.PI / 2);
  const nx = cx + (r - 4) * Math.cos(needleAngle);
  const ny = cy + (r - 4) * Math.sin(needleAngle);
  tiltCtx.beginPath();
  tiltCtx.moveTo(cx, cy);
  tiltCtx.lineTo(nx, ny);
  tiltCtx.strokeStyle = '#fff';
  tiltCtx.lineWidth   = 2;
  tiltCtx.stroke();

  // Tilt value text
  tiltCtx.font      = 'bold 11px JetBrains Mono';
  tiltCtx.fillStyle = '#00e5ff';
  tiltCtx.textAlign = 'center';
  tiltCtx.fillText(tilt.toFixed(2) + '°', cx, cy - r/2);
}

// ── SSE data handler ─────────────────────────────────────────────────────────
function updateUI(d) {
  // Elapsed
  document.getElementById('elapsed-badge').textContent = `T = ${d.elapsed}s`;

  // Sensor values
  document.getElementById('sv-tilt').textContent  = d.tilt_angle .toFixed(3);
  document.getElementById('sv-vel') .textContent  = d.velocity   .toFixed(3);
  document.getElementById('sv-ax')  .textContent  = d.accel_x    .toFixed(3);
  document.getElementById('sv-az')  .textContent  = d.accel_z    .toFixed(2);
  document.getElementById('sv-gx')  .textContent  = d.gyro_x     .toFixed(2);
  document.getElementById('sv-mcmd').textContent  = d.current_motor.toFixed(1);

  // Battery
  const bat = d.battery_pct;
  document.getElementById('bat-txt').textContent  = bat.toFixed(1) + '%';
  document.getElementById('battery-bar').style.width = bat + '%';

  // Motors
  const ml = Math.abs(d.motor_left),  mr = Math.abs(d.motor_right);
  document.getElementById('ml-val').textContent = d.motor_left.toFixed(0)  + ' rpm';
  document.getElementById('mr-val').textContent = d.motor_right.toFixed(0) + ' rpm';
  document.getElementById('ml-bar').style.width = Math.min(ml/6, 100) + '%';
  document.getElementById('mr-bar').style.width = Math.min(mr/6, 100) + '%';

  // Status
  document.getElementById('nav-status') .textContent = d.nav_status;
  document.getElementById('bal-status') .textContent = d.balance_status;

  // Right sidebar
  const pct = d.total_steps > 0
      ? Math.round(d.step_idx / d.total_steps * 100)
      : 0;
  const circumference = 295.3;
  document.getElementById('ring-progress').style.strokeDashoffset =
      circumference - (pct / 100) * circumference;
  document.getElementById('ring-pct').textContent = pct + '%';

  robotPos = d.robot_pos;
  document.getElementById('info-pos') .textContent = `[${robotPos[0]}, ${robotPos[1]}]`;
  document.getElementById('info-step').textContent = `${d.step_idx} / ${d.total_steps}`;
  document.getElementById('info-loops').textContent = d.loops;
  document.getElementById('info-pathlen').textContent = d.total_steps;
  document.getElementById('info-elapsed').textContent = d.elapsed + 's';

  // Charts
  const labels = d.time_history;
  pidChart.data.labels   = labels;
  pidChart.data.datasets[0].data = d.tilt_history;
  pidChart.update('none');

  motorChart.data.labels   = labels;
  motorChart.data.datasets[0].data = d.motor_history;
  motorChart.update('none');

  // Map
  posLog   = d.pos_log;
  renderMap();

  // Tilt gauge
  renderTiltGauge(d.current_tilt);
}

// ── Fetch ORB stats ──────────────────────────────────────────────────────────
async function pollOrbStats() {
  try {
    const resp = await fetch('/api/camera');
    const data = await resp.json();
    document.getElementById('orb-kp')    .textContent = data.kp_count;
    document.getElementById('orb-frame') .textContent = data.frame_idx;
    document.getElementById('orb-badge') .textContent = `ORB: ${data.kp_count} kp`;
  } catch(e) {}
}
setInterval(pollOrbStats, 500);

// ── Fetch initial grid ────────────────────────────────────────────────────────
async function fetchInitialState() {
  try {
    const resp = await fetch('/api/state');
    const data = await resp.json();
    gridData  = data.grid;
    pathData  = data.path;
    startPos  = data.start;
    goalPos   = data.goal;
    renderMap();
  } catch(e) {
    console.error('[ViBot-S] Failed to fetch initial state:', e);
  }
}

// ── SSE connection ────────────────────────────────────────────────────────────
function connectSSE() {
  const evtSource = new EventSource('/stream/events');
  evtSource.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      posLog   = d.pos_log  || posLog;
      pathData = pathData   || [];
      updateUI(d);
    } catch(err) { console.warn('[SSE] parse error', err); }
  };
  evtSource.onerror = () => {
    console.warn('[SSE] connection error – reconnecting in 2s');
    evtSource.close();
    setTimeout(connectSSE, 2000);
  };
}

// ── Boot ─────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  fetchInitialState();
  connectSSE();
  setInterval(renderMap, 100);
});

// Resize map canvas
new ResizeObserver(() => renderMap())
  .observe(document.getElementById('map-canvas-wrap'));
</script>
</body>
</html>
"""


# ─── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  ViBot-S Dashboard – starting at http://localhost:5000")
    print("  Press Ctrl+C to stop.")
    print("=" * 62)

    # Pre-warm singletons
    get_robot()
    _get_camera()
    _get_imu()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
