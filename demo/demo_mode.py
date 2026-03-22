"""
demo_mode.py
------------
ViBot-S One-Click Demo Launcher
================================

Run this script to:
  1. Pre-warm the robot simulation engine (A*, PID loop, IMU)
  2. Run a quick console data-stream preview
  3. Launch the Flask dashboard on http://localhost:5000
  4. Automatically open the browser

Usage
-----
    python demo/demo_mode.py
    python demo/demo_mode.py --port 8080
    python demo/demo_mode.py --no-browser

Arguments
---------
    --port        TCP port for the dashboard server (default 5000)
    --no-browser  Skip auto-open of the browser
    --preview     Show a console sensor preview then exit (no server)
    --host        Host to bind to (default 0.0.0.0)
"""

import sys, os, time, argparse, threading, webbrowser

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from realtime.live_robot  import LiveRobot
from realtime.live_camera import LiveCamera
from realtime.live_imu    import SimulatedIMU
from dashboard.components import summarise_state


# ─── Colour helpers ───────────────────────────────────────────────────────────
RESET  = "\033[0m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"
DIM    = "\033[2m"


def cprint(text: str, colour: str = RESET):
    print(f"{colour}{text}{RESET}")


# ─── Banner ───────────────────────────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   ██╗   ██╗██╗██████╗  ██████╗ ████████╗       ███████╗    ║
║   ██║   ██║██║██╔══██╗██╔═══██╗╚══██╔══╝       ██╔════╝    ║
║   ██║   ██║██║██████╔╝██║   ██║   ██║    █████╗███████╗    ║
║   ╚██╗ ██╔╝██║██╔══██╗██║   ██║   ██║    ╚════╝╚════██║    ║
║    ╚████╔╝ ██║██████╔╝╚██████╔╝   ██║          ███████║    ║
║     ╚═══╝  ╚═╝╚═════╝  ╚═════╝    ╚═╝          ╚══════╝    ║
║                                                              ║
║  Self-Balancing Vision-Based Indoor Navigation Robot         ║
║  ─────────────────── DEMO MODE ──────────────────────       ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_banner():
    cprint(BANNER, CYAN)
    cprint("  Mode      : SIMULATED HARDWARE", YELLOW)
    cprint("  ORB SLAM  : OpenCV ORB feature detection", GREEN)
    cprint("  Planning  : A* path planner (diagonal 8-conn)", GREEN)
    cprint("  Control   : PID self-balancing (Kp=40 Ki=5 Kd=8)", GREEN)
    cprint("  IMU       : MPU6050-style simulated sensor", GREEN)
    cprint("  Dashboard : Flask + Chart.js + SSE streaming", GREEN)
    print()


# ─── Console sensor preview ───────────────────────────────────────────────────
def console_preview(seconds: float = 5.0):
    """Stream sensor data to console for `seconds` seconds."""
    cprint("\n[DEMO] Starting console sensor preview …", CYAN)
    robot = LiveRobot()
    robot.start()
    imu   = SimulatedIMU()

    t0 = time.time()
    while time.time() - t0 < seconds:
        state = robot.read_state()
        imu_d = imu.read_once()

        tilt  = state["current_tilt"]
        motor = state["current_motor"]
        rp    = state["robot_pos"]
        step  = state["step_idx"]
        total = state["total_steps"]
        bat   = imu_d["battery_pct"]
        vel   = imu_d["velocity"]

        bar_len  = 30
        filled   = int(bar_len * step / max(total, 1))
        bar      = "█" * filled + "░" * (bar_len - filled)
        pct      = int(100 * step / max(total, 1))

        tilt_sym = "↗" if tilt > 0 else "↙" if tilt < 0 else "="
        col      = GREEN if abs(tilt) < 2 else YELLOW if abs(tilt) < 5 else RED

        print(
            f"\r  {CYAN}pos{RESET}=[{rp[0]:02d},{rp[1]:02d}]  "
            f"{CYAN}tilt{RESET}={col}{tilt_sym}{tilt:+6.2f}°{RESET}  "
            f"{CYAN}PWM{RESET}={motor:+6.1f}  "
            f"{CYAN}vel{RESET}={vel:.2f}m/s  "
            f"{CYAN}bat{RESET}={bat:.1f}%  "
            f"{CYAN}path{RESET}=[{bar}] {pct}%",
            end="", flush=True
        )
        time.sleep(0.15)

    robot.stop()
    print(f"\n{GREEN}[DEMO] Preview complete.{RESET}\n")


# ─── Browser auto-open ────────────────────────────────────────────────────────
def open_browser(url: str, delay: float = 1.5):
    """Open the dashboard in the default browser after a short delay."""
    def _open():
        time.sleep(delay)
        cprint(f"[DEMO] Opening browser → {url}", CYAN)
        webbrowser.open(url)
    t = threading.Thread(target=_open, daemon=True)
    t.start()


# ─── Flask launcher ───────────────────────────────────────────────────────────
def launch_dashboard(host: str = "0.0.0.0", port: int = 5000):
    """Import app and start the Flask server (blocking)."""
    # Import here to avoid side-effects if only --preview is used
    from dashboard.app import app, get_robot, _get_camera, _get_imu

    cprint(f"\n[DEMO] Pre-warming simulation engine …", YELLOW)
    get_robot()
    _get_camera()
    _get_imu()
    cprint(f"[DEMO] Engine ready.", GREEN)

    cprint(f"\n[DEMO] Dashboard running at → http://{host}:{port}", BOLD + CYAN)
    cprint(f"       Press Ctrl+C to stop.\n", DIM)

    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ViBot-S One-Click Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo/demo_mode.py                    # default – browser + dashboard
  python demo/demo_mode.py --port 8080        # custom port
  python demo/demo_mode.py --no-browser       # no auto-browser
  python demo/demo_mode.py --preview          # console preview only
        """
    )
    parser.add_argument("--port",       type=int, default=5000,
                        help="Dashboard port (default 5000)")
    parser.add_argument("--host",       type=str, default="0.0.0.0",
                        help="Bind host (default 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open browser automatically")
    parser.add_argument("--preview",    action="store_true",
                        help="Console sensor preview only (no server)")
    parser.add_argument("--preview-time", type=float, default=6.0,
                        help="Duration of console preview in seconds (default 6)")
    args = parser.parse_args()

    print_banner()

    if args.preview:
        console_preview(args.preview_time)
        return

    # Short console preview before server start
    console_preview(seconds=3.0)

    # Auto-open browser
    if not args.no_browser:
        open_browser(f"http://localhost:{args.port}", delay=1.8)

    # Launch Flask dashboard (blocking)
    try:
        launch_dashboard(host=args.host, port=args.port)
    except KeyboardInterrupt:
        cprint("\n[DEMO] Shutting down – goodbye!", CYAN)


if __name__ == "__main__":
    main()
