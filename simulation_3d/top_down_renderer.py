"""
top_down_renderer.py  –  Engineering-grade RViz renderer for ViBot-S
----------------------------------------------------------------------
Improvements over previous version:
  ● Robot: filled triangle + solid base-circle + velocity arrow
  ● Grid: every-5-cell major lines + every-1-cell minor lines, metric labels
  ● Axes: thicker X(red)/Y(green) with arrowheads drawn at 0,0
  ● Layer toggles: map · path · robot · trajectory · fov · axes · legend
  ● Legend panel in bottom-right corner
  ● Smooth interpolation (linear pos + shortest-path angle)
  ● Cinematic-quality anti-aliased drawing
"""

import cv2
import numpy as np
import math
import time

# ── Engineering color palette (BGR) ──────────────────────────────────────────
BG_UNEXPLORED  = (228, 231, 233)
FREE_SPACE     = (250, 250, 250)
OBSTACLE       = ( 18,  18,  18)
GRID_MINOR     = (216, 216, 216)
GRID_MAJOR     = (185, 185, 185)
AXIS_X         = (  0,   0, 200)   # red X
AXIS_Y         = (  0, 160,   0)   # green Y
PATH_COLOR     = ( 50, 170,  50)   # green path
TRAIL_BASE     = (170, 210, 170)   # trajectory dots
START_COLOR    = (190, 110,  20)   # blue-orange start
GOAL_CROSS     = ( 20,  20, 200)   # red goal
ROBOT_BODY     = ( 40,  90, 210)   # blue robot fill
ROBOT_RING     = (  5,  45, 130)   # dark ring
HEADING_ARR    = (  0,   0, 200)   # heading arrow
VEL_ARR        = ( 10, 130, 210)   # velocity arrow
FOV_FILL       = (180, 160,  45)   # FOV cone
HUD_BG         = (242, 244, 245)
TEXT_DARK      = ( 20,  20,  20)
LABEL_MUTED    = ( 90,  90,  90)
BORDER         = ( 80,  80,  80)
LEGEND_BG      = (248, 249, 250)


class TopDownRenderer:
    FOV_HALF    = 28        # degrees
    FOV_DEPTH   = 4.0       # cells
    TRAIL_MAX   = 800

    def __init__(self, width=800, height=550, scale=14):
        self._w     = width
        self._h     = height
        self._scale = scale

        # Interpolation state
        self._prev_rx:  float = 0.0
        self._prev_ry:  float = 0.0
        self._prev_rth: float = 0.0
        self._first:    bool  = True
        self._t0        = time.monotonic()
        self._step_t    = 0.45   # seconds

        # Trail
        self._trail: list = []

        # Layer flags
        self.show_map        = True
        self.show_path       = True
        self.show_robot      = True
        self.show_trajectory = True
        self.show_fov        = True
        self.show_axes       = True
        self.show_legend     = True

    # ─────────────────────────────────────────────────────────────────────────
    def render(self, rx, ry, rth, grid, path, goal, nav_status="NAVIGATING"):
        # ── Smooth interpolation ──────────────────────────────────────────────
        moved = (self._first
                 or abs(rx - self._prev_rx) > 0.06
                 or abs(ry - self._prev_ry) > 0.06)
        if moved:
            self._first   = False
            self._prev_rx  = rx;  self._prev_ry  = ry;  self._prev_rth = rth
            self._t0       = time.monotonic()

        alpha = min((time.monotonic() - self._t0) / self._step_t, 1.0)
        irx   = self._prev_rx  + alpha * (rx  - self._prev_rx)
        iry   = self._prev_ry  + alpha * (ry  - self._prev_ry)
        irth  = self._prev_rth + alpha * _adiff(rth, self._prev_rth)

        # Estimated velocity direction (trail-based)
        vel_x = rx - self._prev_rx
        vel_y = ry - self._prev_ry
        vel_mag = math.hypot(vel_x, vel_y)

        # ── Canvas ───────────────────────────────────────────────────────────
        frame = np.full((self._h, self._w, 3), BG_UNEXPLORED, dtype=np.uint8)

        if not grid or not grid[0]:
            return frame

        R, C = len(grid), len(grid[0])
        s = self._scale
        mpw, mph = C * s, R * s
        ox = (self._w - mpw) // 2     # map left edge
        oy = 38 + (self._h - 38 - mph) // 2  # map top edge (below HUD)

        def px(col, row):  # grid → pixel center
            return (ox + int(col * s + s / 2), oy + int(row * s + s / 2))

        rpx = px(irx, iry)

        # ── 1. Map layer (cells) ──────────────────────────────────────────────
        if self.show_map:
            for r in range(R):
                for c in range(C):
                    x1 = ox + c * s;  y1 = oy + r * s
                    col = OBSTACLE if grid[r][c] == 1 else FREE_SPACE
                    cv2.rectangle(frame, (x1, y1), (x1+s, y1+s), col, -1)

            # Minor grid lines (every cell)
            for c in range(0, C+1):
                px_x = ox + c * s
                cv2.line(frame, (px_x, oy), (px_x, oy+mph), GRID_MINOR, 1)
            for r in range(0, R+1):
                px_y = oy + r * s
                cv2.line(frame, (ox, px_y), (ox+mpw, px_y), GRID_MINOR, 1)

            # Major grid lines (every 5 cells) + labels
            for c in range(0, C+1, 5):
                px_x = ox + c * s
                cv2.line(frame, (px_x, oy), (px_x, oy+mph), GRID_MAJOR, 1)
                cv2.putText(frame, str(c), (px_x - 5, oy + mph + 14),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, LABEL_MUTED, 1, cv2.LINE_AA)
            for r in range(0, R+1, 5):
                px_y = oy + r * s
                cv2.line(frame, (ox, px_y), (ox+mpw, px_y), GRID_MAJOR, 1)
                cv2.putText(frame, str(r), (ox - 26, px_y + 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, LABEL_MUTED, 1, cv2.LINE_AA)

        # ── 2. Coordinate axes ───────────────────────────────────────────────
        if self.show_axes:
            o = (ox, oy)
            cv2.arrowedLine(frame, o, (ox+60, oy), AXIS_X, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.arrowedLine(frame, o, (ox, oy+60), AXIS_Y, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(frame, "X", (ox+64, oy+5),  cv2.FONT_HERSHEY_PLAIN, 1.0, AXIS_X, 1, cv2.LINE_AA)
            cv2.putText(frame, "Y", (ox-12, oy+75), cv2.FONT_HERSHEY_PLAIN, 1.0, AXIS_Y, 1, cv2.LINE_AA)
            cv2.putText(frame, "0,0", (ox+3, oy-4), cv2.FONT_HERSHEY_PLAIN, 0.75, TEXT_DARK, 1, cv2.LINE_AA)

        # ── 3. A* Path ───────────────────────────────────────────────────────
        if self.show_path and path and len(path) > 1:
            pts = np.array([px(p[1], p[0]) for p in path],
                           dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, PATH_COLOR, 2, cv2.LINE_AA)

            # Start circle
            sp = px(path[0][1], path[0][0])
            cv2.circle(frame, sp, 7, START_COLOR, -1, cv2.LINE_AA)
            cv2.putText(frame, "START", (sp[0]+8, sp[1]-4),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, START_COLOR, 1, cv2.LINE_AA)

        # Goal marker
        if self.show_path and goal is not None:
            gr, gc = (goal[0], goal[1]) if hasattr(goal, '__len__') else (goal, goal)
            gp = px(gc, gr)
            cv2.drawMarker(frame, gp, GOAL_CROSS, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.circle(frame, gp, 10, GOAL_CROSS, 1, cv2.LINE_AA)
            cv2.putText(frame, "GOAL", (gp[0]+12, gp[1]-10),
                        cv2.FONT_HERSHEY_PLAIN, 0.85, GOAL_CROSS, 1, cv2.LINE_AA)

        # ── 4. Trajectory trail ──────────────────────────────────────────────
        self._trail.append(rpx)
        if len(self._trail) > self.TRAIL_MAX:
            self._trail.pop(0)

        if self.show_trajectory and len(self._trail) > 1:
            n = len(self._trail)
            for i in range(1, n):
                a = i / n
                g = int(155 + 55 * a)
                cv2.circle(frame, self._trail[i], 1, (g, g+10, g), -1)

        # ── 5. Camera FOV cone ───────────────────────────────────────────────
        if self.show_fov:
            fhr  = math.radians(self.FOV_HALF)
            dlen = self.FOV_DEPTH * s
            al   = irth - fhr;  ar = irth + fhr
            fan  = [rpx] + [
                _ipt(rpx[0] + dlen * math.cos(al + k*(2*fhr)/10),
                     rpx[1] + dlen * math.sin(al + k*(2*fhr)/10))
                for k in range(11)
            ]
            ov = frame.copy()
            cv2.fillPoly(ov, [np.array(fan, np.int32)], FOV_FILL)
            cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
            cv2.polylines(frame, [np.array(fan, np.int32)], True, FOV_FILL, 1, cv2.LINE_AA)

        # ── 6. Robot (circle + triangle + arrows) ────────────────────────────
        if self.show_robot:
            rr = 9    # body radius
            bx, by = rpx

            # Base circle
            cv2.circle(frame, rpx, rr + 2, ROBOT_RING, -1, cv2.LINE_AA)
            cv2.circle(frame, rpx, rr - 1, ROBOT_BODY, -1, cv2.LINE_AA)

            # Direction triangle tip
            tip  = _ipt(bx + (rr+7)*math.cos(irth),    by + (rr+7)*math.sin(irth))
            bl   = _ipt(bx + rr*0.7*math.cos(irth+2.3), by + rr*0.7*math.sin(irth+2.3))
            br   = _ipt(bx + rr*0.7*math.cos(irth-2.3), by + rr*0.7*math.sin(irth-2.3))
            cv2.fillPoly(frame, [np.array([tip,bl,br],np.int32)], ROBOT_BODY)

            # Heading arrow
            h_end = _ipt(bx+(rr+14)*math.cos(irth), by+(rr+14)*math.sin(irth))
            cv2.arrowedLine(frame, rpx, h_end, HEADING_ARR, 2, cv2.LINE_AA, tipLength=0.35)

            # Velocity arrow (only if moving)
            if vel_mag > 0.05:
                vel_ang = math.atan2(vel_y, vel_x)
                v_end   = _ipt(bx + (rr+20)*math.cos(vel_ang), by + (rr+20)*math.sin(vel_ang))
                cv2.arrowedLine(frame, rpx, v_end, VEL_ARR, 1, cv2.LINE_AA, tipLength=0.4)

            # Label
            cv2.putText(frame, "robot", (bx+rr+3, by-rr-2),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, ROBOT_BODY, 1, cv2.LINE_AA)

        # ── 7. Legend box ────────────────────────────────────────────────────
        if self.show_legend:
            lx, ly, lw, lh = self._w - 145, self._h - 128, 140, 122
            cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), (210,212,214), -1)
            cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), BORDER, 1)
            cv2.putText(frame, "LEGEND", (lx+8, ly+15),
                        cv2.FONT_HERSHEY_PLAIN, 0.85, TEXT_DARK, 1, cv2.LINE_AA)
            entries = [
                (ROBOT_BODY,   "circle", "Robot"),
                (PATH_COLOR,   "line",   "A* Path"),
                (GOAL_CROSS,   "cross",  "Goal"),
                (START_COLOR,  "circle", "Start"),
                (TRAIL_BASE,   "dot",    "Trajectory"),
                (OBSTACLE,     "square", "Obstacle"),
            ]
            for i, (col, shape, name) in enumerate(entries):
                ey = ly + 28 + i * 16
                ex = lx + 12
                if shape == "circle":
                    cv2.circle(frame, (ex+6, ey-3), 5, col, -1, cv2.LINE_AA)
                elif shape == "line":
                    cv2.line(frame, (ex, ey-3), (ex+12, ey-3), col, 2, cv2.LINE_AA)
                elif shape == "cross":
                    cv2.drawMarker(frame, (ex+6, ey-3), col, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
                elif shape == "dot":
                    cv2.circle(frame, (ex+6, ey-3), 2, col, -1)
                elif shape == "square":
                    cv2.rectangle(frame, (ex+1, ey-7), (ex+11, ey+1), col, -1)
                cv2.putText(frame, name, (ex+18, ey),
                            cv2.FONT_HERSHEY_PLAIN, 0.75, TEXT_DARK, 1, cv2.LINE_AA)

        # ── 8. HUD strip ─────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (self._w, 35), HUD_BG, -1)
        cv2.line(frame, (0, 35), (self._w, 35), BORDER, 1)
        hdg = math.degrees(irth) % 360
        hud = (f"  POS ({irx:.2f}, {iry:.2f}) m"
               f"   HDG {hdg:.1f}\u00b0"
               f"   STATUS: {nav_status}"
               f"   STEPS: {len(path)}")
        cv2.putText(frame, hud, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, TEXT_DARK, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (self._w-1, self._h-1), BORDER, 1)
        return frame

    # ─────────────────────────────────────────────────────────────────────────
    def get_jpeg(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return bytes(buf) if ok else b""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ipt(x, y):
    return (int(round(x)), int(round(y)))

def _adiff(tgt, src):
    d = tgt - src
    while d >  math.pi: d -= 2*math.pi
    while d < -math.pi: d += 2*math.pi
    return d
