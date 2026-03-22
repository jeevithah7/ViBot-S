import numpy as np
import cv2
import math
import os

class RobotCamera:
    def __init__(self, width=640, height=360, fov_deg=60.0):
        self.W = width
        self.H = height
        self.focal = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        self.cx = width / 2.0
        self.cy = height / 2.0

        # Load REAL texture from static assets if available, else fallback
        self.texture = self._load_realistic_texture()

    def _load_realistic_texture(self):
        """Loads a real indoor image from the static/textures folder for the 1P view."""
        path = os.path.join("dashboard", "static", "textures", "hallway_wall.png")
        if os.path.exists(path):
            tex = cv2.imread(path)
            if tex is not None:
                # Resize to a consistent power-of-two for warping accuracy
                return cv2.resize(tex, (512, 512))
        
        # Fallback to high-quality synthetic if file missing
        tex = np.zeros((512, 512, 3), dtype=np.uint8)
        tex[:] = (40, 42, 45)
        for x in range(0, 512, 128):
            cv2.rectangle(tex, (x+20, 50), (x+108, 462), (25, 27, 30), -1)
        return tex

    def render(self, robot_x, robot_y, robot_theta, walls):
        """High-fidelity projection using REAL high-res textures."""
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        
        # 1. Background (Vignetted moody horizon)
        for y in range(int(self.cy)):
            ratio = y / self.cy
            frame[y, :] = (12+10*ratio, 14+12*ratio, 18+15*ratio)
            frame[self.H-1-y, :] = (10+15*ratio, 12+18*ratio, 15+22*ratio)

        cos_t = math.cos(robot_theta)
        sin_t = math.sin(robot_theta)
        
        polygons = []
        for wall in walls:
            tx1, ty1 = wall[0] - robot_x, wall[1] - robot_y
            tx2, ty2 = wall[2] - robot_x, wall[3] - robot_y
            
            z1 =  tx1 * cos_t + ty1 * sin_t
            x1 = -tx1 * sin_t + ty1 * cos_t
            z2 =  tx2 * cos_t + ty2 * sin_t
            x2 = -tx2 * sin_t + ty2 * cos_t
            
            if z1 < 0.1 and z2 < 0.1: continue
            
            cx1, cz1, cx2, cz2 = x1, z1, x2, z2
            u1_tex, u2_tex = 0.0, 1.0
            if cz1 < 0.1:
                t = (0.1 - cz1) / (cz2 - cz1); cx1 += t * (cx2 - cx1); cz1 = 0.1; u1_tex = t
            if cz2 < 0.1:
                t = (0.1 - cz2) / (cz1 - cz2); cx2 += t * (cx1 - cx2); cz2 = 0.1; u2_tex = 1.0 - t
            
            def project(x, z, h):
                u = self.focal * x / z + self.cx
                v = self.focal * h / z + self.cy
                return u, v
            
            u1_b, v1_b = project(cx1, cz1, 0.6)
            u1_t, v1_t = project(cx1, cz1, -0.6)
            u2_b, v2_b = project(cx2, cz2, 0.6)
            u2_t, v2_t = project(cx2, cz2, -0.6)
            
            poly = np.array([[u1_t,v1_t],[u2_t,v2_t],[u2_b,v2_b],[u1_b,v1_b]], dtype=np.float32)
            polygons.append(((cz1+cz2)/2.0, poly, u1_tex, u2_tex))

        polygons.sort(key=lambda x: x[0], reverse=True)
        for d, poly, u1, u2 in polygons:
            src = np.float32([[u1*511,0],[u2*511,0],[u2*511,511],[u1*511,511]])
            try:
                M = cv2.getPerspectiveTransform(src, poly)
                warped = cv2.warpPerspective(self.texture, M, (self.W, self.H))
                mask = np.zeros((self.H, self.W), dtype=np.uint8)
                cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
                # Shading for depth
                shade = max(0.15, math.exp(-d / 15.0))
                warped = (warped.astype(np.float32) * shade).astype(np.uint8)
                np.copyto(frame, warped, where=(mask[:,:,None] == 255))
            except: continue

        # Visual Effects (Grain + Lens Vignette)
        mask = np.zeros((self.H, self.W), dtype=np.float32)
        cv2.circle(mask, (int(self.cx), int(self.cy)), int(self.W*0.9), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (self.W//2 | 1, self.W//2 | 1), 0)
        frame = (frame.astype(np.float32) * mask[:,:,None]).astype(np.uint8)
        
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
