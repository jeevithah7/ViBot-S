import math

class Robot3D:
    def __init__(self, x=1.0, y=1.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0
        self.w = 0.0

    def set_cmd_vel(self, v, w):
        """Set linear velocity v (m/s) and angular velocity w (rad/s)"""
        self.v = v
        self.w = w

    def step(self, dt):
        """Update pose based on unicycle kinematics"""
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.w * dt
        
        # Keep theta within [-pi, pi]
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
