
import abc
import numpy as np 
from ..core.basics import calc_max_prf

class Components(abc.ABC): 
    def __init__(self, x, y, vel, acc, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = vel
        self.acceleration = acc
        self.phi = 0

    def update_acceleration(self, acc):
        self.acceleration = acc

    def update_velocity(self, dt):
        self.velocity += self.acceleration * dt

    def update_position(self, dt):
        self.x += self.velocity * np.cos(self.theta) * dt
        self.y += self.velocity * np.sin(self.theta) * dt

    def update(self, dt):
        self.update_velocity(dt)
        self.update_position(dt)

class Radar(Components):
    def __init__(self, r_max, pt, gt, s_min, beamwidth, irradPattern, x=0, y=0, theta=0, rpm=1):
        super().__init__(x, y, 0, 0, theta)
        self.rpm = rpm
        self.prf = calc_max_prf(r_max)
        self.pt = pt
        self.gt = gt
        self.s_min = s_min
        self.beamwidth = beamwidth
        self.irradPattern = irradPattern

    def rotate(self, dt):
        self.theta += self.rpm * dt

    def update(self, dt):
        self.rotate(dt)


class Target(Components):
    def __init__(self, x, y, vel, acc, theta):
        super().__init__(x, y, vel, acc, theta)


class OrbitalTarget(Target):
    def __init__(self, r, speed, acceleration, clockwise=False, alpha_start=0):
        x = r * np.cos(alpha_start)
        y = r * np.sin(alpha_start)
        
        direction_factor = -1 if clockwise else 1
        theta = alpha_start + (direction_factor * np.pi / 2)
        
        super().__init__(x, y, speed, acceleration, theta)
        
        self.r = r
        self.alpha = alpha_start
        self.clockwise = clockwise
        
        self.omega = speed / r if r != 0 else 0
        self.beta = acceleration / r if r != 0 else 0
        
        if clockwise:
            self.omega = -self.omega
            self.beta = -self.beta

    def update(self, dt):
        self.omega += self.beta * dt
        self.alpha += self.omega * dt
        self.x = self.r * np.cos(self.alpha)
        self.y = self.r * np.sin(self.alpha)
        # Keep theta tangent to path
        direction_factor = 1 if self.omega >= 0 else -1
        self.theta = self.alpha + (direction_factor * np.pi / 2)

        