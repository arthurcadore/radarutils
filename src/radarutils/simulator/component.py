
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


class NestedOrbitalTarget(Target):
    def __init__(self, r1, speed1, acc1, r2, speed2, acc2, 
                 clockwise1=False, clockwise2=False, 
                 alpha1_start=0, alpha2_start=0):
        """
        Target that orbits a point which itself orbits the radar.
        
        Args:
            r1: Primary orbit radius (distance from radar to orbit center)
            speed1: Tangential speed of primary orbit
            acc1: Tangential acceleration of primary orbit
            r2: Secondary orbit radius (distance from primary orbit center to target)
            speed2: Tangential speed of secondary orbit
            acc2: Tangential acceleration of secondary orbit
            clockwise1: Primary orbit direction
            clockwise2: Secondary orbit direction
            alpha1_start: Initial phase of primary orbit
            alpha2_start: Initial phase of secondary orbit
        """
        # Initial positions
        x1 = r1 * np.cos(alpha1_start)
        y1 = r1 * np.sin(alpha1_start)
        x2 = r2 * np.cos(alpha2_start)
        y2 = r2 * np.sin(alpha2_start)
        
        super().__init__(x1 + x2, y1 + y2, 0, 0, 0)
        
        self.r1 = r1
        self.alpha1 = alpha1_start
        self.omega1 = speed1 / r1 if r1 != 0 else 0
        self.beta1 = acc1 / r1 if r1 != 0 else 0
        if clockwise1:
            self.omega1 = -self.omega1
            self.beta1 = -self.beta1
            
        self.r2 = r2
        self.alpha2 = alpha2_start
        self.omega2 = speed2 / r2 if r2 != 0 else 0
        self.beta2 = acc2 / r2 if r2 != 0 else 0
        if clockwise2:
            self.omega2 = -self.omega2
            self.beta2 = -self.beta2

    def update(self, dt):
        # Update primary orbit
        self.omega1 += self.beta1 * dt
        self.alpha1 += self.omega1 * dt
        
        # Update secondary orbit
        self.omega2 += self.beta2 * dt
        self.alpha2 += self.omega2 * dt
        
        # Calculate components
        x1 = self.r1 * np.cos(self.alpha1)
        y1 = self.r1 * np.sin(self.alpha1)
        
        x2 = self.r2 * np.cos(self.alpha2)
        y2 = self.r2 * np.sin(self.alpha2)
        
        # Total position
        self.x = x1 + x2
        self.y = y1 + y2
        
        # Velocity components for heading calculation
        vx = -self.omega1 * self.r1 * np.sin(self.alpha1) - self.omega2 * self.r2 * np.sin(self.alpha2)
        vy = self.omega1 * self.r1 * np.cos(self.alpha1) + self.omega2 * self.r2 * np.cos(self.alpha2)
        
        self.theta = np.arctan2(vy, vx)
        self.velocity = np.sqrt(vx**2 + vy**2)


        