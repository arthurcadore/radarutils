import numpy as np
from .component import Radar, Target, OrbitalTarget, NestedOrbitalTarget

class PPI(): 
    def __init__(self, dimensions: tuple[int, int] = (1000, 1000), t=10, dt=0.0001):
        self.dimensions = dimensions
        self.t = t
        self.dt = dt
        self.elapsed_time = 0.0
        self.targets = []
        self.clutters = []
        self.radar = None
        self.r_max = self.dimensions[0]
    
    @property
    def theta_low(self):
        if self.radar:
            return self.radar.theta - self.radar.beamwidth / 2
        return 0

    @property
    def theta_high(self):
        if self.radar:
            return self.radar.theta + self.radar.beamwidth / 2
        return 0

    def add_target(self, x, y, vel=0, acc=0, theta=0):
        target = Target(x, y, vel, acc, theta)
        self.targets.append(target)

    def add_radar(self, pt=1000, gt=30, s_min=1e-10, beamwidth=10, irradPattern=None, theta=0, rpm=1):
        self.radar = Radar(self.r_max, pt, gt, s_min, beamwidth, irradPattern, theta=theta, rpm=rpm)

    def add_orbital_target(self, r, speed, acceleration=0, clockwise=False, alpha_start=0):
        target = OrbitalTarget(r, speed, acceleration, clockwise, alpha_start)
        self.targets.append(target)

    def add_nested_orbital_target(self, r1, speed1, acc1, r2, speed2, acc2, clockwise1=False, clockwise2=False, alpha1_start=0, alpha2_start=0):
        target = NestedOrbitalTarget(r1, speed1, acc1, r2, speed2, acc2, clockwise1, clockwise2, alpha1_start, alpha2_start)
        self.targets.append(target)

    def add_clutter():
        pass

    def update(self):
        self.elapsed_time += self.dt
        detections = []
        
        for i, target in enumerate(self.targets):
            target.update(self.dt)

            if self.radar:
                # Target range and angle calculation
                r = np.sqrt(target.x**2 + target.y**2)
                alpha = np.degrees(np.arctan2(target.y, target.x)) % 360
                
                # Beam boundaries normalized to [0, 360)
                l = self.theta_low % 360
                h = self.theta_high % 360

                # Check if target is inside the beam (handling wrap-around at 360)
                in_beam = False
                if l <= h:
                    in_beam = (l <= alpha <= h)
                else: # Wrap around case (e.g. low=355, high=5)
                    in_beam = (alpha >= l or alpha <= h)

                if in_beam:
                    # Calculate normalized error [0-1] where 0 is center, 1 is edge
                    # Shortest angular difference between boresight and target azimuth
                    radar_az = self.radar.theta % 360
                    diff = ((alpha - radar_az + 180) % 360) - 180
                    
                    # Normalize by half-beamwidth (max possible error if in beam)
                    # Use a small epsilon to avoid div by zero if beamwidth is somehow 0
                    bw_half = max(self.radar.beamwidth / 2, 0.001)
                    norm_error = min(abs(diff) / bw_half, 1.0)
                    
                    detections.append((r, norm_error, i))

                # Detection events
                was_in_beam = getattr(target, 'in_beam', False)
                if in_beam and not was_in_beam:
                    print(f"[{self.elapsed_time:.3f}s] Target ENTER: R={r:.2f}, Az={alpha:.2f}°")
                elif not in_beam and was_in_beam:
                    print(f"[{self.elapsed_time:.3f}s] Target EXIT:  R={r:.2f}, Az={alpha:.2f}°")
                
                target.in_beam = in_beam
        
        if self.radar:
            self.radar.update(self.dt)
            
        return detections