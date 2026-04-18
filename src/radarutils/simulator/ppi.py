from .component import Radar, Target, OrbitalTarget

class PPI(): 
    def __init__(self, dimensions: tuple[int, int] = (1000, 1000), t=10, dt=0.0001):
        self.dimensions = dimensions
        self.t = t
        self.dt = dt
        self.targets = []
        self.clutters = []
        self.radar = None
        self.r_max = self.dimensions[0]

    def add_target(self, x, y, vel=0, acc=0, theta=0):
        target = Target(x, y, vel, acc, theta)
        self.targets.append(target)

    def add_radar(self, pt=1000, gt=30, s_min=1e-10, beamwidth=1, irradPattern=None, theta=0, rpm=1):
        self.radar = Radar(self.r_max, pt, gt, s_min, beamwidth, irradPattern, theta=theta, rpm=rpm)

    def add_orbital_target(self, r, speed, acceleration=0, clockwise=False, alpha_start=0):
        target = OrbitalTarget(r, speed, acceleration, clockwise, alpha_start)
        self.targets.append(target)

    def add_clutter():
        pass

    def update(self):
        for target in self.targets:
            target.update(self.dt)
        
        self.radar.update(self.dt)