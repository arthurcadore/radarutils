import numpy as np
from .antenna import GainPattern
from .wave import Wavefront

class Radar:
    def __init__(self, res_deg, smin_db, Pw=0.1, pattern_type="ideal",
                 gain_dBi=1, x=0, y=0, ang_vel_deg_s=10, phi=0):

        self.ang_vel = ang_vel_deg_s
        self.x = x
        self.y = y
        self.Pw = Pw
        self.smin_db = smin_db
        self.phi = phi

        self.gain = GainPattern(res_deg, pattern_type, gain_dBi)
        self.waves = []

    def iluminate(self):
        wf = Wavefront(
            self.Pw,
            self.gain,
            self.smin_db,
            self.phi,
            x0=self.x,
            y0=self.y
        )
        self.waves.append(wf)

    def rotate(self, dt):
        step_ang = self.ang_vel * dt
        self.phi = (self.phi + step_ang) % 360

if __name__ == "__main__":
    from .animations import animate_radar

    radar = Radar(5, 1, -80, 1, "sinc")
    R_values = np.linspace(1, 100, 100)

    animate_radar(radar, R_values)

