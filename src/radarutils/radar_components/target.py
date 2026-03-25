from .wave import Wavefront
from .antenna import GainPattern


class Target:
    def __init__(self, x, y, threshold_db=-120, res_deg=1, cs=1):
        self.x = x
        self.y = y
        self.threshold_db = threshold_db
        self.cs = cs

        self.gain = GainPattern(
            res_deg=res_deg,
            pattern_type="isotropic",
            gain_dBi=0
        )

        self.waves = []

    def receive_wave(self, power_received, current_t):
        power_reflected = power_received * self.cs
        wf = Wavefront(
            Pw=power_reflected,
            gain_pattern=self.gain,
            threshold_db=self.threshold_db,
            center=0,
            x0=self.x,
            y0=self.y
        )

        wf.t0 = current_t
        self.waves.append(wf)