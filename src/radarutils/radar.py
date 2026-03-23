import numpy as np
from .antenna import GainPattern
from .wave import Wavefront

class Radar:
    def __init__(self, res_deg, smin_db, Pw=0.1, pattern_type="ideal",
                 gain_dBi=1, x=0, y=0, ang_vel_deg_s=10, phi=0, pulse_width_s=1e-7):

        self.ang_vel = ang_vel_deg_s
        self.x = x
        self.y = y
        self.Pw = Pw
        self.smin_db = smin_db
        self.phi = phi
        self.pulse_width_s = pulse_width_s

        self.gain = GainPattern(res_deg, pattern_type, gain_dBi)
        self.waves = []

        self.tx_times = []
        self.tx_amplitudes = []

        self.rx_times = []
        self.rx_amplitudes = []

    def iluminate(self, current_t, dt):
        n_samples = int(self.pulse_width_s / dt)

        for i in range(n_samples):
            t_offset = i * dt

            wf = Wavefront(
                self.Pw,
                self.gain,
                self.smin_db,
                self.phi,
                x0=self.x,
                y0=self.y
            )

            wf.t0 = current_t + t_offset
            self.waves.append(wf)

            self.tx_times.append(wf.t0)
            self.tx_amplitudes.append(wf.Pw)


    def rotate(self, dt):
        step_ang = self.ang_vel * dt
        self.phi = (self.phi + step_ang) % 360

    def receive_wave(self, power_received, current_t):
        self.rx_times.append(current_t)
        self.rx_amplitudes.append(power_received)

if __name__ == "__main__":
    from .animations import animate_radar

    radar = Radar(5, 1, -80, 1, "sinc")
    R_values = np.linspace(1, 100, 100)

    animate_radar(radar, R_values)

