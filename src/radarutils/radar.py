import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from .antenna import GainPattern
from .wave import Wavefront

class Radar: 
    def __init__(self, ang_vel, res_deg, smin_db, Pw=0.1, pattern_type="ideal", gain_dBi=1, x=0, y=0):
        self.ang_vel = ang_vel
        self.x = x
        self.y = y
        self.Pw = Pw
        self.smin_db = smin_db
        # onde to olhando
        self.phi = 0
        self.gain = GainPattern(res_deg, pattern_type, gain_dBi)
        self.waves = []

    def iluminate(self):
        self.waves.append(Wavefront(self.Pw, self.gain, self.smin_db, self.phi))

    def rotate(self):
        #TODO: implementar step correspondente a tempo e velocidade corretos.
        # precisa ser menor que 360 sempre. 
        step_ang = 30
        self.phi = self.phi + step_ang
        
if __name__ == "__main__":
    from .animations import animate_radar

    radar = Radar(5, 1, -80, 1, "sinc")
    R_values = np.linspace(1, 100, 100)

    animate_radar(radar, R_values)

