import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from .antenna import GainPattern
from .wave import Wavefront

class Radar: 
    def __init__(self, ang_vel, res_deg, smin_db, Pw=1, pattern_type="ideal", gain_dBi=10, x=0, y=0):
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
        
def animate_radar(radar: Radar, R_values):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_facecolor("black")

    max_R = np.max(R_values)

    ax.set_xlim(-max_R * 1.1, max_R * 1.1)
    ax.set_ylim(-max_R * 1.1, max_R * 1.1)

    ax.plot(0, 0, 'wo')

    collection = None

    # 🔥 frames onde novos pulsos serão emitidos
    pulse_frames = [0, len(R_values)//3, 2*len(R_values)//3]

    def update(frame_idx):
        nonlocal collection

        R = R_values[frame_idx]

        # 🔥 emitir novos pulsos em momentos específicos
        if frame_idx in pulse_frames:
            radar.iluminate()
            radar.rotate()  # opcional: girar a cada pulso

        segments = []
        powers_db = []

        # 🔥 iterar sobre TODOS os pulsos ativos
        active_waves = []

        for wave in radar.waves:
            wave.update(R)

            if not wave.arcs:
                continue  # pulso morreu

            active_waves.append(wave)

            for arc in wave.arcs:
                theta = wave.gain_pattern.theta_vec[arc.idx]
                dtheta = wave.gain_pattern.theta_res

                theta_segment = np.array([
                    theta - dtheta/2,
                    theta + dtheta/2
                ])

                x = R * np.cos(theta_segment)
                y = R * np.sin(theta_segment)

                segments.append(np.column_stack((x, y)))
                powers_db.append(arc.power_db)

        # 🔥 remover ondas mortas
        radar.waves = active_waves

        if not segments:
            return []

        powers_db = np.array(powers_db)

        norm = np.clip(
            (powers_db - radar.smin_db) / (0 - radar.smin_db),
            0, 1
        )

        colors = cm.turbo(norm)

        if collection:
            collection.remove()

        collection = LineCollection(segments, colors=colors, linewidths=2)
        ax.add_collection(collection)

        ax.set_title(
            f"R = {R:.2f} | Pulsos ativos: {len(radar.waves)}"
        )

        return [collection]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(R_values),
        interval=50,
        blit=True
    )

    plt.show()

if __name__ == "__main__":

    radar = Radar(1, 1, -60)
    R_values = np.linspace(1, 100, 100)

    animate_radar(radar, R_values)

