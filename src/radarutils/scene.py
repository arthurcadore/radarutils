import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

class Scene:
    def __init__(self, t_max, dt, x_lim=(-500, 500), y_lim=(-500, 500), c=299792458):
        self.t_max = t_max
        self.dt = dt
        self.t_vec = np.arange(0, t_max + dt, dt)
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.C = c
        self.radars = []
        # Controladores de tempo para o pulso
        self._last_pulse_time = -np.inf 

    def add_radar(self, radar):
        self.radars.append(radar)

    def _setup_plot(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor("black")
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_aspect('equal')
        
        for r in self.radars:
            ax.plot(r.x, r.y, 'wo', markersize=5)
        
        return fig, ax

    def _update_frame(self, frame_idx, ax, collections, pulse_interval_s):
        current_t = self.t_vec[frame_idx]
        artists = []
        
        # Lógica de disparo baseada em tempo real (segundos)
        should_pulse = False
        if current_t - self._last_pulse_time >= pulse_interval_s:
            should_pulse = True
            self._last_pulse_time = current_t

        for i, radar in enumerate(self.radars):
            # Rotação absoluta phi(t)
            radar.phi = (radar.ang_vel * current_t) % 360
            
            if should_pulse:
                radar.iluminate()
                radar.waves[-1].t0 = current_t

            segments = []
            powers_db = []
            active_waves = []

            for wf in radar.waves:
                delta_t = current_t - wf.t0
                wf.R = self.C * delta_t 
                
                # Atualiza potências com R absoluto
                wf.update(step=0) 

                limit = max(abs(v) for v in self.x_lim + self.y_lim)
                if wf.arcs and wf.R < limit * 1.5:
                    active_waves.append(wf)
                    for arc in wf.arcs:
                        theta = wf.gain_pattern.theta_vec[arc.idx]
                        dtheta = wf.gain_pattern.res_deg 
                        theta_seg = np.array([theta - dtheta/2, theta + dtheta/2])
                        
                        x = radar.x + wf.R * np.cos(theta_seg)
                        y = radar.y + wf.R * np.sin(theta_seg)
                        segments.append(np.column_stack((x, y)))
                        powers_db.append(arc.power_db)
            
            radar.waves = active_waves

            if collections[i]: 
                collections[i].remove()
            
            if segments:
                norm = np.clip((np.array(powers_db) - radar.smin_db) / (0 - radar.smin_db), 0, 1)
                coll = LineCollection(segments, colors=cm.turbo(norm), linewidths=1.5)
                ax.add_collection(coll)
                collections[i] = coll
                artists.append(coll)
            else:
                collections[i] = None

        ax.set_title(f"Simulação: {current_t:.2e}s / {self.t_max}s", color="white")
        return artists

    def run_simulation(self, pulse_interval_s=0.0001, save_gif=False, filename="radar_sim.gif"):
        fig, ax = self._setup_plot()
        collections = [None] * len(self.radars)
        
        # Reset do timer para o início da simulação
        self._last_pulse_time = -pulse_interval_s 

        anim = FuncAnimation(
            fig, 
            self._update_frame, 
            frames=len(self.t_vec),
            fargs=(ax, collections, pulse_interval_s),
            interval=30, 
            blit=True,
            repeat=False 
        )

        if save_gif:
            print(f"Gerando GIF: {filename}...")
            writer = PillowWriter(fps=25)
            anim.save(filename, writer=writer)
            print("GIF salvo com sucesso.")
        
        plt.show()

if __name__ == "__main__":
    from .radar import Radar
    
    scene = Scene(t_max=0.00005, dt=0.00000005)
    r1 = Radar(res_deg=1, smin_db=-50, x=0, y=0, ang_vel_deg_s=3600000, pattern_type="cosine", Pw=1000)

    scene.add_radar(r1)    
    scene.run_simulation(pulse_interval_s=0.0000008, save_gif=False)