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
        self.targets = []

        self._last_pulse_time = -np.inf

    def add_radar(self, radar):
        self.radars.append(radar)

    def add_target(self, target):
        self.targets.append(target)

    def _setup_plot(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor("black")
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_aspect('equal')

        for r in self.radars:
            ax.plot(r.x, r.y, 'wo', markersize=5)

        for t in self.targets:
            ax.plot(t.x, t.y, 'ro', markersize=5)

        return fig, ax

    def _process_wave(self, wf, current_t, segments, powers_db):
        delta_t = current_t - wf.t0
        wf.R = self.C * delta_t
        wf.update(step=0)

        limit = max(abs(v) for v in self.x_lim + self.y_lim)

        if wf.arcs and wf.R < limit * 1.5:

            for arc in wf.arcs:
                theta = wf.gain_pattern.theta_vec[arc.idx]
                dtheta = wf.gain_pattern.res_deg
                theta_seg = np.array([theta - dtheta/2, theta + dtheta/2])

                x = wf.x0 + wf.R * np.cos(theta_seg)
                y = wf.y0 + wf.R * np.sin(theta_seg)

                segments.append(np.column_stack((x, y)))
                powers_db.append(arc.power_db)

            return True

        return False

    def _check_collision(self, wf, current_t):
        for i, target in enumerate(self.targets):

            if i in wf.hit_targets:
                continue

            dist = np.sqrt((target.x - wf.x0)**2 + (target.y - wf.y0)**2)

            if abs(dist - wf.R) < self.C * self.dt:

                angle = np.degrees(np.arctan2(
                    target.y - wf.y0,
                    target.x - wf.x0
                )) % 360

                idx = int(round(angle / wf.gain_pattern.res_deg)) % len(wf.arcs)

                if idx < len(wf.arcs):

                    arc = wf.arcs[idx]

                    power_w = 10**(arc.power_db / 10)

                    target.receive_wave(power_w, current_t)

                    wf.hit_targets.add(i)

    def _update_frame(self, frame_idx, ax, collections, pulse_interval_s):
        current_t = self.t_vec[frame_idx]

        artists = []
        segments = []
        powers_db = []

        should_pulse = False

        if current_t - self._last_pulse_time >= pulse_interval_s:
            should_pulse = True
            self._last_pulse_time = current_t

        for radar in self.radars:
            radar.phi = (radar.ang_vel * current_t) % 360

            if should_pulse:
                radar.iluminate()
                radar.waves[-1].t0 = current_t

            active_waves = []

            for wf in radar.waves:
                self._check_collision(wf, current_t)

                if self._process_wave(wf, current_t, segments, powers_db):
                    active_waves.append(wf)

            radar.waves = active_waves

        for target in self.targets:
            active_waves = []

            for wf in target.waves:
                if self._process_wave(wf, current_t, segments, powers_db):
                    active_waves.append(wf)

            target.waves = active_waves

        if collections[0]:
            collections[0].remove()

        if segments:
            norm = np.clip((np.array(powers_db) + 120) / 120, 0, 1)

            coll = LineCollection(
                segments,
                colors=cm.turbo(norm),
                linewidths=1.5
            )

            ax.add_collection(coll)
            collections[0] = coll
            artists.append(coll)

        ax.set_title(f"t = {current_t:.2e}s", color="white")

        return artists

    def run_simulation(self, pulse_interval_s=0.0001):
        fig, ax = self._setup_plot()

        collections = [None]

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

        plt.show()

if __name__ == "__main__":
    from .radar import Radar
    from .target import Target
    
    scene = Scene(
        t_max=0.00001,
        dt=0.00000002,
        x_lim=(-500, 500),
        y_lim=(-500, 500)
    )
    
    radar = Radar(
        res_deg=1,
        smin_db=-80,
        x=0,
        y=0,
        ang_vel_deg_s=90,
        pattern_type="sinc",
        Pw=10
    )
    
    t1 = Target(x=400,y=0)

    scene.add_radar(radar)
    scene.add_target(t1)

    scene.run_simulation(
        pulse_interval_s=0.000001
    )