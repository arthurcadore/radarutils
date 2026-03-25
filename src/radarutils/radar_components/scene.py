import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

from ..core.data import ExportData, ImportData


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
        # 🔥 calcula proporção real
        width = abs(self.x_lim[1] - self.x_lim[0])
        height = abs(self.y_lim[1] - self.y_lim[0])
        ratio = width / height

        base_size = 6  # altura base
        fig_width = base_size * ratio
        fig_height = base_size

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        fig.patch.set_facecolor("#1e2129")
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

    def _check_radar_collision(self, wf, current_t):
        for radar in self.radars:

            dist = np.sqrt((radar.x - wf.x0)**2 + (radar.y - wf.y0)**2)

            if abs(dist - wf.R) < self.C * self.dt:

                # potência média dos arcos ativos
                if wf.arcs:
                    powers = [10**(arc.power_db / 10) for arc in wf.arcs]
                    power_received = np.mean(powers)

                    radar.receive_wave(power_received, current_t)

    def _update_frame(self, frame_idx, fig, ax, collections, pulse_interval_s):
        current_t = self.t_vec[frame_idx]

        ax.clear()

        nx = 10
        ny = 10
        sub = 5

        xticks = np.linspace(self.x_lim[0], self.x_lim[1], nx + 1)
        yticks = np.linspace(self.y_lim[0], self.y_lim[1], ny + 1)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticks(np.linspace(self.x_lim[0], self.x_lim[1], nx*sub + 1), minor=True)
        ax.set_yticks(np.linspace(self.y_lim[0], self.y_lim[1], ny*sub + 1), minor=True)

        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.6, alpha=0.5)
        ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.3, alpha=0.2)
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")

        ax.set_facecolor("black")
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_aspect('equal')

        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        for r in self.radars:
            ax.plot(r.x, r.y, 'wo', markersize=5)

        for t in self.targets:
            ax.plot(t.x, t.y, 'ro', markersize=5)

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
                radar.iluminate(current_t, self.dt)

            active_waves = []

            for wf in radar.waves:
                self._check_collision(wf, current_t)

                if self._process_wave(wf, current_t, segments, powers_db):
                    active_waves.append(wf)

            radar.waves = active_waves

        for target in self.targets:
            active_waves = []

            for wf in target.waves:
                self._check_radar_collision(wf, current_t)

                if self._process_wave(wf, current_t, segments, powers_db):
                    active_waves.append(wf)

            target.waves = active_waves

        if segments:
            norm = np.clip((np.array(powers_db) + 120) / 120, 0, 1)

            coll = LineCollection(
                segments,
                colors=cm.turbo(norm),
                linewidths=1.5
            )

            ax.add_collection(coll)
            artists.append(coll)

        ax.set_title(f"t = {current_t:.2e}s", color="white")

        if frame_idx == len(self.t_vec) - 1:
            self.anim.event_source.stop()
            plt.close(fig)

        return artists

    def run_simulation(self, pulse_interval_s=0.0001, save_type="gif", save_name="simulacao"):
        fig, ax = self._setup_plot()

        collections = [None]

        self._last_pulse_time = -pulse_interval_s

        self.anim = FuncAnimation(
            fig,
            self._update_frame,
            frames=len(self.t_vec),
            fargs=(fig, ax, collections, pulse_interval_s),
            interval=30,
            blit=False,
            repeat=False
        )

        if save_type == "gif":
            writer = PillowWriter(fps=30)
            self.anim.save(
                f"{save_name}.gif",
                writer=writer,
                dpi=100,
                savefig_kwargs={
                    "facecolor": "#1e2129",
                    "bbox_inches": "tight"
                }
            )
            plt.close(fig)
        elif save_type == "mp4":
            writer = FFMpegWriter(fps=30) 
            self.anim.save(f"{save_name}.mp4", writer=writer) 
            plt.close(fig)
        else: 
            plt.show()

    def save_data(self):
        import numpy as np

        for i, radar in enumerate(self.radars):
            # TX
            t_tx = np.array(radar.tx_times)
            a_tx = np.array(radar.tx_amplitudes)
            tx = (t_tx, a_tx)
            print(f"tx_data_radar_{i}", tx)

            ExportData(tx, f"tx_data_radar_{i}").save()

            # RX
            t_rx = np.array(radar.rx_times)
            a_rx = np.array(radar.rx_amplitudes)
            rx = (t_rx, a_rx)
            print(f"rx_data_radar_{i}", rx)


            ExportData(rx, f"rx_data_radar_{i}").save()
    


if __name__ == "__main__":
    from .radar import Radar
    from .target import Target
    
    scene = Scene(
        t_max=0.00002,
        dt=0.00000002,
        x_lim=(-1200, 1200),
        y_lim=(-600, 600)
    )
    
    radar = Radar(
        res_deg=0.1,
        smin_db=-80,
        x=0,
        y=0,
        ang_vel_deg_s=90,
        pattern_type="sinc",
        Pw=10
    )
    
    t1 = Target(x=480,y=0)

    scene.add_radar(radar)
    scene.add_target(t1)

    scene.run_simulation(
        pulse_interval_s=3.5e-6,
    )
    scene.save_data()