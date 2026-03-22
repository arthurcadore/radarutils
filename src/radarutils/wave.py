import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


from .antenna import GainPattern

class ArcComponent:
    def __init__(self, idx, Pw, Gt_lin):
        r"""
        Arc component unit for a WaveFront simulation. The ArcComponent its the tinnest non redundant area of the simulation to detect a target.

        Args: 
            Idx (int): Index of ArcComponent of the wavefront simulation instance
            Gt (float): Gain of the transmitting antenna (in $dBi$).
            R (float): Range to the target in meters ($m$).
        """
        self.PtW = Pw
        self.Gt_lin = Gt_lin
        self.idx = idx
        self.power_db = None

    def compute_power_db(self, R):
        r"""
        Arc component unit for a WaveFront simulation. To calculate the power at a distance ($R$) by a radar system, we can use the radar range equation, which is given by: 

        $$
        \begin{equation}
            P_w = \frac{P_t \cdot G_t}{(4\pi)R^2}
        \end{equation}
        $$

        Where:
            - $P_w$ is the power received by the radar in watts ($W$).
            - $P_t$ is the transmitted power in watts ($W$).
            - $G_t$ is the gain of the transmitting antenna.
            - $R$ is the range to the target in meters ($m$).

        Args: 
            R (float): Range to the target in meters ($m$).

        Returns:
            Pr_dBW (float): Power received by the radar in decibels with respect to one watt ($dBW$).

        <div class="referencia">
            <b>Reference:</b>
            <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Pg - 18) </p>
        </div>
        """
        P = self.PtW * self.Gt_lin / (4 * np.pi * R**2)
        self.power_db = 10 * np.log10(P + 1e-20)
        return self.power_db


# comentar que thr = s_min

class Wavefront:
    def __init__(self, Pw, gain_pattern:GainPattern, threshold_db, center):
        r"""


        """ 
        self.Pw = Pw
        self.threshold_db = threshold_db
        self.gain_pattern = gain_pattern
        self.phi = center
        self.arcs = []
        self._build_arcs()

    def _build_arcs(self):
        # todo, explicar formula depois, mas basicamente é pra chegar no index com base na precisão angular para rotacionar o gain pattern
        self.phi_idx = int((self.phi/360)*self.gain_pattern.len_vec) 
        rotated_gain = np.roll(self.gain_pattern.gain_lin_vec, self.phi_idx)

        for i, _ in enumerate(self.gain_pattern.theta_vec):
            self.arcs.append(
                ArcComponent(i, self.Pw, rotated_gain[i])
            )

    def update(self, R):
        active_arcs = []

        for arc in self.arcs:
            p_db = arc.compute_power_db(R)

            # remove se cair abaixo do threashold
            if p_db >= self.threshold_db:
                active_arcs.append(arc)

        self.arcs = active_arcs


def animate_wavefront(wavefront: Wavefront, R_values):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_facecolor("black")

    max_R = np.max(R_values)

    ax.set_xlim(-max_R * 1.1, max_R * 1.1)
    ax.set_ylim(-max_R * 1.1, max_R * 1.1)

    ax.plot(0, 0, 'wo')

    collection = None

    def update(frame):
        nonlocal collection

        R = frame

        # atualiza modelo (remove arcos mortos)
        wavefront.update(R)

        arcs = wavefront.arcs

        if not arcs:
            return []

        segments = []
        powers_db = []

        for arc in arcs:
            theta = wavefront.gain_pattern.theta_vec[arc.idx]

            # largura angular (resolução)
            dtheta = wavefront.gain_pattern.theta_res

            theta_segment = np.array([theta - dtheta/2, theta + dtheta/2])

            x = R * np.cos(theta_segment)
            y = R * np.sin(theta_segment)

            segments.append(np.column_stack((x, y)))
            powers_db.append(arc.power_db)

        powers_db = np.array(powers_db)

        norm = np.clip((powers_db - wavefront.threshold_db) / (0 - wavefront.threshold_db), 0, 1)
        colors = cm.turbo(norm)

        # remove frame anterior
        if collection:
            collection.remove()

        collection = LineCollection(segments, colors=colors, linewidths=2)
        ax.add_collection(collection)

        ax.set_title(f"R = {R:.2f} | Arcos: {len(arcs)}")

        return [collection]

    anim = FuncAnimation(
        fig,
        update,
        frames=R_values,
        interval=50,
        blit=True
    )

    plt.show()


if __name__ == "__main__":
    gp = GainPattern(1, "cosine", 10, 30)
    wf = Wavefront(0.01, gp, -100, 0)

    R_values = np.linspace(1, 100, 100)

    animate_wavefront(
        wavefront=wf,
        R_values=R_values,
    )