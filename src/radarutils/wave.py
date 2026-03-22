import numpy as np
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
        self.Pw = Pw
        self.threshold_db = threshold_db
        self.gain_pattern = gain_pattern
        self.phi = center
        self.arcs = []
        self._build_arcs()
        self.R=0

    def _build_arcs(self):
        # todo, explicar formula depois, mas basicamente é pra chegar no index com base na precisão angular para rotacionar o gain pattern
        shift_indices = int(np.deg2rad(self.phi) / self.gain_pattern.res_deg)
        rotated_gain = np.roll(self.gain_pattern.Hgain_lin_vec, shift_indices)
    
        self.arcs = []
        for i in range(len(self.gain_pattern.theta_vec)):
            self.arcs.append(
                ArcComponent(i, self.Pw, rotated_gain[i])
            )

    def update(self, step=1):
        self.R += step 
        active_arcs = []

        for arc in self.arcs:
            # Agora usa o self.R interno
            p_db = arc.compute_power_db(self.R)

            if p_db >= self.threshold_db:
                active_arcs.append(arc)

        self.arcs = active_arcs

        
if __name__ == "__main__":
    from .animations import animate_wavefront
    gp = GainPattern(10, "cosine", 10, 30)
    wf = Wavefront(1, gp, -100, 0)

    R_values = np.linspace(1, 100, 100)

    animate_wavefront(
        wavefront_template=wf,
        R_values=R_values,
    )