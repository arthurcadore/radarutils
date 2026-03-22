import numpy as np
import matplotlib.pyplot as plt

# alterar para classe abstrata.
class GainPattern:
    def __init__(self, res_deg, pattern_type="ideal", gain_dBi=10, beamw_deg=60,
                 n=10, sidelobe_level=0.05, N=8, d_lambda=0.5):
        
        r"""
        Gain Pattern simulation. Used to create a irradiation diagram used for futher signal propagation. The diagram simulation is devided in $N$ components that represents the angular resolution of the simulation based on a angle $\theta$, the expression for $N$: 

        $$
        \begin{equation}
            N = \frac{360}{\theta}
        \end{equation}
        $$

        Where: 
            - $N$ is the number of components (ArcComponents) on the wavefront simulation.
            - $\theta$ is the angular precision of the wavefront simulation in degrees.
        
        Args:
            res_deg (float): Angular precision degree ($\theta$) used for irradiation diagram precision.
            pattern_type (string): Pattern for the irradiation diagram, used for select the pattern equation.
            gain_dBi (float): Max gain in dBi of the antenna. 
            beamw_deg (deg): Beamwidth of the antenna (Beam from max dBi to -3dB). 
        """

        self.theta_res = np.deg2rad(res_deg)
        self.pattern_type = pattern_type
        self.gain_dBi = gain_dBi
        self.beamwidth_3dB = np.deg2rad(beamw_deg)

        # novos parâmetros
        self.n = n
        self.sidelobe_level = sidelobe_level
        self.N = N
        self.d_lambda = d_lambda

        self.theta_vec = np.arange(0, 2*np.pi, self.theta_res)
        self.len_vec = len(self.theta_vec)
        self.gain_dBi_vec = None


        if self.pattern_type == "isotropic":
            self.gain_dBi_vec = self.isotropic_gain()

        elif self.pattern_type == "ideal":
            self.gain_dBi_vec = self.ideal_gain()

        elif self.pattern_type == "sinc":
            self.gain_dBi_vec = self.sinc_gain(5)

        elif self.pattern_type == "cosine":
            self.gain_dBi_vec = self.cosine_sectoral()

        elif self.pattern_type == "taper":
            self.gain_dBi_vec = self.cosine_taper()

        elif self.pattern_type == "array":
            self.gain_dBi_vec = self.array_factor()

        else:
            raise ValueError("Pattern type não suportado")

        self.gain_lin_vec = np.power(10, self.gain_dBi_vec / 10)

    # ---------------------------------
    # isotropic
    # ---------------------------------
    def isotropic_gain(self):
        return np.full_like(self.theta_vec, self.gain_dBi, dtype=float)

    # ---------------------------------
    # ideal (já existente)
    # ---------------------------------
    def ideal_gain(self):
        theta_3db = self.beamwidth_3dB / 2

        n = np.log(0.5) / np.log(np.cos(theta_3db))
        gains_linear = np.zeros_like(self.theta_vec)

        for i, theta in enumerate(self.theta_vec):
            theta_norm = np.arctan2(np.sin(theta), np.cos(theta))

            if abs(theta_norm) <= theta_3db:
                gains_linear[i] = np.cos(theta_norm) ** n
            else:
                gains_linear[i] = 1e-4

        gains_linear /= np.max(gains_linear)

        return self.gain_dBi + 10 * np.log10(gains_linear + 1e-20)

    # ---------------------------------
    # sinc (já existente)
    # ---------------------------------
    def sinc_gain(self, atten_fac=10.0):
        gains_linear = np.zeros_like(self.theta_vec)
        k = 2 * np.pi / self.beamwidth_3dB

        for i, theta in enumerate(self.theta_vec):
            theta_norm = np.arctan2(np.sin(theta), np.cos(theta))
            u = k * np.sin(theta_norm)

            if np.isclose(u, 0):
                val = 1.0
            else:
                val = (np.sin(u) / u) ** 2
                val *= np.exp(-atten_fac * abs(theta_norm))

            gains_linear[i] = val

        # atenua back lobe
        for i, theta in enumerate(self.theta_vec):
            theta_norm = np.arctan2(np.sin(theta), np.cos(theta))
            if abs(theta_norm) > np.pi / 2:
                gains_linear[i] *= 0.01

        gains_linear /= np.max(gains_linear)

        return self.gain_dBi + 10 * np.log10(gains_linear + 1e-20)

    def cosine_sectoral(self, atten_fac=3):
        gains_linear = np.zeros_like(self.theta_vec)

        for i, theta in enumerate(self.theta_vec):
            theta_norm = np.arctan2(np.sin(theta), np.cos(theta))

            val = np.cos(theta_norm) ** (self.n * atten_fac)
            gains_linear[i] = max(val, 0)

        gains_linear /= np.max(gains_linear)

        return self.gain_dBi + 10 * np.log10(gains_linear + 1e-20)


def plot_GainPattern(gp):

    theta = gp.theta_vec
    gain = gp.gain_dBi_vec

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    ax.plot(theta, gain)
    ax.set_title("Diagrama de Irradiação (dBi)")

    # opcional: ajustar escala
    ax.set_rlabel_position(90)
    ax.grid(True)

    return plt


if __name__ == "__main__":
    gp = GainPattern(
        res_deg=0.1,
        pattern_type="cosine",
        gain_dBi=10,
        beamw_deg=1
    )

    plot = plot_GainPattern(gp)
    plot.show()