import numpy as np
import matplotlib.pyplot as plt

class GainPattern:
    def __init__(self, res_deg, pattern_type="ideal", gain_dBi=10, beamw_deg=60):
        
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

        self.res_deg = np.deg2rad(res_deg)
        self.pattern_type = pattern_type
        self.gain_dBi = gain_dBi
        self.beamwidth_3dB = np.deg2rad(beamw_deg)
        self.theta_vec = np.arange(0, 2*np.pi + self.res_deg, self.res_deg)
        self.phi_vec   = np.arange(0, 2*np.pi + self.res_deg, self.res_deg)
        self.theta_len = len(self.theta_vec)
        self.phi_len = len(self.phi_vec)

        # Matriz Quadrada horizontal x Vertical 
        self.Hgain_dBi_vec = None
        self.Vgain_dBi_vec = None
        self.gain_dBi_matrix = None

        if self.pattern_type == "isotropic":
            self.gain_dBi_matrix = self.isotropic_gain()
            # gera os vetores de H e V plane com base na primeira coluna e primeira linha.
            self.Hgain_dBi_vec = self.gain_dBi_matrix[0, :]
            self.Vgain_dBi_vec = self.gain_dBi_matrix[:, 0]

        elif self.pattern_type == "ideal":
            self.Hgain_dBi_vec, self.Vgain_dBi_vec  = self.ideal_gain()
                
        elif self.pattern_type == "sinc":
            self.Hgain_dBi_vec, self.Vgain_dBi_vec = self.sinc_gain()

        elif self.pattern_type == "cosine":
            self.Hgain_dBi_vec, self.Vgain_dBi_vec = self.cosine_gain()
        else:
            raise ValueError("Pattern type não suportado")

        self.Hgain_lin_vec = np.power(10, self.Hgain_dBi_vec / 10)
        self.Vgain_lin_vec = np.power(10, self.Vgain_dBi_vec / 10)


    def isotropic_gain(self):
        r"""
            Function to create a isotropic antenna, which in 2D corresponds to the same gain in all possible directions. Based on that, the equation to isotropic can be expressed as: 

            $$
            \begin{equation}
                G(\theta, \phi) = G_i
            \end{equation}
            $$

            Where: 
                - $G$ is the gain of the antenna on a angle $\theta$ and $\phi$
                - $G_i$ is a constant of gain for the antenna.
            
            Returns: 
                gain_dBi_matrix (float): Matrix of gain for $\theta$ and $\phi$.

            <div class="referencia">
                <b>Reference:</b>
                <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Pg - 18) </p>
            </div>
        """

        gain_dBi_matrix = np.full(
            (self.phi_len, self.theta_len),
            self.gain_dBi,
            dtype=float
        )    
        return gain_dBi_matrix


    def ideal_gain(self, atten_db=-40):
        r"""
            Function to create a ideal irradiation pattern for setorial transmission, which corresponds to the $\cos$ decay on beamwidth range and a very low gain outside the beamwidth range. The expression to calculate this pattern is given by: 

            $$
                \begin{equation}
                G(\theta) =
                \begin{cases}
                \cos^n(\theta), & |\theta| \leq \dfrac{\theta_{3\text{dB}}}{2} \\
                \epsilon, & |\theta| > \dfrac{\theta_{3\text{dB}}}{2}
                \end{cases}
                \end{equation}
            $$
            
            Where: 
                - $G$ is the gain of the antenna on a angle $\theta$.
                - $n$ is a variable for the angle directivity. 
                - $\epsilon$ is a attenuation value for outside the beamwidth ($-40dB$ by default).

            Args: 
                atten_db (float): $\epsilon$ is a attenuation value for outside the beamwidth ($-40dB$ by default).

            Returns: 
                H_plane (float): Gain vector of irradiation pattern on Horizontal Axis.
                V_plane (float): Gain vector of irradiation pattern on Vertical Axis.

            <div class="referencia">
                <b>Reference:</b>
                <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Pg - 18) </p>
            </div>
        """
        atten_lin = 10 ** (atten_db / 10)
        theta_3db = self.beamwidth_3dB / 2

        n = np.log(0.5) / np.log(np.cos(theta_3db))
        gains_linear = np.zeros_like(self.theta_vec)

        for i, theta in enumerate(self.theta_vec):
            theta_norm = np.arctan2(np.sin(theta), np.cos(theta))

            if abs(theta_norm) <= theta_3db:
                gains_linear[i] = np.cos(theta_norm) ** n
            else:
                gains_linear[i] = atten_lin

        gains_linear /= np.max(gains_linear)

        H_plane = self.gain_dBi + 10 * np.log10(gains_linear + 1e-20)
        V_plane = H_plane

        return H_plane, V_plane


    def sinc_gain(self, atten_db=10):
        r"""
            Function to create an irradiation pattern based on a squared sinc function, commonly used to approximate the radiation pattern of a uniformly illuminated linear aperture. The normalized gain pattern is given by:

            $$
            \begin{equation}
            G(\theta) = \left[ \frac{\sin(\frac{2\pi}{\theta_{3\text{dB}}} \sin(\theta))}{\frac{2\pi}{\theta_{3\text{dB}}} \sin(\theta)} \right]^2 \cdot e^{-\alpha |\theta|}
            \end{equation}
            $$

            Where:
                - $G(\theta)$ is the normalized antenna gain
                - $\alpha$ is the sidelobe attenuation factor
                - $e^{-\alpha |\theta|}$ is an exponential attenuation used to suppress sidelobes

            Args:
                atten_db (float): Exponential attenuation factor used to suppress sidelobes.

            Returns:
                gain_db_vec (np.ndarray): Gain vector in dB.

            <div class="referencia">
                <b>References:</b>
                <p>
                C. A. Balanis - Antenna Theory: Analysis and Design, 4th Ed., Chapter 10<br>
                R. C. Hansen - Phased Array Antennas
                </p>
            </div>
        """
        theta_norm = np.arctan2(np.sin(self.theta_vec), np.cos(self.theta_vec))
        atten_lin = 10 ** (atten_db / 10)

        k = 2 * np.pi / self.beamwidth_3dB
        u = k * np.sin(theta_norm)

        gains_linear = np.ones_like(u)
        mask = ~np.isclose(u, 0.0)
        gains_linear[mask] = (np.sin(u[mask]) / u[mask]) ** 2

        gains_linear *= np.exp(-atten_lin * np.abs(theta_norm))

        H_plane = self.gain_dBi + 10 * np.log10(gains_linear + 1e-20)
        V_plane = H_plane

        return H_plane, V_plane

    def cosine_gain(self, atten_db=9):
        r"""
            Function to create a sectoral irradiation pattern, similar to a omnidirectional vertical 2D plane, based on a cosine power model. The radiation pattern is defined as:

            $$
            \begin{equation}
            G(\theta) = \cos^{n \cdot \alpha}(\theta)
            \end{equation}
            $$

            Where:
                - $G(\theta)$ is the normalized antenna gain
                - $n$ is a variable for the angle directivity. 
                - $\alpha$ is the sidelobe attenuation factor

            Args:
                atten_db (float): Exponential attenuation factor used to suppress sidelobes.

            Returns:
                gain_db_vec (np.ndarray): Gain vector in dB.

            <div class="referencia">
                <b>References:</b>
                <p>
                C. A. Balanis - Antenna Theory: Analysis and Design, 4th Ed.<br>
                T. S. Rappaport - Wireless Communications: Principles and Practice
                </p>
            </div>
            """

        gains_linear = np.zeros_like(self.theta_vec)
        atten_lin = int(10 ** (atten_db / 10))

        for i, theta in enumerate(self.theta_vec):
            theta_norm = np.arctan2(np.sin(theta), np.cos(theta))

            val = np.cos(theta_norm) ** (10 * atten_lin)
            gains_linear[i] = max(val, 0)

        gains_linear /= np.max(gains_linear)

        H_plane = self.gain_dBi + 10 * np.log10(gains_linear + 1e-20)
        V_plane = np.full_like(self.theta_vec, self.gain_dBi, dtype=float)
        
        return H_plane, V_plane

if __name__ == "__main__":
    from ..visualization.plotter import create_figure, save_figure, GainPatternPlot, GainPattern3DPlot
    
    gp1 = GainPattern(
        res_deg=1,
        pattern_type="isotropic",
        gain_dBi=10,
        beamw_deg=45
    )

    gp2 = GainPattern(
        res_deg=1,
        pattern_type="ideal",
        gain_dBi=10,
        beamw_deg=45
    )

    gp3 = GainPattern(
        res_deg=1,
        pattern_type="sinc",
        gain_dBi=10,
        beamw_deg=45
    )

    gp4 = GainPattern(
        res_deg=1,
        pattern_type="cosine",
        gain_dBi=10,
        beamw_deg=45
    )

    patterns, grid = create_figure(2, 2)

    GainPatternPlot(
        patterns, grid, (0, 0),
        gp=gp1,
        title="Isotropic Pattern V/H Plane",
        colors=["red", "blue"],
    )

    GainPatternPlot(
        patterns, grid, (0, 1),
        gp=gp2,
        title="Ideal Pattern V/H Plane",
        colors=["red", "blue"],
    )

    GainPatternPlot(
        patterns, grid, (1, 0),
        gp=gp3,
        title="Sinc Pattern V/H Plane",
        colors=["red", "blue"],
    )

    GainPatternPlot(
        patterns, grid, (1, 1),
        gp=gp4,
        title="Cosine Pattern V/H Plane",
        colors=["red", "blue"],
        r_min=-100,
        r_max=15,
    )

    save_figure(patterns, "example_antenna_patterns.pdf")


    patternIso, grid = create_figure(1, 2)
    GainPatternPlot(
        patternIso, grid, (0, 0),
        gp=gp1,
        title="Isotropic Pattern V/H Plane",
        colors=["red", "blue"],
        r_min=-15,
        r_max=15,
    )

    GainPattern3DPlot(
        patternIso,
        grid,
        (0, 1),
        gp=gp1,
        title="3D Pattern (Estimated from H/V)",
        cmap="jet"
    )

    save_figure(patternIso, "example_antenna_pattern_Iso.pdf")