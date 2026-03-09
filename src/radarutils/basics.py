# """
# Implementation of BER vs Eb/N0 simulation for the ARGOS-3 standard.
#
# Author: Arthur Cadore
# Date: 8-09-2025
# """

import numpy as np
from .env_vars import LIGHT_SPEED

def calculate_Ae(g, f, c=LIGHT_SPEED):
    r"""
    Calculate the effective area ($A_e$) of an antenna given its gain ($g$) and frequency ($f$). The $A_e$ can be calculated using the following equation:

    $$
    \begin{equation}
        A_e = \frac{G \lambda^2}{4\pi}
    \end{equation}
    $$

    Where: 
        - $A_e$ is the effective area of the antenna in square meters $m^2$.
        - $G$ is the linear gain of the antenna.
        - $\lambda$ is the wavelength of the signal, calculated as $\lambda = \frac{c}{f}$.
        - $c$ is the speed of light in $m/s$.

    Args:
        g (float): Gain of the antenna in $dB$.
        f (float): Frequency of operation in $Hz$.
        c (float): Speed of light in $m/s$.

    Returns:
        Ae (float): Effective area ($A_e$) of the antenna in square meters ($m^2$).

    <div class="referencia">
        <b>Reference:</b>
        <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Pg - 18) </p>
    </div>
    """

    g_linear = 10 ** (g / 10)  # Convert gain from dB to linear scale
    wavelength = c / f 
    Ae = (g_linear * wavelength**2) / (4 * np.pi)
    return Ae

if __name__ == "__main__":

    # antenna effective area example
    g = 30  # Gain in dB
    f = 3e9  # Frequency in Hz
    
    Ae = calculate_Ae(g, f)
    print(f"Ae: {Ae} m^2")

