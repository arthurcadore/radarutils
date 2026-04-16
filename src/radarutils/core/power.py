
import numpy as np 

def mag_to_db(signal: np.ndarray) -> np.ndarray:
    r"""
    Converts the signal magnitude to a logarithmic scale ($dB$). The conversion process is given by the expression below.

    $$
     dB(x) = 20 \log_{10}\left(\frac{|x|}{x_{peak} + 10^{-12}}\right)
    $$

    Where:
        - $x$: Signal to be converted to $dB$.
        - $x_{peak}$: Peak magnitude of the signal.
        - $10^{-12}$: Constant to avoid division by zero.
    
    Args:
        signal: Array with signal data
        
    Returns:
        Array with signal converted to $dB$
    """
    mag = np.abs(signal)
    peak = np.max(mag) if np.max(mag) != 0 else 1.0
    mag = mag / peak
    return 20 * np.log10(mag + 1e-12)


def db_to_linear(db: float) -> float:
    r"""
    Convert dB to linear scale.

    $$
    \begin{equation}
        P = 10^{\frac{P_{dB}}{10}}
    \end{equation}
    $$

    where:
        - $P$ is the power in linear scale
        - $P_{dB}$ is the power in dB
    
    Args:
        db (float): Power in dB
        
    Returns:
        P_lin (float): Power in linear scale
    """

    P_lin = 10 ** (db / 10)
    return P_lin

def linear_to_db(linear: float) -> float:
    r"""
    Convert linear scale to dB.
    
    $$
    \begin{equation}
        P_{dB} = 10 \log_{10}(P_{linear})
    \end{equation}
    $$

    where:
        - $P_{linear}$ is the power in linear scale
        - $P_{dB}$ is the power in dB
    
    Args:
        linear (float): Power in linear scale
        
    Returns:
        P_db (float): Power in dB
    """
    P_db = 10 * np.log10(linear)
    return P_db


class CalcRadarPower:
    """
    To calculate the power received ($P_r$) by a radar system, we can use the radar range equation, which is given by: 

    $$
    \begin{equation}
        P_r = \frac{P_t \cdot G_t \cdot \sigma \cdot A_e}{(4\pi)^2 R^4}
    \end{equation}
    $$

    Where:
        - $P_r$ is the power received by the radar in watts ($W$).
        - $P_t$ is the transmitted power in watts ($W$).
        - $G_t$ is the gain of the transmitting antenna.
        - $\sigma$ is the radar cross-section of the target in square meters ($m^2$).
        - $A_e$ is the effective area of the receiving antenna in square meters ($m^2$).
        - $R$ is the range to the target in meters ($m$).
        

    """
    def __init__(self, Pr, Pt):
        pass

    def at_distance(self, D):
        pass
    def at_radar_returned(self, A_e):
        pass
    def at_radar_absorbed(self):
        pass

def power_received(Pt, Gt, R, Cs, Ae): 
    r"""
    Args: 
        Pt (float): Transmitted power in watts ($W$).
        Gt (float): Gain of the transmitting antenna (in $dBi$).
        R (float): Range to the target in meters ($m$).
        Cs (float): Radar cross-section of the target in square meters ($m^2$).
        Ae (float): Effective area of the receiving antenna in square meters ($m^2$).

    Returns:
        Pr (float): Power received by the radar in watts ($W$).
        Pr_dBW (float): Power received by the radar in decibels with respect to one watt ($dBW$).

    <div class="referencia">
        <b>Reference:</b>
        <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Pg - 18) </p>
    </div>
    """

    # Checking Inputs
    if Gt < 0:
        raise ValueError("Gain (Gt) must be a non-negative value in dB.")
    Gt_linear = 10 ** (Gt / 10)  # Convert gain from dB to linear scale

    if R <= 0:
        raise ValueError("Range (R) must be a positive value in meters.")

    P_target = Pt * Gt_linear / (4 * np.pi * R**2)  # Power density at the target
    P_returned = (P_target * Cs) / (4 * np.pi * R**2)  # Power reflected back to the radar 
    Pr = P_returned * Ae  # Power absorbed by the radar

    # Checking if Pr is greater than 0
    if Pr < 0:
        raise ValueError("Power received (Pr) must be a non-negative value.")
    Pr_dBW = 10 * np.log10(Pr)  # Convert power received to dBW
    
    return Pr, Pr_dBW, P_target, P_returned