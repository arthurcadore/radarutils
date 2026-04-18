# """
# This module contains basic functions for radar calculations;
#
# Author: Arthur Cadore
# """

import numpy as np
from .env_vars import *

def calc_radial_resolution(Tp, c=LIGHT_SPEED):
    r"""
    Calculate the radial resolution ($\delta_r$) of a radar system given its pulse repetition period ($T_p$) and the speed of light ($c$). The radial resolution can be calculated using the following equation:
    
    $$
    \begin{equation}
        \delta_r = \frac{c \cdot T_p}{2}
    \end{equation}
    $$
    
    Where:
        - $\delta_r$ is the radial resolution in meters.
        - $c$ is the speed of light in meters per second.
        - $T_p$ is the pulse repetition period in seconds.
    
    Args:
        Tp (float): Pulse repetition period in seconds.
        c (float): Speed of light in meters per second (default: 299792458).
    
    Returns:
        delta_r (float): Radial resolution in meters.
    
    <div class="referencia">
        <b>Reference:</b>
        <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Pg - 10) </p>
    </div>
    """
    if Tp < 0:
        raise ValueError("Pulse Repetition period (Tp) must be a non-negative value in seconds")
    
    delta_r = (c * Tp) / 2

    return delta_r
    

def calc_time_to_eco(R, c=LIGHT_SPEED):
    r"""
    Calculate the time it takes for a signal to travel from the radar to the target and back.
    
    $$
    \begin{equation}
        t = \frac{2R}{c}
    \end{equation}
    $$
    
    Where:
        - $t$ is the time in seconds.
        - $R$ is the distance in meters.
        - $c$ is the speed of light in meters per second.
    
    Args:
        R (float): Distance in meters.
        c (float): Speed of light in meters per second (default: 299792458).
    
    Returns:
        tte (float): Time to echo in seconds.

    <div class="referencia">
        <b>Reference:</b>
        <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Pg - 10) </p>
    </div>
    """
    if R < 0:
        raise ValueError("Distance (R) must be a non-negative value in meters.")
    
    tte = (2 * R) / c

    return tte

def calc_unambiguous_range(Tp, c=LIGHT_SPEED):
    r"""
    Calculate the maxium radius of detection that doesn't have ambiguity between two (or more) pulses $R_un$. The maximum radius is related with the period of pulses (or with the pulse repetition frequency $PRF$), based on:  

    $$
    \begin{equation}
        R_{un} = \frac{T_p \cdot c}{2}
    \end{equation}
    $$

    Where: 
        - $R_{un}$ is the maxium radius of detection without ambiguity. 
        - $T_p$ is the pulse Repetition period in seconds. 
        - $c$ is the light speed in the medium. 

    Args: 
        Tp (float): Pulse repetition period in seconds. 
        c (integer): Light speed (default $c=299792458$). 

    Returns: 
        run (float): Radius of maxium detection without ambiguity in meters.

    <div class="referencia">
        <b>Reference:</b>
        <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Pg - 10) </p>
    </div>
    """
    if Tp < 0: 
        raise ValueError("Pulse Repetition period (Tp) must be a non-negative value in seconds")
    
    run = (Tp*c)/2

    return run

def calc_max_prf(R, c=LIGHT_SPEED):
    r"""
    Calculate the maximum pulse repetition frequency ($PRF_{max}$) of a radar system given its maximum range ($R$) and the speed of light ($c$). The $PRF_{max}$ can be calculated using the following equation:
    
    $$
    \begin{equation}
        PRF_{max} = \frac{c}{2R}
    \end{equation}
    $$
    
    Where:
        - $PRF_{max}$ is the maximum pulse repetition frequency in Hertz ($Hz$).
        - $c$ is the speed of light in meters per second (default: 299792458).
        - $R$ is the maximum range in meters ($m$).
    
    Args:
        R (float): Maximum range in meters ($m$).
        c (float): Speed of light in meters per second (default: 299792458).
    
    Returns:
        prf_max (float): Maximum pulse repetition frequency in Hertz ($Hz$).
    
    <div class="referencia">
        <b>Reference:</b>
        <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Pg - 10) </p>
    </div>
    """
    if R <= 0:
        raise ValueError("Maximum range (R) must be a positive value in meters.")
    
    prf_max = c / (2 * R)
    
    return prf_max


def calc_effective_area(g, f, c=LIGHT_SPEED):
    r"""
    Calculate the effective area ($A_e$) of an antenna given its gain ($g$) and frequency ($f$). The $A_e$ can be calculated using the following equation:

    $$
    \begin{equation}
        A_e = \frac{G \lambda^2}{4\pi}
    \end{equation}
    $$

    Where: 
        - $A_e$ is the effective area of the antenna in square meters $m^2$.
        - $G$ is the gain of the antenna (in $dBi$).
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

    # Checking Inputs
    if g < 0:
        raise ValueError("Gain (g) must be a non-negative value in dB.")
    g_linear = 10 ** (g / 10)  # Convert gain from dB to linear scale

    if f <= 0:
        raise ValueError("Frequency (f) must be a positive value in Hz.")

    wavelength = c / f 
    Ae = (g_linear * wavelength**2) / (4 * np.pi)
    return Ae


def calc_power_received(Pt, Gt, R, Cs, Ae): 
    r"""

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


class CalcMaxRange: 
        def __init__(self, R_max, Pt, Cs, Pr_min):

            # commom parameters for all methods
            self.Pt = Pt
            self.Cs = Cs
            self.Pr_min = Pr_min
        
            # maximum range calculated 
            self.max_range = R_max

        @classmethod
        def from_range_equation(cls, Pt, Gt, Cs, Ae, Pr_min):
            r"""
            Method to calculate the maximum range ($R_{max}$) of a radar system by using the gain ($G_t$) and effective area ($A_e$) of the antenna. The maximum range can be calculated using the following equation derived from the radar range equation:

            $$
            \begin{equation}
                R_{max} = \left[ \frac{P_t \cdot G_t \cdot A_e \cdot \sigma}{(4\pi)^2 P_{r_{min}}} \right]^{\frac{1}{4}}
            \end{equation}
            $$  

            Where:
                - $R_{max}$ is the maximum range of the radar in meters ($m$).
                - $P_t$ is the transmitted power in watts ($W$).
                - $G_t$ is the gain of the transmitting antenna. 
                - $A_e$ is the effective area of the receiving antenna in square meters ($m^2$).
                - $\sigma$ is the radar cross-section of the target in square meters ($m^2$).
                - $P_{r_{min}}$ is the minimum detectable power by the radar in watts ($W$).

            Args:
                Pt (float): Transmitted power in watts ($W$).
                Gt (float): Gain of the transmitting antenna (in $dBi$).
                Cs (float): Radar cross-section of the target in square meters ($m^2$).
                Ae (float): Effective area of the receiving antenna in square meters ($m^2$).
                Pr_min (float): Minimum detectable power by the radar in watts ($W$).   

            Returns:
                R_max (float): Maximum range of the radar in meters ($m$).    

            """
            if Gt < 0:
                raise ValueError("Gain (Gt) must be a non-negative value in dB.")
            Gt_linear = 10 ** (Gt / 10)  # Convert gain from dB to linear scale

            if Ae <= 0:
                raise ValueError("Effective area (Ae) must be a positive value in square meters.")

            R_max = ((Pt * Gt_linear * Ae * Cs) / ((4 * np.pi)**2 * Pr_min))** (1/4)

            return cls(R_max, Pt, Cs, Pr_min)
    
        @classmethod
        def from_antenna_gain(cls, Pt, Gt, f, Cs, Pr_min, c = LIGHT_SPEED):
            r"""
            Method to calculate the maximum range ($R_{max}$) of a radar system by using the gain ($G_t$) and frequency ($f$) of the antenna. The maximum range can be calculated using the following equation derived from the radar range equation, note that for this equation, the TX and RX antennas are the same, so the gain is squared:

            $$
            \begin{equation}
                R_{max} = \left[ \frac{P_t \cdot G_t^2 \cdot \lambda^2 \cdot \sigma}{(4\pi)^3 P_{r_{min}}} \right]^{\frac{1}{4}}
            \end{equation}
            $$

            Where:
                - $R_{max}$ is the maximum range of the radar in meters ($m$).
                - $P_t$ is the transmitted power in watts ($W$).
                - $G_t$ is the gain of the transmitting antenna.
                - $\lambda$ is the wavelength of the signal, calculated as $\lambda = \frac{c}{f}$.
                - $\sigma$ is the radar cross-section of the target in square meters ($m^2$).
                - $P_{r_{min}}$ is the minimum detectable power by the radar in watts ($W$).

            Args:
                Pt (float): Transmitted power in watts ($W$).
                Gt (float): Gain of the transmitting antenna (in $dBi$).
                f (float): Frequency of operation in $Hz$.
                Cs (float): Radar cross-section of the target in square meters ($m^2$).
                Pr_min (float): Minimum detectable power by the radar in watts ($W$).
                c (float): Speed of light in $m/s$.

            Returns:
                R_max (float): Maximum range of the radar in meters ($m$).

            """

            
            if Gt < 0:
                raise ValueError("Gain (Gt) must be a non-negative value in dB.")
            Gt_linear = 10 ** (Gt / 10)  # Convert gain from dB to linear scale

            if f <= 0:
                raise ValueError("Frequency (f) must be a positive value in Hz.")

            wavelength = c / f
            
            R_max = ((Pt * Gt_linear**2 * wavelength**2 * Cs) / ((4 * np.pi)**3 * Pr_min)) ** (1/4)

            return cls(R_max, Pt, Cs, Pr_min)
        
        @classmethod
        def from_effective_area(cls, Pt, Ae, Cs, f, Pr_min, c = LIGHT_SPEED):

            r"""

            Method to calculate the maximum range ($R_{max}$) of a radar system by using the effective area ($A_e$) and frequency ($f$) of the antenna. The maximum range can be calculated using the following equation derived from the radar range equation, note that for this equation, the TX and RX antennas are the same, so the effective area is squared:

            $$
            \begin{equation}
                R_{max} = \left[ \frac{P_t \cdot A_e^2 \cdot \sigma}{4\pi \cdot \lambda^2 \cdot P_{r_{min}}} \right]^{\frac{1}{4}}
            \end{equation}
            $$  

            Where:
                - $R_{max}$ is the maximum range of the radar in meters ($m$)
                - $P_t$ is the transmitted power in watts ($W$).
                - $A_e$ is the effective area of the receiving antenna in square meters ($m^2$).
                - $\sigma$ is the radar cross-section of the target in square meters ($m^2$).
                - $\lambda$ is the wavelength of the signal, calculated as $\lambda = \frac{c}{f}$.
                - $P_{r_{min}}$ is the minimum detectable power by the radar in watts ($W$).
            
            Args:
                Pt (float): Transmitted power in watts ($W$).
                Ae (float): Effective area of the receiving antenna in square meters ($m^2$).
                Cs (float): Radar cross-section of the target in square meters ($m^2$). 
                f (float): Frequency of operation in $Hz$.
                Pr_min (float): Minimum detectable power by the radar in watts ($W$).
                c (float): Speed of light in $m/s$.

            Returns:
                R_max (float): Maximum range of the radar in meters ($m$).
            
            """
            
            if Ae <= 0:
                raise ValueError("Effective area (Ae) must be a positive value in square meters.")

            if f <= 0:
                raise ValueError("Frequency (f) must be a positive value in Hz.")

            wavelength = c / f
            
            R_max = ((Pt * Ae**2 * Cs) / (4 * np.pi * wavelength**2 * Pr_min))** (1/4)
            return cls(R_max, Pt, Cs, Pr_min)