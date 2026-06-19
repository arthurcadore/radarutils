import os
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from radarutils.visualization.plotter import save_figure, create_figure, TimePlot

from radarutils.core.waveform import (
    generate_lfm_chirp,
    build_rx_echo,
    apply_awgn,
    matched_filter,
    doppler_frequency
)
from radarutils.core.integrator import CoherentIntegrator, NonCoherentIntegrator


def plot_integration_chain(coherent: bool):
    """
    Plots the simple processing verification chain:
    Tx -> Rx -> Matched Filter -> Integrator
    
    Args:
        coherent (bool): If True, uses CoherentIntegrator. Otherwise, NonCoherentIntegrator.
    """
    # ── Simulation Parameters ──────────────────────────────────────────
    c = 3e8               # Speed of light (m/s)
    f_c = 1e9             # Carrier frequency (1 GHz)
    B = 5e6               # Chirp bandwidth (5 MHz)
    T_P = 20e-6           # Tx pulse duration (20 µs)
    T_PRI = 150e-6        # Pulse Repetition Interval (150 µs)
    fs = 20e6             # Sampling rate (20 MS/s)
    n_samples = int(T_PRI * fs)
    n_pulse = int(T_P * fs)
    t = np.arange(n_samples) / fs
    chirp_rate = B / T_P
    
    # ── Tx Pulse ─────────────────────────────────────────────────────────
    tx = generate_lfm_chirp(n_samples, n_pulse, chirp_rate, t)
    
    # ── Target ─────────────────────────────────────────────────────────────
    R0 = 10000.0          # Initial distance 10 km
    v_r = 150.0           # Radial velocity 150 m/s
    amplitude = 0.05      # Simulated attenuation from the radar equation
    snr_db = 8.0         # Signal-to-Noise Ratio (AWGN) in dB
    n_int = 5           # Number of pulses to integrate
    
    mti_pulses = []
    out_integrated = np.zeros(n_samples)
    
    if coherent:
        integrator = CoherentIntegrator(n_int)
        mode_str = "Coherent"
    else:
        integrator = NonCoherentIntegrator(n_int)
        mode_str = "Non-Coherent"
        
    # ── Generation and processing of N pulses ──────────────────────────────
    for i in range(n_int):
        t_pri_start = i * T_PRI
        current_R = R0 - v_r * t_pri_start
        current_tau = 2 * current_R / c
        current_delay_samples = int(current_tau * fs)
        current_doppler = doppler_frequency(v_r, f_c, c)
        
        # Carrier phase: phi = 2*pi*f_c*tau
        current_carrier_phase = (2.0 * np.pi * f_c * current_tau) % (2.0 * np.pi)
        
        rx_real, rx_complex = build_rx_echo(
            t=t,
            n_samples=n_samples,
            n_pulse=n_pulse,
            chirp_rate=chirp_rate,
            amplitude=amplitude,
            delay_samples=current_delay_samples,
            carrier_phase=current_carrier_phase,
            doppler_hz=current_doppler,
            tau=current_tau,
        )
        
        # Add AWGN
        rx_noisy = apply_awgn(rx_real, snr_db=snr_db, peak_amplitude=amplitude)
        
        # Matched Filter
        comp_disp, comp_cplx = matched_filter(rx_noisy, rx_complex, tx[:n_pulse], n_pulse)
        
        mti_pulses.append(comp_disp)
        
        # Integration
        out_integrated = integrator.process(comp_disp, comp_cplx)

    # ── Plot 1: Simple chain Tx -> Rx -> MF (1st pulse only) ───────
    fig1, grid1 = create_figure(3, 1, figsize=(16, 8))    
    t_us = t * 1e6
    
    tp1 = TimePlot(fig1, grid1, pos=(0, 0), t=t_us, signals=tx, time_unit="s", 
                   title="Tx Pulse", colors=["blue"], labels=["TX"])
    tp1.plot()
    tp1.ax.set_xlabel("Time (µs)")
    tp1.ax.set_ylabel("Amplitude")
    
    # Recalculating the Rx of the first pulse for plotting
    rx_real_0, rx_complex_0 = build_rx_echo(
        t, n_samples, n_pulse, chirp_rate, amplitude,
        int((2 * R0 / c) * fs), 0.0, doppler_frequency(v_r, f_c, c), 2 * R0 / c
    )
    rx_noisy_0 = apply_awgn(rx_real_0, snr_db=snr_db, peak_amplitude=amplitude)
    
    tp2 = TimePlot(fig1, grid1, pos=(1, 0), t=t_us, signals=rx_noisy_0, time_unit="s",
                   title=f"Rx Pulse (Attenuated + Delayed + AWGN SNR={snr_db}dB)", colors=["orange"], labels=["RX"], amp_norm=True)
    tp2.plot()
    tp2.ax.set_xlabel("Time (µs)")
    tp2.ax.set_ylabel("Amplitude")
    
    tp3 = TimePlot(fig1, grid1, pos=(2, 0), t=t_us, signals=mti_pulses[0], time_unit="s",
                   title="Matched Filter - Pulse 1", colors=["green"], labels=["MF"])
    tp3.plot()
    tp3.ax.set_xlabel("Time (µs)")
    tp3.ax.set_ylabel("Magnitude")
    
    # ── Plot 2: N pulses -> Integration ───────────────────────────────────
    fig2, grid2 = create_figure(2, 1, figsize=(16, 8))
    
    # Colors and labels for multiple pulses
    colors_mp = [f"C{i%10}" for i in range(n_int)]
    labels_mp = [f"Pulse {i+1}" for i in range(n_int)]
    
    tp4 = TimePlot(fig2, grid2, pos=(0, 0), t=t_us, signals=mti_pulses, time_unit="s",
                   title="Post-MF Pulses", colors=colors_mp, labels=labels_mp)
    tp4.plot()
    tp4.ax.set_xlabel("Time (µs)")
    tp4.ax.set_ylabel("Magnitude")
    # Force alpha of signals (optional, manually adjusting the plot lines)
    for line in tp4.ax.lines:
        line.set_alpha(0.6)
        
    tp5 = TimePlot(fig2, grid2, pos=(1, 0), t=t_us, signals=out_integrated, time_unit="s",
                   title=f"Integrated Output ({mode_str})", colors=["red"], labels=["Integrated"], log_y=True, ylim=[0.001, 1000])
    tp5.plot()
    tp5.ax.set_xlabel("Time (µs)")
    
    # Save to assets/
    suffix = "coherent" if coherent else "noncoherent"
    save_figure(fig1, f"integrator_chain_{suffix}.pdf")
    save_figure(fig2, f"integrator_pulses_{suffix}.pdf")

if __name__ == '__main__':
    print("Generating plots for Non-Coherent Integration...")
    plot_integration_chain(coherent=False)
    
    print("Generating plots for Coherent Integration...")
    plot_integration_chain(coherent=True)
