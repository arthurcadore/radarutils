#!/usr/bin/env python3
"""
Generate clutter distribution plots.
This script generates plots comparing the empirical histogram of the generated clutter
with the theoretical Probability Density Function (PDF) for each distribution.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from radarutils.core.clutter import RayleighClutter, RiceClutter, WeibullClutter
from radarutils.visualization.plotter import save_figure

def plot_rayleigh(output_dir: Path):
    N = 100000
    A = 2.0
    
    # Generate samples
    clutter = RayleighClutter(n_samples=N, amplitude=A)
    samples = clutter.generate()
    envelope = np.abs(samples)
    
    # Theoretical PDF
    r = np.linspace(0, np.max(envelope), 500)
    pdf = clutter.generate_pdf(r)
    
    plt.figure(figsize=(12, 5))
    plt.hist(envelope, bins=200, density=True, alpha=0.6, color='blue', label='Histograma')
    plt.plot(r, pdf, 'r-', lw=2, label=f'PDF Teórica (A={A})')
    plt.xlabel('Amplitude')
    plt.ylabel('Densidade de Probabilidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure(plt.gcf(), "rayleigh_clutter.pdf")
    plt.close()

def plot_rice(output_dir: Path):
    N = 100000
    A = 2.0
    K = 2.0
    
    # Generate samples
    clutter = RiceClutter(n_samples=N, amplitude=A, k_factor=K)
    samples = clutter.generate()
    envelope = np.abs(samples)
    
    # Theoretical PDF
    r = np.linspace(0, np.max(envelope), 500)
    pdf = clutter.generate_pdf(r)
    
    plt.figure(figsize=(12, 5))
    plt.hist(envelope, bins=200, density=True, alpha=0.6, color='green', label='Histograma')
    plt.plot(r, pdf, 'r-', lw=2, label=f'PDF Teórica (A={A}, K={K})')
    plt.xlabel('Amplitude')
    plt.ylabel('Densidade de Probabilidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure(plt.gcf(), "rice_clutter.pdf")
    plt.close()

def plot_weibull(output_dir: Path):
    N = 100000
    A = 2.0
    c = 1.5
    
    # Generate samples
    clutter = WeibullClutter(n_samples=N, amplitude=A, shape=c)
    samples = clutter.generate()
    envelope = np.abs(samples)
    
    # Theoretical PDF
    r = np.linspace(0.001, np.max(envelope), 500)
    pdf = clutter.generate_pdf(r)
    
    plt.figure(figsize=(12, 5))
    plt.hist(envelope, bins=200, density=True, alpha=0.6, color='purple', label='Histograma')
    plt.plot(r, pdf, 'r-', lw=2, label=f'PDF Teórica (A={A}, c={c})')
    plt.xlabel('Amplitude')
    plt.ylabel('Densidade de Probabilidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure(plt.gcf(), "weibull_clutter.pdf")
    plt.close()

def main():
    plot_rayleigh(None)
    plot_rice(None)
    plot_weibull(None)

if __name__ == "__main__":
    main()
