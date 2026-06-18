#!/usr/bin/env python3
"""
Generate clutter distribution plots.
This script generates plots comparing the empirical histogram of the generated clutter
with the theoretical Probability Density Function (PDF) for each distribution.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0

# Add src to path for imports
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
    sigma = A / np.sqrt(2)
    r = np.linspace(0, np.max(envelope), 500)
    pdf = (r / sigma**2) * np.exp(-r**2 / (2 * sigma**2))
    
    plt.figure(figsize=(10, 6))
    plt.hist(envelope, bins=200, density=True, alpha=0.6, color='blue', label='Histograma')
    plt.plot(r, pdf, 'r-', lw=2, label=f'PDF Teórica (A={A})')
    plt.title('Distribuição de Clutter Rayleigh')
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
    sigma = A / np.sqrt(2 * (K + 1))
    nu = A * np.sqrt(K / (K + 1))
    r = np.linspace(0, np.max(envelope), 500)
    
    pdf = (r / sigma**2) * np.exp(-(r**2 + nu**2) / (2 * sigma**2)) * i0(r * nu / sigma**2)
    
    plt.figure(figsize=(10, 6))
    plt.hist(envelope, bins=200, density=True, alpha=0.6, color='green', label='Histograma')
    plt.plot(r, pdf, 'r-', lw=2, label=f'PDF Teórica (A={A}, K={K})')
    plt.title(f'Distribuição de Clutter Rice (K={K})')
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
    K = 1.5
    
    # Generate samples
    clutter = WeibullClutter(n_samples=N, amplitude=A, shape=K)
    samples = clutter.generate()
    envelope = np.abs(samples)
    
    # Theoretical PDF
    lam = A
    c = K
    r = np.linspace(0.001, np.max(envelope), 500)
    
    pdf = (c / lam) * (r / lam)**(c - 1) * np.exp(-(r / lam)**c)
    
    plt.figure(figsize=(10, 6))
    plt.hist(envelope, bins=200, density=True, alpha=0.6, color='purple', label='Histograma')
    plt.plot(r, pdf, 'r-', lw=2, label=f'PDF Teórica (A={A}, K={K})')
    plt.title(f'Distribuição de Clutter Weibull (K={K})')
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
    
    print("Plots de clutter gerados com sucesso na pasta de assets!")

if __name__ == "__main__":
    main()
