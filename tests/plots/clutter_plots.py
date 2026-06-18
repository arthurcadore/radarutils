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
from radarutils.visualization.plotter import save_figure, create_figure, PDFplot

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
    fig, grid = create_figure(1, 1, figsize=(12, 5))
    
    PDFplot(
        fig=fig,
        grid=grid,
        pos=(0, 0),
        pdf_x=r,
        pdf_y=pdf,
        hist=True,
        samples=envelope,
        bins=200,
        orientation="vertical",
        legend=f'PDF Teórica (A={A})',
        colors=["blue"]
    )
    
    save_figure(fig, "rayleigh_clutter.pdf")

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
    fig, grid = create_figure(1, 1, figsize=(12, 5))
    
    PDFplot(
        fig=fig,
        grid=grid,
        pos=(0, 0),
        pdf_x=r,
        pdf_y=pdf,
        hist=True,
        samples=envelope,
        bins=200,
        orientation="vertical",
        legend=f'PDF Teórica (A={A}, K={K})',
        colors=["green"]
    )
    
    save_figure(fig, "rice_clutter.pdf")

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
    fig, grid = create_figure(1, 1, figsize=(12, 5))
    
    PDFplot(
        fig=fig,
        grid=grid,
        pos=(0, 0),
        pdf_x=r,
        pdf_y=pdf,
        hist=True,
        samples=envelope,
        bins=200,
        orientation="vertical",
        legend=f'PDF Teórica (A={A}, c={c})',
        colors=["purple"]
    )
    
    save_figure(fig, "weibull_clutter.pdf")

def main():
    plot_rayleigh(None)
    plot_rice(None)
    plot_weibull(None)

if __name__ == "__main__":
    main()
