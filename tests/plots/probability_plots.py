#!/usr/bin/env python3
"""
Generate probability plots.
"""

import os
import sys
from pathlib import Path
from radarutils.core.env_vars import *
from radarutils.core.probability import NoiseAWGN
from radarutils.visualization.plotter import PDFplot, create_figure, save_figure

def awgn():

    awgn = NoiseAWGN(n=1000000, sigma=0.5, seed=10)

    pdf = awgn.pdf_values
    x = awgn.x

    fig_gauss, grid_gauss = create_figure(1, 1, figsize=(16, 9))
    PDFplot(
        fig_gauss, grid_gauss, (0,0),
        pdf_x=x,
        pdf_y=pdf,
        samples=awgn.samples,
        variance=awgn.variance_value,
        colors=NOISE_DENSITY_COLOR,
        title=(f"$PDF - Gaussian Noise - \\sigma$ {awgn.sigma}"),
        legend=[r"$p(x)$"],
        ylim=NOISE_DENSITY_YLIM,
        hist=True,
    )
    save_figure(fig_gauss, "pdf-gaussian-noise.pdf")


if __name__ == "__main__":
    awgn()
