#!/usr/bin/env python3
"""
Generate probability plots.
"""

import os
import sys
from pathlib import Path
from radarutils.core.env_vars import *
from radarutils.core.probability import noise_awgn
from radarutils.visualization.plotter import GaussianNoisePlot, create_figure, save_figure

def awgn():

    awgn = noise_awgn(n=1000000, sigma=0.3, seed=10)

    fig_gauss, grid_gauss = create_figure(1, 1, figsize=(16, 9))
    GaussianNoisePlot(
        fig_gauss, grid_gauss, (0,0),
        variance=awgn.variance_value,
        colors=NOISE_DENSITY_COLOR,
        title=(NOISE_DENSITY_TITLE + f" - $\\sigma$ {awgn.sigma}"),
        legend=[r"$p(x)$"],
        ylim=(-2,2),
        span=200,
    ).plot()
    save_figure(fig_gauss, "example_noise_gaussian_sigma.pdf")


if __name__ == "__main__":
    awgn()
