#!/usr/bin/env python3
"""
Generate antenna pattern plots.
This script generates plots outside of pytest to avoid LaTeX dependencies in CI.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from radarutils.core.env_vars import *
from radarutils.core.antenna import GainPattern
from radarutils.visualization.plotter import GainPatternPlot, GainPattern3DPlot, create_figure, save_figure

def main():    
    # Create antenna patterns
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

    # Create 2x2 grid plot
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

    # Save first plot
    save_figure(patterns, "example_antenna_patterns.pdf")

    # Create 1x2 grid for 3D pattern
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

if __name__ == "__main__":
    main()
