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

import radarutils


def main():
    """Generate antenna pattern plots."""
    print("Generating antenna pattern plots...")
    
    # Create output directory
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    # Create antenna patterns
    gp1 = radarutils.radar_components.antenna.GainPattern(
        res_deg=1,
        pattern_type="isotropic",
        gain_dBi=10,
        beamw_deg=45
    )

    gp2 = radarutils.radar_components.antenna.GainPattern(
        res_deg=1,
        pattern_type="ideal",
        gain_dBi=10,
        beamw_deg=45
    )

    gp3 = radarutils.radar_components.antenna.GainPattern(
        res_deg=1,
        pattern_type="sinc",
        gain_dBi=10,
        beamw_deg=45
    )

    gp4 = radarutils.radar_components.antenna.GainPattern(
        res_deg=1,
        pattern_type="cosine",
        gain_dBi=10,
        beamw_deg=45
    )

    # Create 2x2 grid plot
    patterns, grid = radarutils.visualization.plotter.create_figure(2, 2)

    radarutils.visualization.plotter.GainPatternPlot(
        patterns, grid, (0, 0),
        gp=gp1,
        title="Isotropic Pattern V/H Plane",
        colors=["red", "blue"],
    )

    radarutils.visualization.plotter.GainPatternPlot(
        patterns, grid, (0, 1),
        gp=gp2,
        title="Ideal Pattern V/H Plane",
        colors=["red", "blue"],
    )

    radarutils.visualization.plotter.GainPatternPlot(
        patterns, grid, (1, 0),
        gp=gp3,
        title="Sinc Pattern V/H Plane",
        colors=["red", "blue"],
    )

    radarutils.visualization.plotter.GainPatternPlot(
        patterns, grid, (1, 1),
        gp=gp4,
        title="Cosine Pattern V/H Plane",
        colors=["red", "blue"],
        r_min=-100,
        r_max=15,
    )

    # Save first plot
    plot1_path = output_dir / "example_antenna_patterns.pdf"
    radarutils.visualization.plotter.save_figure(patterns, str(plot1_path.absolute()))
    print(f"Saved: {plot1_path}")

    # Create 1x2 grid for 3D pattern
    patternIso, grid = radarutils.visualization.plotter.create_figure(1, 2)
    
    radarutils.visualization.plotter.GainPatternPlot(
        patternIso, grid, (0, 0),
        gp=gp1,
        title="Isotropic Pattern V/H Plane",
        colors=["red", "blue"],
        r_min=-15,
        r_max=15,
    )

    radarutils.visualization.plotter.GainPattern3DPlot(
        patternIso,
        grid,
        (0, 1),
        gp=gp1,
        title="3D Pattern (Estimated from H/V)",
        cmap="jet"
    )

    # Save second plot
    plot2_path = output_dir / "example_antenna_pattern_Iso.pdf"
    radarutils.visualization.plotter.save_figure(patternIso, str(plot2_path.absolute()))
    print(f"Saved: {plot2_path}")

    print(f"\nAll plots saved to: {output_dir.absolute()}")
    print("Files generated:")
    for plot_file in sorted(output_dir.glob("*antenna*.pdf")):
        if plot_file.exists():
            size_kb = plot_file.stat().st_size / 1024
            print(f"  - {plot_file} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
