"""Test antenna pattern plots generation."""

import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

import radarutils


class TestAntennaPlots:
    """Test antenna pattern plot generation."""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_plots_dir(self):
        """Create directory for test plots."""
        plots_dir = Path("assets")
        plots_dir.mkdir(exist_ok=True)
        self.__class__.plots_dir = plots_dir
        yield
        # Keep plots for inspection
        
    def test_antenna_patterns_plots(self):
        """Generate antenna pattern plots (from antenna.py __main__)."""
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
        plot1_path = self.plots_dir / "example_antenna_patterns.pdf"
        radarutils.visualization.plotter.save_figure(patterns, str(plot1_path.absolute()))
        plt.close()
        
        assert plot1_path.exists()

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
        plot2_path = self.plots_dir / "example_antenna_pattern_Iso.pdf"
        radarutils.visualization.plotter.save_figure(patternIso, str(plot2_path.absolute()))
        plt.close()
        
        assert plot2_path.exists()
        
        print(f"\nAntenna plots saved to: {self.plots_dir.absolute()}")
        print("Files generated:")
        for plot_file in sorted(self.plots_dir.glob("*antenna*.pdf")):
            print(f"  - {plot_file}")
            # Verify file size
            if plot_file.exists():
                size_kb = plot_file.stat().st_size / 1024
                print(f"    Size: {size_kb:.1f} KB")
                if size_kb < 1:
                    print(f"    WARNING: File appears to be empty!")


if __name__ == "__main__":
    # Run standalone
    pytest.main([__file__, "-v", "-s"])
