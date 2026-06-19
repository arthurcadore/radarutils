"""
Radar Simulator Package
Provides classes for simulating radar scenes, components, and PPI visualization.
"""

from radarutils.simulator.pulse import WaveformParams, waveform_params_from_rmax
from radarutils.simulator import pipeline
from radarutils.simulator.pipeline import (
    frontend_stage,
    build_rx_signal,
    add_awgn_stage,
    apply_clutter,
    matched_filter_stage,
    mti_stage,
    integration_stage,
    cfar_stage,
    estimated_ppi_stage,
)
