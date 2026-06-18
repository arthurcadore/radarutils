"""Core module for radarutils.

This module contains fundamental radar calculations, data structures, and
environmental variables, as well as theoretical signal processing algorithms
(MTI, pulse integration, CA-CFAR, clutter, waveform generation).
"""

from . import basics
from . import data
from . import env_vars
from . import clutter
from . import mti
from . import integrator
from . import cfar
from . import waveform

__all__ = [
    "basics",
    "data",
    "env_vars",
    "clutter",
    "mti",
    "integrator",
    "cfar",
    "waveform",
]
