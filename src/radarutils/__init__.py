"""
RadarUtils - A open-source library for Python 3 providing tools and equations for analysis and simulation of radar systems.

Author: Arthur Cadore
Date: 09-03-2026
"""

from . import animations
from . import antenna
from . import basics
from . import data
from . import env_vars
from . import plotter
from . import radar
from . import scene
from . import target
from . import wave

__version__ = "1.0.7"
__author__ = "Arthur Cadore"
__email__ = "arthurbarcella.ita@gmail.com"

__all__ = [
    "animations",
    "antenna", 
    "basics",
    "data",
    "env_vars",
    "plotter",
    "radar",
    "scene",
    "target",
    "wave",
]
