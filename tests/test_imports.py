"""Test imports for radarutils package."""

import pytest


def test_import_radarutils():
    """Test that radarutils can be imported."""
    import radarutils
    assert radarutils is not None


def test_import_core():
    """Test that core module can be imported."""
    import radarutils
    assert radarutils.core is not None


def test_import_core_modules():
    """Test that core submodules can be imported."""
    import radarutils
    assert radarutils.core.basics is not None
    assert radarutils.core.data is not None
    assert radarutils.core.env_vars is not None

def test_import_visualization():
    """Test that visualization module can be imported."""
    import radarutils
    assert radarutils.visualization is not None


def test_import_visualization_modules():
    """Test that visualization submodules can be imported."""
    import radarutils
    assert radarutils.visualization.plotter is not None
