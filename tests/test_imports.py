"""Test imports for radarutils package."""

import pytest


def test_import_radarutils():
    """Test that radarutils can be imported."""
    import radarutils
    assert radarutils is not None


def test_import_basics():
    """Test that basics module can be imported."""
    from radarutils import basics
    assert basics is not None


def test_import_antenna():
    """Test that antenna module can be imported."""
    from radarutils import antenna
    assert antenna is not None


def test_import_radar():
    """Test that radar module can be imported."""
    from radarutils import radar
    assert radar is not None


def test_import_target():
    """Test that target module can be imported."""
    from radarutils import target
    assert target is not None


def test_import_wave():
    """Test that wave module can be imported."""
    from radarutils import wave
    assert wave is not None


def test_import_scene():
    """Test that scene module can be imported."""
    from radarutils import scene
    assert scene is not None


def test_import_plotter():
    """Test that plotter module can be imported."""
    from radarutils import plotter
    assert plotter is not None


def test_import_animations():
    """Test that animations module can be imported."""
    from radarutils import animations
    assert animations is not None


def test_import_data():
    """Test that data module can be imported."""
    from radarutils import data
    assert data is not None
