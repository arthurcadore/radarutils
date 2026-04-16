"""Test basic functionality of radarutils.basics module."""

import pytest
import numpy as np
from radarutils.core.basics import *


def test_CalcMaxRange():
    """Test CalcMaxRange class."""
    g = 30 
    f = 3e9
    r = 1000
    Pt = 1000
    Cs = 1

    Ae = calc_effective_area(g, f)
    Pr_min = 1e-10

    range1 = CalcMaxRange.from_range_equation(Pt, g, Cs, Ae, Pr_min).max_range
    range2 = CalcMaxRange.from_antenna_gain(Pt, g, f, Cs, Pr_min).max_range
    range3 = CalcMaxRange.from_effective_area(Pt, Ae, Cs, f, Pr_min).max_range

    assert np.isclose(range1, range2, rtol=1e-10)
    assert np.isclose(range2, range3, rtol=1e-10)

def test_calc_unambiguous_range():
    """Test unambiguous range calculation."""
    # Test with known values
    Tp = 1e-3  # 1 ms period
    c = 3e8    # Speed of light
    
    result = calc_unambiguous_range(Tp, c)
    expected = (Tp * c) / 2
    assert np.isclose(result, expected, rtol=1e-10)
    assert result > 0


def test_calc_unambiguous_range_negative():
    """Test that negative period raises ValueError."""
    with pytest.raises(ValueError):
        calc_unambiguous_range(-1.0)


def test_effective_area():
    """Test effective area calculation."""
    gain_db = 20  # 20 dB gain (100 linear)
    frequency = 10e9  # 10 GHz
    c = 3e8
    
    result = calc_effective_area(gain_db, frequency, c)
    wavelength = c / frequency
    gain_linear = 10 ** (gain_db / 10)  # Convert dB to linear
    expected = (gain_linear * wavelength**2) / (4 * np.pi)
    assert np.isclose(result, expected, rtol=1e-10)
    assert result > 0

def test_array_inputs():
    """Test that functions work with different input types."""
    # Test with float
    period_float = 1e-3
    range_float = calc_unambiguous_range(period_float)
    expected_float = (period_float * 299792458) / 2  # Use exact speed of light
    assert np.isclose(range_float, expected_float, rtol=1e-10)
    
    # Test with integer
    period_int = 1  # 1 second
    range_int = calc_unambiguous_range(period_int)
    expected_int = (period_int * 299792458) / 2  # Use exact speed of light
    assert np.isclose(range_int, expected_int, rtol=1e-10)
    
    # Test effective area with different frequency values
    gain_db = 10
    freq1 = 1e9
    freq2 = 10e9
    
    ae1 = calc_effective_area(gain_db, freq1)
    ae2 = calc_effective_area(gain_db, freq2)
    
    # Higher frequency should result in smaller effective area
    assert ae2 < ae1
    assert ae1 > 0 and ae2 > 0

def test_frequency_wavelength_relationship():
    """Test frequency and wavelength relationship through effective_area function."""
    gain_db = 20  # 20 dB gain
    frequency = 10e9
    c = 299792458
    
    # Calculate effective area
    ae = calc_effective_area(gain_db, frequency, c)
    
    # Verify it's positive and reasonable
    assert ae > 0
    
    # Test with different frequency
    frequency2 = 20e9  # Double frequency
    ae2 = calc_effective_area(gain_db, frequency2, c)
    
    # At double frequency, wavelength is half, so effective area should be quarter
    # (since Ae is proportional to wavelength^2)
    expected_ratio = 0.25
    actual_ratio = ae2 / ae
    assert np.isclose(actual_ratio, expected_ratio, rtol=1e-10)
