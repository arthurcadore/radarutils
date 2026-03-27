"""
Test module for radarutils.core.probability module.
"""

import numpy as np
import pytest
from radarutils.core.probability import noise_awgn


class TestNoiseAWGN:
    """Test class for noise_awgn functionality."""

    def test_initialization_basic(self):
        """Test basic initialization of noise_awgn class."""
        n = 1000
        sigma = 1.0
        awgn = noise_awgn(n=n, sigma=sigma)
        
        assert awgn.n == n
        assert awgn.sigma == sigma
        assert len(awgn.samples) == n
        assert isinstance(awgn.variance_value, (float, np.floating))
        assert len(awgn.pdf_values) == n

    def test_initialization_with_seed(self):
        """Test initialization with seed for reproducible results."""
        n = 100
        sigma = 0.5
        seed = 42
        
        awgn1 = noise_awgn(n=n, sigma=sigma, seed=seed)
        awgn2 = noise_awgn(n=n, sigma=sigma, seed=seed)
        
        np.testing.assert_array_equal(awgn1.samples, awgn2.samples)
        np.testing.assert_array_equal(awgn1.pdf_values, awgn2.pdf_values)

    def test_generate_method(self):
        """Test the generate method for AWGN samples."""
        n = 1000
        sigma = 2.0
        awgn = noise_awgn(n=n, sigma=sigma, seed=123)
        
        # Test that samples have correct properties
        assert len(awgn.samples) == n
        assert isinstance(awgn.samples, np.ndarray)
        
        # Test mean is close to zero (within statistical tolerance)
        mean_sample = np.mean(awgn.samples)
        assert abs(mean_sample) < 0.1  # Should be close to 0
        
        # Test standard deviation is close to sigma
        std_sample = np.std(awgn.samples)
        assert abs(std_sample - sigma) < 0.1

    def test_pdf_method(self):
        """Test the probability density function calculation."""
        n = 100
        sigma = 1.5
        awgn = noise_awgn(n=n, sigma=sigma, seed=456)
        
        # Test PDF at zero should be maximum
        pdf_at_zero = awgn.pdf(0.0)
        pdf_at_sigma = awgn.pdf(sigma)
        pdf_at_2sigma = awgn.pdf(2 * sigma)
        
        assert pdf_at_zero > pdf_at_sigma
        assert pdf_at_sigma > pdf_at_2sigma
        
        # Test PDF values are positive
        assert np.all(awgn.pdf_values > 0)
        
        # Test PDF formula correctness at specific points
        expected_pdf_zero = 1.0 / np.sqrt(2 * np.pi * sigma**2)
        np.testing.assert_allclose(pdf_at_zero, expected_pdf_zero, rtol=1e-10)

    def test_variance_method(self):
        """Test the variance calculation from samples."""
        n = 10000  # Large sample for statistical accuracy
        sigma = 0.8
        awgn = noise_awgn(n=n, sigma=sigma, seed=789)
        
        # Variance should be close to sigma^2
        expected_variance = sigma**2
        assert abs(awgn.variance_value - expected_variance) < 0.05

    def test_different_sigma_values(self):
        """Test AWGN with different sigma values."""
        sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
        n = 1000
        seed = 999
        
        for sigma in sigmas:
            awgn = noise_awgn(n=n, sigma=sigma, seed=seed)
            
            # Check that standard deviation matches sigma (within tolerance)
            std_sample = np.std(awgn.samples)
            assert abs(std_sample - sigma) < 0.2
            
            # Check that variance matches sigma^2 (within tolerance)
            # Use relative tolerance for better accuracy across different sigma values
            var_sample = np.var(awgn.samples)
            expected_variance = sigma**2
            relative_error = abs(var_sample - expected_variance) / expected_variance
            assert relative_error < 0.2

    def test_different_sample_sizes(self):
        """Test AWGN with different sample sizes."""
        sample_sizes = [10, 100, 1000, 10000]
        sigma = 1.0
        seed = 555
        
        for n in sample_sizes:
            awgn = noise_awgn(n=n, sigma=sigma, seed=seed)
            
            assert len(awgn.samples) == n
            assert len(awgn.pdf_values) == n
            assert isinstance(awgn.variance_value, (float, np.floating))

    def test_statistical_properties(self):
        """Test statistical properties of generated AWGN."""
        n = 100000  # Large sample for better statistics
        sigma = 1.0
        awgn = noise_awgn(n=n, sigma=sigma, seed=42)
        
        # Test mean is approximately zero
        mean = np.mean(awgn.samples)
        assert abs(mean) < 0.01  # Should be very close to 0 for large n
        
        # Test standard deviation is approximately sigma
        std = np.std(awgn.samples)
        assert abs(std - sigma) < 0.01
        
        # Test variance is approximately sigma^2
        variance = np.var(awgn.samples)
        assert abs(variance - sigma**2) < 0.01
        
        # Test skewness is approximately zero (Gaussian property)
        skewness = np.mean(((awgn.samples - mean) / std)**3)
        assert abs(skewness) < 0.1
        
        # Test kurtosis is approximately 3 (Gaussian property)
        kurtosis = np.mean(((awgn.samples - mean) / std)**4)
        assert abs(kurtosis - 3) < 0.2

    def test_pdf_normalization(self):
        """Test that PDF integrates to approximately 1."""
        sigma = 1.0
        awgn = noise_awgn(n=1000, sigma=sigma, seed=123)
        
        # Create a range of values for integration
        x = np.linspace(-5*sigma, 5*sigma, 1000)
        pdf_values = awgn.pdf(x)
        
        # Numerical integration using trapezoidal rule
        integral = np.trapezoid(pdf_values, x)
        
        # Should be close to 1 (within numerical integration error)
        assert abs(integral - 1.0) < 0.01

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small sigma
        awgn_small = noise_awgn(n=100, sigma=1e-6, seed=42)
        assert np.all(np.abs(awgn_small.samples) < 1e-4)  # Samples should be very small
        
        # Test with single sample
        awgn_single = noise_awgn(n=1, sigma=1.0, seed=42)
        assert len(awgn_single.samples) == 1
        assert len(awgn_single.pdf_values) == 1

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        n = 1000
        sigma = 1.5
        seed = 777
        
        # Generate multiple instances with same seed
        instances = [noise_awgn(n=n, sigma=sigma, seed=seed) for _ in range(3)]
        
        # All should be identical
        for i in range(1, len(instances)):
            np.testing.assert_array_equal(instances[0].samples, instances[i].samples)
            np.testing.assert_array_equal(instances[0].pdf_values, instances[i].pdf_values)
            assert instances[0].variance_value == instances[i].variance_value

    def test_pdf_symmetry(self):
        """Test that PDF is symmetric around zero."""
        sigma = 1.0
        awgn = noise_awgn(n=100, sigma=sigma, seed=456)
        
        # Test symmetry at various points
        test_points = [0.1, 0.5, 1.0, 2.0]
        
        for x in test_points:
            pdf_pos = awgn.pdf(x)
            pdf_neg = awgn.pdf(-x)
            np.testing.assert_allclose(pdf_pos, pdf_neg, rtol=1e-10)
