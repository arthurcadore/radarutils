import numpy as np
import pytest
from scipy.special import i0

from radarutils.core.clutter import (
    RayleighClutter,
    RiceClutter,
    WeibullClutter,
    clutter_from_str,
)

def test_rayleigh_distribution():
    N = 200000
    A = 2.0
    
    clutter = RayleighClutter(n_samples=N, amplitude=A)
    samples = clutter.generate()
    envelope = np.abs(samples)
    
    # Histograma empírico (densidade)
    hist, bin_edges = np.histogram(envelope, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # PDF Teórica
    pdf = clutter.generate_pdf(bin_centers)
    
    # Erro Quadrático Médio (MSE)
    mse = np.mean((hist - pdf)**2)
    print(f"\nErro da distribuição Rayleigh: {mse}")
    
    # Margem de erro bem conservadora
    assert mse < 1e-3, f"Erro da distribuição Rayleigh está muito alto: MSE={mse}"

def test_rice_distribution():
    N = 200000
    A = 2.0
    K = 2.0
    
    clutter = RiceClutter(n_samples=N, amplitude=A, k_factor=K)
    samples = clutter.generate()
    envelope = np.abs(samples)
    
    # Histograma empírico (densidade)
    hist, bin_edges = np.histogram(envelope, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # PDF Teórica
    pdf = clutter.generate_pdf(bin_centers)
    
    # Erro Quadrático Médio (MSE)
    mse = np.mean((hist - pdf)**2)
    print(f"\nErro da distribuição Rice: {mse}")
    
    assert mse < 1e-3, f"Erro da distribuição Rice está muito alto: MSE={mse}"

def test_weibull_distribution():
    N = 200000
    A = 2.0
    K = 1.5  # shape
    
    clutter = WeibullClutter(n_samples=N, amplitude=A, shape=K)
    samples = clutter.generate()
    envelope = np.abs(samples)
    
    # Histograma empírico (densidade)
    hist, bin_edges = np.histogram(envelope, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # PDF Teórica
    pdf = clutter.generate_pdf(bin_centers)
    
    # Erro Quadrático Médio (MSE)
    mse = np.mean((hist - pdf)**2)
    print(f"\nErro da distribuição Weibull: {mse}")
    
    assert mse < 1e-3, f"Erro da distribuição Weibull está muito alto: MSE={mse}"

def test_clutter_from_str_factory():
    """Testa se a fábrica de clutter instancia corretamente as classes."""
    N = 100
    
    assert clutter_from_str("none", N) is None
    assert clutter_from_str("", N) is None
    
    rayleigh = clutter_from_str("rayleigh", N)
    assert isinstance(rayleigh, RayleighClutter)
    
    rice = clutter_from_str("rice", N)
    assert isinstance(rice, RiceClutter)
    
    weibull = clutter_from_str("weibull", N)
    assert isinstance(weibull, WeibullClutter)
    
    with pytest.raises(ValueError):
        clutter_from_str("distribuicao_invalida", N)
