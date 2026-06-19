import numpy as np
import pytest

from radarutils.core.integrator import (
    NonCoherentIntegrator,
    CoherentIntegrator,
    integrator_from_str,
)

def test_noncoherent_integrator():
    """Testa a integração não-coerente (soma de potências)."""
    n_int = 4
    integrator = NonCoherentIntegrator(n_int=n_int)
    
    val = 2.0
    signal = np.array([val, val, val])
    
    out = np.zeros_like(signal)
    for _ in range(n_int):
        out = integrator.process(mti_real=signal)
        
    # Não-coerente soma as potências: 4 * (2.0^2) = 16.0
    expected = np.array([16.0, 16.0, 16.0])
    
    print(f"\n[NonCoherent] Sinal de entrada (real): {val} (potência: {val**2})")
    print(f"[NonCoherent] Após {n_int} pulsos - Esperado: {expected[0]}, Obtido: {out[0]}")
    
    np.testing.assert_allclose(out, expected)

def test_coherent_integrator():
    """Testa a integração coerente (soma vetorial IQ e depois potência)."""
    n_int = 4
    integrator = CoherentIntegrator(n_int=n_int)
    
    # Sinal IQ: 1 + 1j (magnitude = sqrt(2), potência = 2)
    val = 1.0 + 1.0j
    signal_cplx = np.array([val, val, val], dtype=complex)
    
    out = np.zeros(3)
    for _ in range(n_int):
        out = integrator.process(mti_real=np.zeros(3), comp_complex=signal_cplx)
        
    # Coerente soma vetorialmente: 4 * (1 + 1j) = 4 + 4j
    # Potência: |4 + 4j|^2 = 16 + 16 = 32.0
    expected = np.array([32.0, 32.0, 32.0])
    
    print(f"\n[Coherent] Sinal de entrada (IQ): {val} (potência: {np.abs(val)**2})")
    print(f"[Coherent] Após {n_int} pulsos - Esperado: {expected[0]}, Obtido: {out[0]}")
    print(f"[Coherent] Note o ganho (SNR linear): N² = {n_int**2} vs N = {n_int} no modo não-coerente.")
    
    np.testing.assert_allclose(out, expected)

def test_coherent_integrator_requires_complex():
    """Verifica se o integrador coerente levanta erro ao não receber sinal complexo."""
    integrator = CoherentIntegrator(n_int=4)
    with pytest.raises(ValueError):
        integrator.process(mti_real=np.array([1.0, 2.0]))

def test_integrator_from_str_factory():
    """Testa se a fábrica instancia corretamente as classes de integrador."""
    n_int = 8
    
    nc = integrator_from_str("noncoherent", n_int)
    assert isinstance(nc, NonCoherentIntegrator)
    assert nc.n_int == n_int
    
    co = integrator_from_str("coherent", n_int)
    assert isinstance(co, CoherentIntegrator)
    assert co.n_int == n_int
    
    # Testa ignorar case e caracteres especiais
    assert isinstance(integrator_from_str("Non-Coherent", n_int), NonCoherentIntegrator)
    assert isinstance(integrator_from_str("COHERENT", n_int), CoherentIntegrator)
    
    with pytest.raises(ValueError):
        integrator_from_str("tipo_invalido", n_int)
