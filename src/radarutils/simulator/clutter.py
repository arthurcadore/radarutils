r"""
clutter.py — Modelagem de clutter para a simulação de radar.

Este módulo concentra as rotinas necessárias para simular os ecos indesejados
originados do ambiente, como retornos de solo, chuva ou mar.
"""

import numpy as np

def generate_rayleigh_clutter(n_samples: int, amplitude: float = 1e-6) -> np.ndarray:
    r"""
    Gera ruído complexo gaussiano em banda base que representa clutter
    com estatística de envelope Rayleigh.

    Esse tipo de ruído é frequentemente utilizado como modelo estatístico
    para retorno de múltiplos espalhadores simultâneos.

    Args:
        n_samples (int): Número de amostras complexas ao longo do tempo (ex: N_SAMPLES).
        amplitude (float): Valor representativo da amplitude do sinal de clutter.

    Returns:
        np.ndarray: Sinal complexo representando a adição de clutter aos sinais IQ.
    """
    return amplitude * (
        np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    ) / np.sqrt(2)
