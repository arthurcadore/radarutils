r"""
cfar.py — Detector CA-CFAR.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

def ca_cfar(
    signal: np.ndarray,
    n_guard: int,
    n_train: int,
    alpha: float,
) -> np.ndarray:
    r"""
    Detector CA-CFAR vetorizado via ``uniform_filter1d``.

    Estima o piso de ruído local de cada célula sob teste (CUT) usando a média
    das células de treinamento vizinhas, separadas por células de guarda, e
    define o limiar adaptativo como:

    $$
    T_i = \alpha \cdot \hat{P}_{\text{noise},i}
    $$

    onde :math:`\hat{P}_{\text{noise},i}` é a média de potência nas
    :math:`2 N_{\text{train}}` células de treinamento ao redor da CUT :math:`i`
    (excluindo as :math:`2 N_{\text{guard}}` células de guarda).

    Para uma distribuição exponencial do ruído (envelope Rayleigh ao quadrado),
    a probabilidade de falso alarme teórica é:

    $$
    P_{\text{FA}} = \left(1 + \frac{\alpha}{N_{\text{train}}}\right)^{-N_{\text{train}}}
    $$

    A implementação usa ``uniform_filter1d`` com ``mode='reflect'`` para evitar
    subestimação do piso de ruído nas bordas (o que causaria falsos alarmes nas
    extremidades do vetor).

    Args:
        signal (np.ndarray): Sinal de entrada (potência, após integração e MTI).
        n_guard (int): Número de células de guarda de cada lado da CUT.
        n_train (int): Número de células de treinamento de cada lado.
        alpha (float): Fator multiplicativo do limiar (controla P_FA).

    Returns:
        np.ndarray: Vetor de limiares adaptativos, um por célula do sinal.

    References:
        P. P. Gandhi & S. A. Kassam — "Analysis of CFAR Processors in
        Nonhomogeneous Background", IEEE Trans. AES, 1988.
    """
    win_t = 2 * (n_guard + n_train) + 1
    win_g = 2 * n_guard + 1

    sum_t = uniform_filter1d(signal, size=win_t, mode='reflect') * win_t
    sum_g = uniform_filter1d(signal, size=win_g, mode='reflect') * win_g
    n_tr  = max(win_t - win_g, 1)

    return np.maximum(alpha * (sum_t - sum_g) / n_tr, 0.0)
