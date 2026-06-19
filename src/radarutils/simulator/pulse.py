r"""
pulse.py — Definição do pulso modulado TX (chirp LFM).

Centraliza os parâmetros derivados da forma de onda e a geração do sinal
TX, separando a configuração do transmissor da lógica de processamento
do pipeline.

Contém:
  - ``WaveformParams``: dataclass com todos os parâmetros derivados do pulso.
  - ``waveform_params_from_rmax()``: factory que calcula WaveformParams a partir
    do alcance máximo do radar e potência transmitida.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radarutils.simulator.constants import C, B, N_SAMPLES
from radarutils.core.waveform import generate_lfm_chirp


# ──────────────────────────────────────────────────────────────────────────
#  Parâmetros de forma de onda (waveform params)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class WaveformParams:
    """
    Parâmetros derivados da forma de onda LFM para um dado r_max.

    Estes parâmetros são usados em toda a cadeia de processamento e podem
    ser gerados de forma conveniente via :func:`waveform_params_from_rmax`.

    Attributes:
        T_PRI (float): Período de repetição de pulso (s).
        T_P (float): Duração do pulso transmitido (s).
        fs (float): Taxa de amostragem (Hz).
        t (np.ndarray): Vetor de tempo para um PRI completo (s).
        t_us (np.ndarray): Vetor de tempo em µs (para plots).
        k (float): Taxa de varredura do chirp LFM (Hz/s).
        n_p (int): Número de amostras do pulso TX.
        tx (np.ndarray): Sinal TX (chirp LFM real, normalizado).
        P_tx_dbm (float): Potência TX em dBm.
    """
    T_PRI: float
    T_P: float
    fs: float
    t: np.ndarray
    t_us: np.ndarray
    k: float
    n_p: int
    tx: np.ndarray
    P_tx_dbm: float = 60.0


def waveform_params_from_rmax(r_max: float, pt: float = 1000.0) -> WaveformParams:
    """
    Deriva os parâmetros de forma de onda LFM a partir do alcance máximo.

    A relação usada é::

        T_PRI = 2 * r_max / c
        T_P   = T_PRI / 7
        fs    = N_SAMPLES / T_PRI
        k     = B / T_P

    Args:
        r_max (float): Alcance máximo do radar (m).
        pt (float): Potência transmitida em Watts. Padrão: 1000 W.

    Returns:
        WaveformParams: Estrutura com todos os parâmetros da forma de onda.
    """
    T_PRI = 2.0 * r_max / C
    T_P   = T_PRI / 7.0
    fs    = N_SAMPLES / T_PRI
    t     = np.linspace(0, T_PRI, N_SAMPLES, endpoint=False)
    t_us  = t * 1e6
    k     = B / T_P
    n_p   = int(T_P * fs)
    tx    = generate_lfm_chirp(N_SAMPLES, n_p, k, t)
    P_tx_dbm = 10.0 * np.log10(pt * 1e3)

    return WaveformParams(
        T_PRI=T_PRI, T_P=T_P, fs=fs, t=t, t_us=t_us,
        k=k, n_p=n_p, tx=tx, P_tx_dbm=P_tx_dbm,
    )
