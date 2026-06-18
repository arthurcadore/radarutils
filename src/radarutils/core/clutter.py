r"""
clutter.py — Geração de clutter.
"""

import numpy as np

def generate_rayleigh_clutter(n_samples: int, amplitude: float = 1e-6) -> np.ndarray:
    r"""
    Gera ruído complexo gaussiano em banda base que representa clutter com
    estatística de envelope Rayleigh.

    O modelo de envelope Rayleigh é amplamente adotado para representar o
    retorno de múltiplos espalhadores simultâneos (solo, chuva, mar). A parte
    real e imaginária são processos gaussianos independentes e identicamente
    distribuídos, de modo que o envelope segue a distribuição de Rayleigh:

    $$
    p(r) = \frac{r}{\sigma^2} \exp\!\left(-\frac{r^2}{2\sigma^2}\right), \quad r \ge 0
    $$

    onde :math:`\sigma` é o parâmetro de escala relacionado à amplitude.

    Args:
        n_samples (int): Número de amostras complexas (ex.: N_SAMPLES por PRI).
        amplitude (float): Amplitude característica do clutter (controla σ).

    Returns:
        np.ndarray: Sinal complexo de clutter IQ com envelope Rayleigh.

    References:
        Merill I. Skolnik — Introduction To Radar Systems, 3rd Ed. (Cap. 7).
    """
    return amplitude * (
        np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    ) / np.sqrt(2)
