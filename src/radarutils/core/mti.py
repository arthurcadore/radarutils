r"""
mti.py — Moving Target Indicator.
"""

import numpy as np

class MTI:
    r"""
    Cancelador delay-line de 1 atraso — Moving Target Indicator (MTI).

    Suprime ecos de alvos estacionários (clutter) pelo princípio de que
    retornos fixos possuem amplitude e fase constantes entre PRIs consecutivos.
    A saída é o módulo da diferença entre pulsos adjacentes:

    $$
    y[n] = \bigl|x[n] - x[n-1]\bigr|
    $$

    onde :math:`x[n]` é o sinal do PRI atual (saída do filtro casado) e
    :math:`x[n-1]` é o PRI anterior armazenado no buffer interno.

    O ganho de cancelamento de clutter (Clutter Attenuation — CA) para um
    cancelador de 1 atraso ideal é:

    $$
    \text{CA} = 20 \log_{10}\!\left(2 \sin(\pi f_c T)\right) \; \text{dB}
    $$

    onde :math:`f_c` é a frequência de Doppler do clutter e :math:`T` é o PRI.

    Args:
        n_samples (int): Número de amostras por PRI (define o tamanho do buffer).

    References:
        Merill I. Skolnik — Introduction To Radar Systems, 3rd Ed. (Cap. 3).
    """

    def __init__(self, n_samples: int):
        self._mf_prev = np.zeros(n_samples)

    def process(self, comp_disp: np.ndarray) -> np.ndarray:
        r"""
        Aplica o filtro MTI subtraindo o pulso anterior do atual.

        Args:
            comp_disp (np.ndarray): Sinal real do PRI atual (saída do filtro casado).

        Returns:
            np.ndarray: Sinal pós-MTI :math:`|x[n] - x[n-1]|`.
        """
        mti_output    = np.abs(comp_disp - self._mf_prev)
        self._mf_prev = comp_disp.copy()
        return mti_output
