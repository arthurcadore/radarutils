r"""
integrator.py — Integrador de pulsos.
"""

import numpy as np
from collections import deque

class PulseIntegrator:
    r"""
    Integrador de pulsos — coerente ou não-coerente.

    Mantém um buffer circular dos últimos ``n_int`` PRIs e acumula energia
    para melhorar a relação sinal-ruído (SNR).

    **Integração Não-Coerente** (detecção de envelope):

    $$
    z[n] = \sum_{k=0}^{N_{\text{int}}-1} |y_k[n]|^2
    $$

    Ganho de SNR ≈ :math:`\sqrt{N_{\text{int}}}` (≈ :math:`10\log_{10}(N_{\text{int}})/2` dB).

    **Integração Coerente** (preservação de fase IQ):

    $$
    z[n] = \left|\sum_{k=0}^{N_{\text{int}}-1} \tilde{y}_k[n]\right|^2
    $$

    Ganho de SNR ≈ :math:`N_{\text{int}}` (≈ :math:`20\log_{10}(N_{\text{int}})` dB),
    porém exige coerência de fase entre PRIs.

    Args:
        n_int (int): Número de PRIs a integrar.
        coherent (bool): Se ``True``, realiza integração coerente (IQ complexo).

    References:
        Merill I. Skolnik — Introduction To Radar Systems, 3rd Ed. (Cap. 2).
    """

    def __init__(self, n_int: int, coherent: bool):
        self.n_int    = n_int
        self.coherent = coherent
        self._buffer  = deque(maxlen=n_int)

    def process(
        self,
        mti_real: np.ndarray,
        comp_complex: np.ndarray = None,
    ) -> np.ndarray:
        r"""
        Processa um PRI e retorna o sinal integrado acumulado.

        Args:
            mti_real (np.ndarray): Sinal real pós-MTI (para integração não-coerente).
            comp_complex (np.ndarray | None): Sinal IQ complexo do filtro casado
                (obrigatório se ``coherent=True``).

        Returns:
            np.ndarray: Sinal integrado (potência acumulada).

        Raises:
            ValueError: Se ``coherent=True`` e ``comp_complex`` não for fornecido.
        """
        if self.coherent:
            if comp_complex is None:
                raise ValueError(
                    "comp_complex deve ser fornecido para integração coerente."
                )
            self._buffer.append(comp_complex)
            coh_sum = np.sum(list(self._buffer), axis=0)   # soma complexa
            return np.abs(coh_sum) ** 2                    # envelope de potência
        else:
            self._buffer.append(mti_real ** 2)
            return np.sum(list(self._buffer), axis=0)
