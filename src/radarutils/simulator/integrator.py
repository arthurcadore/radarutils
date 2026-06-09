import numpy as np
from collections import deque

class PulseIntegrator:
    """
    Realiza a integração de pulsos, coerente ou não-coerente, com base
    num histórico (buffer) dos últimos N_INT PRIs (Pulse Repetition Intervals).
    """
    def __init__(self, n_int: int, coherent: bool):
        self.n_int = n_int
        self.coherent = coherent
        self._buffer = deque(maxlen=n_int)

    def process(self, mti_real: np.ndarray, comp_complex: np.ndarray = None) -> np.ndarray:
        """
        Processa os sinais e atualiza o estado de integração.
        
        Args:
            mti_real: Sinal real após processamento MTI (para integração não-coerente).
            comp_complex: Sinal complexo mantendo a fase (para integração coerente).

        Returns:
            Sinal integrado e amplificado.
        """
        if self.coherent:
            # Integração Coerente: soma amplitudes complexas → |soma| melhora SNR de N_INT × em amplitude
            if comp_complex is None:
                raise ValueError("comp_complex must be provided for coherent integration.")
            self._buffer.append(comp_complex)
            coh_sum = np.sum(list(self._buffer), axis=0)   # soma complexa
            return np.abs(coh_sum) ** 2                    # envelope de potência
        else:
            # Integração Não-Coerente: soma de |mti|² dos últimos N_INT PRIs
            self._buffer.append(mti_real ** 2)
            return np.sum(list(self._buffer), axis=0)

