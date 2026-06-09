import numpy as np

class MTI:
    """
    Cancelador delay-line (1 atraso) — suprime ecos fixos (Clutter).
    """
    def __init__(self, n_samples: int):
        self._mf_prev = np.zeros(n_samples)

    def process(self, comp_disp: np.ndarray) -> np.ndarray:
        """
        Aplica o filtro MTI subtraindo o pulso anterior do atual.
        
        Args:
            comp_disp: Sinal atual (saída do filtro casado).
            
        Returns:
            O sinal pós-MTI (valor absoluto da diferença).
        """
        # Subtrai o pulso MF anterior do atual — cancela ecos fixos (clutter)
        mti_output = np.abs(comp_disp - self._mf_prev)
        self._mf_prev = comp_disp.copy()
        return mti_output
