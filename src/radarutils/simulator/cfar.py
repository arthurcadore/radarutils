import numpy as np
from scipy.ndimage import uniform_filter1d

def ca_cfar(signal: np.ndarray, n_guard: int, n_train: int, alpha: float) -> np.ndarray:
    """
    CA-CFAR vectorizado via uniform_filter1d — threshold adaptativo por célula.
    
    Args:
        signal: Sinal de entrada (já pós-integração e MTI).
        n_guard: Número de células de guarda de cada lado.
        n_train: Número de células de treinamento de cada lado.
        alpha: Fator multiplicativo do threshold.
        
    Returns:
        Um array com os valores de threshold estimativos para cada amostra do sinal.
    """
    win_t = 2 * (n_guard + n_train) + 1
    win_g = 2 * n_guard + 1

    # Média móvel sobre a janela total e sobre a janela de guarda
    # Usa mode='reflect' (ou 'nearest') em vez de 'constant' (zero) para não
    # subestimar o piso de ruído nas bordas (o que causa falsos alarmes no início e fim do gráfico)
    sum_t = uniform_filter1d(signal, size=win_t, mode='reflect') * win_t
    sum_g = uniform_filter1d(signal, size=win_g, mode='reflect') * win_g
    n_tr  = max(win_t - win_g, 1)

    return np.maximum(alpha * (sum_t - sum_g) / n_tr, 0.0)
