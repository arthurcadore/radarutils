import numpy as np

# Constantes Físicas e do Radar
C = 3e8                # velocidade da luz (m/s)
F_C = 10e9             # portadora (Hz) — banda X
B = 30e6               # largura de banda do chirp (Hz)
N_SAMPLES = 2000       # amostras por PRI

# Constantes da Equação do Radar
WAVELENGTH_M = 0.03    # λ default: 10 GHz (banda X) em metros
RCS_DEFAULT_M2 = 1.0   # σ default: 1 m² (alvo genérico)
FOUR_PI_3 = (4 * np.pi) ** 3

# SNR e Pulso
DEFAULT_SNR_DB = 20.0

# Processamento de Sinal
N_GUARD = 48           # células de guarda para CFAR
N_TRAIN = 256          # células de treinamento para CFAR
K_CFAR = 10            # limiar multiplicativo ajustado para sinal pós-integrador (N_INT=8)
N_INT = 8              # número de PRIs para integração não-coerente
H_HITS = 4096          # histórico máximo de detecções no PPI estimado
MIN_CFAR_ABS = 1000.0  # threshold absoluto mínimo (suprime falsos alarmes em AWGN)
MAX_MATCH_DIST = 90.0  # Distância máxima (m) para match de alvo real no CFAR

# Constantes Visuais (Eixos Y)
MIN_Y_MTI = 60.0
MIN_Y_INT = 60.0
MIN_Y_CFAR = 150.0
