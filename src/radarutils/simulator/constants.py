"""
constants.py — Constantes globais do simulador de radar.

Todas as constantes físicas, de forma de onda, processamento de sinal e
limites visuais são definidas aqui, em um único ponto de verdade.  Outros
módulos importam diretamente o que precisam, evitando literais espalhados.

Seções:
  1. Constantes Físicas e de Forma de Onda
  2. Equação do Radar (Friis / Monoestático)
  3. Qualidade do Sinal (SNR / Ruído)
  4. Processamento de Sinal (MTI, Integração, CFAR)
  5. Limites de Visualização (Eixos Y dos plots de sinal)
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# 1. Constantes Físicas e de Forma de Onda
# ══════════════════════════════════════════════════════════════════════════

C = 3e8
"""Velocidade da luz no vácuo (m/s)."""

F_C = 10e9
"""Frequência portadora (Hz) — banda X (10 GHz)."""

B = 30e6
"""Largura de banda do chirp LFM transmitido (Hz)."""

N_SAMPLES = 2000
"""
Número de amostras por PRI (Pulse Repetition Interval).

Define a resolução temporal de cada pulso simulado.  A taxa de amostragem
fs = N_SAMPLES / T_PRI é derivada automaticamente do alcance máximo do radar.
"""


# ══════════════════════════════════════════════════════════════════════════
# 2. Equação do Radar (Friis / Monoestático)
# ══════════════════════════════════════════════════════════════════════════

WAVELENGTH_M = 0.03
"""
Comprimento de onda λ (m) — banda X (10 GHz).

λ = c / F_C = 3e8 / 10e9 = 0.03 m
"""

RCS_DEFAULT_M2 = 1.0
"""
Seção transversal radar (RCS) padrão σ (m²).

Valor genérico para um alvo pontual de tamanho médio.  Pode ser
personalizado passando um valor de RCS diferente em simulações futuras.
"""

FOUR_PI_3 = (4 * np.pi) ** 3
"""
Fator geométrico da equação do radar monoestático: (4π)³.

Aparece no denominador da equação de Friis:
    P_rx = (P_tx · G² · λ² · σ) / ((4π)³ · R⁴)
"""


# ══════════════════════════════════════════════════════════════════════════
# 3. Qualidade do Sinal (SNR / Ruído)
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_SNR_DB = 20.0
"""
SNR (Signal-to-Noise Ratio) padrão do AWGN adicionado ao sinal RX (dB).

Controla a relação entre a amplitude do eco normalizado e o desvio padrão
do ruído gaussiano branco adicionado em PulseWidget.update_pulse().
"""


# ══════════════════════════════════════════════════════════════════════════
# 4. Processamento de Sinal
# ══════════════════════════════════════════════════════════════════════════

N_GUARD = 48
"""
Células de guarda de cada lado da célula sob teste (CUT) no CA-CFAR.

Evita que a energia do alvo contamine a estimativa do piso de ruído local.
Valor maior → menos auto-mascaramento do alvo, porém área cega maior.
"""

N_TRAIN = 256
"""
Células de treinamento de cada lado da janela de guarda no CA-CFAR.

Define a janela para estimativa do piso de ruído. Valor maior → estimativa
mais robusta, porém mais lenta para responder a variações de clutter.
"""

K_CFAR = 10
"""
Fator multiplicativo α do threshold CA-CFAR.

threshold_i = α · (média das células de treinamento)

Valor ajustado empiricamente para sinal pós-integrador (N_INT = 8 PRIs).
Valores maiores → menos falsos alarmes, porém menor Pd (probabilidade de detecção).
"""

N_INT = 8
"""
Número de PRIs a integrar (coerente ou não-coerente).

Ganho de integração não-coerente ≈ √N_INT ≈ +4.5 dB (para N_INT=8).
Ganho de integração coerente ≈ N_INT ≈ +18 dB (para N_INT=8, com coerência de fase).
"""

H_HITS = 4096
"""
Capacidade máxima do histórico de hits do PPI Estimado (número de pontos).

Points além desse limite são descartados automaticamente via ``deque(maxlen=H_HITS)``.
"""

MIN_CFAR_ABS = 10.0
"""
Threshold absoluto mínimo do CA-CFAR (unidades do integrador).

Garante que o threshold nunca seja inferior a este valor, suprimindo falsos
alarmes em regiões onde o sinal integrado é puro AWGN (sem alvo).
"""

MAX_MATCH_DIST = 90.0
"""
Distância máxima (m) para associar uma detecção CFAR a um alvo real.

Se a posição estimada da detecção estiver dentro dessa distância de qualquer
alvo real (ground truth), é classificada como Verdadeiro Positivo (TP).
Caso contrário, é registrada como Falso Alarme (FA).
"""


# ══════════════════════════════════════════════════════════════════════════
# 5. Limites de Visualização (eixos Y dos plots de sinal)
# ══════════════════════════════════════════════════════════════════════════

MIN_Y_MTI = 60.0
"""Amplitude mínima do eixo Y do plot MTI (modo não-normalizado)."""

MIN_Y_INT = 60.0
"""Amplitude mínima do eixo Y do plot Integrador (modo não-normalizado)."""

MIN_Y_CFAR = 150.0
"""Amplitude mínima do eixo Y do plot CA-CFAR (modo não-normalizado)."""
