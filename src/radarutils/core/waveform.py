r"""
waveform.py — Geração e processamento de formas de onda radar (core teórico).

Este módulo contém implementações matemáticas puras para geração de pulsos
LFM (Linear Frequency Modulation / chirp) e processamento associado: filtro
casado (pulse compression), cálculo de desvio Doppler e adição de AWGN
calibrado por SNR.

Não possui dependências de interface gráfica (UI) e pode ser utilizado tanto
pelo simulador quanto em análises e scripts independentes.

Conteúdo:
    - generate_lfm_chirp  : Gera o pulso chirp LFM transmitido.
    - build_rx_echo       : Constrói o eco de um alvo com atraso e Doppler.
    - apply_awgn          : Adiciona ruído AWGN calibrado por SNR.
    - matched_filter      : Aplica o filtro casado (correlação cruzada) ao sinal.
    - doppler_frequency   : Calcula o desvio de frequência Doppler.
"""

import numpy as np
import scipy.signal


# ══════════════════════════════════════════════════════════════════════════════
# Forma de onda LFM
# ══════════════════════════════════════════════════════════════════════════════

def generate_lfm_chirp(
    n_samples: int,
    n_pulse: int,
    chirp_rate: float,
    t: np.ndarray,
) -> np.ndarray:
    r"""
    Gera o pulso chirp LFM (Linear Frequency Modulation) transmitido.

    O chirp LFM é definido pela fase instantânea quadrática:

    $$
    s_{\text{TX}}(t) =
    \begin{cases}
        \cos\!\left(\pi k t^2\right), & 0 \le t < T_P \\
        0, & \text{caso contrário}
    \end{cases}
    $$

    onde :math:`k = B / T_P` é a taxa de varredura (``chirp_rate``) em Hz/s,
    :math:`B` é a largura de banda e :math:`T_P` é a duração do pulso.

    A resolução em distância após compressão de pulso é:

    $$
    \delta_r = \frac{c}{2B}
    $$

    e o ganho de processamento (Time-Bandwidth Product) é:

    $$
    G_{\text{TBP}} = B \cdot T_P
    $$

    Args:
        n_samples (int): Número total de amostras por PRI.
        n_pulse (int): Número de amostras do pulso transmitido (:math:`T_P \cdot f_s`).
        chirp_rate (float): Taxa de varredura do chirp :math:`k = B / T_P` (Hz/s).
        t (np.ndarray): Vetor de tempo do PRI completo (s), shape ``(n_samples,)``.

    Returns:
        np.ndarray: Sinal TX real, shape ``(n_samples,)``. Zeros fora do pulso.

    References:
        Nathanson, F.E. — "Radar Design Principles", 2nd Ed. (Cap. 10).
    """
    tx = np.zeros(n_samples)
    t_chirp         = t[:n_pulse]
    tx[:n_pulse]    = np.cos(np.pi * chirp_rate * t_chirp ** 2)
    return tx


def doppler_frequency(v_radial: float, f_carrier: float, c: float) -> float:
    r"""
    Calcula o desvio de frequência Doppler de um alvo com velocidade radial.

    $$
    f_d = \frac{2 v_r f_c}{c}
    $$

    onde :math:`v_r > 0` indica alvo se aproximando (desvio positivo) e
    :math:`v_r < 0` indica alvo se afastando (desvio negativo).

    Args:
        v_radial (float): Velocidade radial do alvo em m/s (positivo = aproximação).
        f_carrier (float): Frequência portadora em Hz.
        c (float): Velocidade da luz no meio em m/s.

    Returns:
        float: Desvio Doppler :math:`f_d` em Hz.

    References:
        Merill I. Skolnik — Introduction To Radar Systems, 3rd Ed. (Cap. 3).
    """
    return 2.0 * v_radial * f_carrier / c


def build_rx_echo(
    t: np.ndarray,
    n_samples: int,
    n_pulse: int,
    chirp_rate: float,
    amplitude: float,
    delay_samples: int,
    carrier_phase: float,
    doppler_hz: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Constrói o eco de um alvo pontual no sinal banda base recebido.

    Para um alvo a distância :math:`R` com velocidade radial :math:`v_r`, o eco
    é uma réplica atrasada e modulada em Doppler do chirp transmitido:

    $$
    s_{\text{RX}}(t) = a \cdot \cos\!\bigl(\pi k (t-\tau)^2
                       + 2\pi f_d (t-\tau) + \phi\bigr)
    $$

    onde :math:`\tau = 2R/c` é o atraso de ida e volta, :math:`a` é a amplitude
    relativa (proporcional à raiz quadrada da potência recebida),
    :math:`f_d` é o desvio Doppler e :math:`\phi = 2\pi f_c \tau \mod 2\pi`
    é a fase da portadora no instante do atraso.

    A versão complexa (IQ) preserva a informação de fase para integração coerente:

    $$
    \tilde{s}_{\text{RX}}(t) = a \cdot e^{j(\pi k (t-\tau)^2 + 2\pi f_d (t-\tau) + \phi)}
    $$

    Args:
        t (np.ndarray): Vetor de tempo do PRI completo (s).
        n_samples (int): Número total de amostras por PRI.
        n_pulse (int): Número de amostras do pulso TX.
        chirp_rate (float): Taxa de varredura :math:`k = B/T_P` (Hz/s).
        amplitude (float): Amplitude relativa do eco (escala de tensão).
        delay_samples (int): Atraso :math:`\tau` em amostras (:math:`\lfloor \tau f_s \rfloor`).
        carrier_phase (float): Fase da portadora :math:`\phi` (radianos).
        doppler_hz (float): Desvio Doppler :math:`f_d` (Hz).
        tau (float): Atraso de ida e volta :math:`\tau = 2R/c` (s).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - ``rx_real``    : Contribuição real do eco, shape ``(n_samples,)``.
            - ``rx_complex`` : Contribuição IQ complexa do eco, shape ``(n_samples,)``.

    References:
        Richards, M.A. — "Fundamentals of Radar Signal Processing", 2nd Ed. (Cap. 4).
    """
    rx_real    = np.zeros(n_samples)
    rx_complex = np.zeros(n_samples, dtype=complex)

    end      = min(delay_samples + n_pulse, n_samples)
    n_actual = end - delay_samples
    if n_actual <= 0:
        return rx_real, rx_complex

    t_local     = t[delay_samples:end] - tau
    chirp_phase = (
        np.pi * chirp_rate * t_local ** 2
        + 2.0 * np.pi * doppler_hz * t_local
        + carrier_phase
    )

    rx_real[delay_samples:end]    += amplitude * np.cos(chirp_phase)
    rx_complex[delay_samples:end] += amplitude * np.exp(1j * chirp_phase)

    return rx_real, rx_complex


def apply_awgn(
    signal: np.ndarray,
    snr_db: float,
    peak_amplitude: float = None,
) -> np.ndarray:
    r"""
    Adiciona ruído AWGN gaussiano calibrado pela SNR desejada.

    O desvio padrão do ruído é calculado a partir da amplitude de pico do
    sinal normalizado, de modo que:

    $$
    \text{SNR} = 20 \log_{10}\!\left(\frac{A_{\text{pico}}}{\sigma_n}\right) = \text{SNR}_{\text{dB}}
    $$

    portanto:

    $$
    \sigma_n = \frac{A_{\text{pico}}}{10^{\text{SNR}_{\text{dB}} / 20}}
    $$

    Args:
        signal (np.ndarray): Sinal de entrada (real, normalizado).
        snr_db (float): Relação sinal-ruído desejada em dB.
        peak_amplitude (float | None): Amplitude de referência para calibração.
            Se ``None``, usa o máximo absoluto de ``signal``.

    Returns:
        np.ndarray: Sinal com ruído AWGN adicionado.

    References:
        Merill I. Skolnik — Introduction To Radar Systems, 3rd Ed. (Cap. 2).
    """
    ref   = float(np.max(np.abs(signal))) if peak_amplitude is None else peak_amplitude
    ref   = ref if ref > 1e-30 else 0.88
    sigma = ref / (10.0 ** (snr_db / 20.0))
    return signal + sigma * np.random.randn(len(signal))


def matched_filter(
    rx_noisy: np.ndarray,
    rx_complex: np.ndarray,
    tx_pulse: np.ndarray,
    n_pulse: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Aplica o Filtro Casado (Matched Filter / Pulse Compression) ao sinal recebido.

    O filtro casado maximiza a SNR na saída para um sinal conhecido em ruído
    branco gaussiano. Para um chirp LFM, a saída é a correlação cruzada entre
    o sinal recebido e o pulso de referência:

    $$
    y_{\text{MF}}(t) = \bigl|s_{\text{RX}}(t) \star s_{\text{TX}}^*(-t)\bigr|
    $$

    O ganho de processamento é igual ao produto tempo-largura de banda
    :math:`G = B \cdot T_P`, resultando em compressão do pulso e melhora da
    resolução em distância de :math:`\delta_r = c / (2B)`.

    Após correlação, o vetor é deslocado ``n_pulse // 2`` amostras para alinhar
    o pico com o instante de eco verdadeiro, e as primeiras amostras (zona cega)
    são zeradas.

    Args:
        rx_noisy (np.ndarray): Sinal real recebido com AWGN.
        rx_complex (np.ndarray): Sinal IQ complexo recebido (sem AWGN).
        tx_pulse (np.ndarray): Pulso TX de referência (primeiras ``n_pulse`` amostras).
        n_pulse (int): Número de amostras do pulso TX (define o deslocamento).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - ``comp_disp``    : Envoltória real da saída MF (para display e MTI).
            - ``comp_complex`` : Saída complexa do MF (para integração coerente).

    References:
        Turin, G.L. — "An Introduction to Matched Filters",
        IRE Trans. Inf. Theory, 1960.
    """
    # Saída real: |correlação(rx_noisy, tx_pulse)| — para MTI e display
    compressed   = np.abs(scipy.signal.correlate(rx_noisy, tx_pulse, mode='same'))
    comp_disp    = np.roll(compressed, n_pulse // 2)
    comp_disp[:n_pulse // 2] = 0.0

    # Saída complexa: correlação(rx_IQ, tx_pulse*) — para integração coerente
    compressed_cplx = scipy.signal.correlate(
        rx_complex, tx_pulse.astype(complex), mode='same'
    )
    comp_complex    = np.roll(compressed_cplx, n_pulse // 2)
    comp_complex[:n_pulse // 2] = 0.0

    return comp_disp, comp_complex
