"""
pulse.py — Geração e modulação de pulsos de radar.

Define as classes base e especializadas para os tipos de pulso utilizados
na simulação de radar.

Hierarquia de classes::

    Pulse (ABC)
    ├── AM_RadarPulse   — pulso amplitude-modulado (envelope retangular)
    └── FM_RadarPulse   — pulso FM (chirp linear)

O método ``modulate()`` retorna o sinal modulado no domínio do tempo.
"""

import numpy as np
from abc import ABC, abstractmethod


# ══════════════════════════════════════════════════════════════════════════
#  Classe Base Abstrata
# ══════════════════════════════════════════════════════════════════════════

class Pulse(ABC):
    """
    Classe base abstrata para pulsos de radar.

    Attributes:
        A (float):  Amplitude do pulso.
        t (float):  Duração total do sinal (s).
        fs (float): Taxa de amostragem (Hz).
    """

    @abstractmethod
    def __init__(self, A: float, t: float, fs: float = 128_000):
        self.A  = A
        self.t  = t
        self.fs = fs

    @abstractmethod
    def modulate(self) -> np.ndarray:
        """Retorna o sinal modulado no domínio do tempo."""


# ══════════════════════════════════════════════════════════════════════════
#  AM Pulse
# ══════════════════════════════════════════════════════════════════════════

class AM_RadarPulse(Pulse):
    """
    Pulso radar com modulação de amplitude (envelope retangular).

    O sinal é uma sequência binária: 1 durante a janela de pulso (Pl),
    0 fora dela. Adequado para representar pulsos simples não-codificados.

    Attributes:
        Prf (float):    Pulse Repetition Frequency (Hz).
        Pl (float):     Duração do pulso (s).
        AM_pulse (ndarray): Máscara binária do pulso.
    """

    def __init__(
        self,
        A:   float,
        t:   float,
        fs:  float = 128_000,
        Prf: float = 10,
        Pl:  float = 0.01,
    ):
        """
        Args:
            A:   Amplitude.
            t:   Duração total do sinal (s).
            fs:  Taxa de amostragem (Hz).
            Prf: Pulse Repetition Frequency (Hz).
            Pl:  Duração do pulso (s).
        """
        super().__init__(A, t, fs)
        self.Prf      = Prf
        self.Pl       = Pl
        self.AM_pulse = self._calc_pulse_windows()

    def _calc_pulse_windows(self) -> np.ndarray:
        """Cria vetor binário (1 onde o pulso está ON, 0 caso contrário)."""
        N    = int(self.t * self.fs)
        time = np.arange(N) / self.fs
        Tprf = 1.0 / self.Prf
        return ((time % Tprf) < self.Pl).astype(int)

    def modulate(self) -> np.ndarray:
        """Retorna a máscara AM como sinal modulado (float)."""
        return self.AM_pulse.astype(float) * self.A


# ══════════════════════════════════════════════════════════════════════════
#  FM Pulse (Chirp Linear)
# ══════════════════════════════════════════════════════════════════════════

class FM_RadarPulse(Pulse):
    """
    Pulso radar com modulação de frequência linear (chirp LFM).

    A frequência varia linearmente de ``f0`` a ``f1`` durante a janela de
    pulso. Fora da janela, o sinal é zero.  O método ``modulate()`` retorna
    o sinal AM+FM resultante.

    Attributes:
        f0 (float):     Frequência inicial do chirp (Hz).
        f1 (float):     Frequência final do chirp (Hz).
        Prf (float):    Pulse Repetition Frequency (Hz).
        Pl (float):     Duração do pulso (s).
        AM_pulse (ndarray): Máscara binária do pulso.
    """

    def __init__(
        self,
        A:   float,
        t:   float,
        fs:  float = 128_000,
        f0:  float = 500,
        f1:  float = 5_000,
        Prf: float = 10,
        Pl:  float = 0.01,
    ):
        """
        Args:
            A:   Amplitude.
            t:   Duração total do sinal (s).
            fs:  Taxa de amostragem (Hz).
            f0:  Frequência inicial do chirp (Hz).
            f1:  Frequência final do chirp (Hz).
            Prf: Pulse Repetition Frequency (Hz).
            Pl:  Duração do pulso (s).
        """
        super().__init__(A, t, fs)
        self.f0       = f0
        self.f1       = f1
        self.Prf      = Prf
        self.Pl       = Pl
        self.AM_pulse = self._calc_pulse_windows()

    def _calc_pulse_windows(self) -> np.ndarray:
        """Cria vetor binário (1 onde o pulso está ON, 0 caso contrário)."""
        N    = int(self.t * self.fs)
        time = np.arange(N) / self.fs
        Tprf = 1.0 / self.Prf
        return ((time % Tprf) < self.Pl).astype(int)

    def _calc_pulse_fm(self) -> np.ndarray:
        """
        Gera vetor de frequências instantâneas representando o chirp LFM.

        Fora da janela de pulso, a frequência é zero.

        Returns:
            freq (ndarray): Frequência instantânea (Hz) em cada amostra.
        """
        N    = int(self.t * self.fs)
        time = np.arange(N) / self.fs
        Tprf = 1.0 / self.Prf
        freq = np.zeros(N)

        for i in range(N):
            tau = time[i] % Tprf
            if tau < self.Pl:
                alpha   = tau / self.Pl                       # posição normalizada [0, 1]
                freq[i] = self.f0 + (self.f1 - self.f0) * alpha
        return freq

    def modulate(self) -> np.ndarray:
        """
        Retorna o sinal modulado AM + FM (chirp janelado).

        Returns:
            signal (ndarray): Sinal no domínio do tempo.
        """
        freq   = self._calc_pulse_fm()
        phase  = np.cumsum(2.0 * np.pi * freq / self.fs)   # integral discreta
        return self.A * np.cos(phase) * self.AM_pulse


# ══════════════════════════════════════════════════════════════════════════
#  Demonstração (execução direta)
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pulse = FM_RadarPulse(
        A=1,
        t=0.5,
        fs=128_000,
        Prf=3,
        Pl=0.05,
    )

    AM     = pulse.AM_pulse
    freq   = pulse._calc_pulse_fm()
    signal = pulse.modulate()
    N      = len(AM)
    ts     = np.arange(N) / pulse.fs

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), tight_layout=True)

    axes[0].plot(ts, AM, linewidth=1)
    axes[0].set_title("Máscara de Pulso (AM_pulse)")
    axes[0].set_xlabel("Tempo (s)")
    axes[0].set_ylabel("Valor (0 ou 1)")
    axes[0].grid(True)

    axes[1].plot(ts, freq)
    axes[1].set_title("Frequência Instantânea do Pulso FM")
    axes[1].set_xlabel("Tempo (s)")
    axes[1].set_ylabel("Frequência (Hz)")
    axes[1].grid(True)

    axes[2].plot(ts, signal)
    axes[2].set_title("Sinal Modulado FM")
    axes[2].set_xlabel("Tempo (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True)

    plt.show()
