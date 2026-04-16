

from abc import ABC, abstractmethod

class pulse(ABC):
    @abstractmethod
    def __init__(self, A, t, fs=128000):
        self.A = A
        self.t = t
        self.fs = fs
    @abstractmethod
    def modulate(self):
        pass

class AM_radar_pulse(pulse):
    def __init__(self, A, t, fs=128000, Prf=10, Pl=0.01):
        super().__init__(A, t, fs)
        self.Prf = Prf
        self.Pl = Pl
        self.AM_pulse = self.calc_pulse_windows()

    def calc_pulse_windows(self): 
        """Create a binary vector (1 where the pulse is ON, 0 otherwise)"""

        # timeline
        N = int(self.t * self.fs)
        time = np.arange(N) / self.fs

        # PRF pulse repetition period
        Tprf = 1 / self.Prf

        # pulse mask: 1 during Pl, else 0
        pulse = ((time % Tprf) < self.Pl).astype(int)
        
        return pulse


class FM_radar_pulse(pulse): 
    r"""
    This class generate simple radar pulses that change the information of frequency during the pulse period. 
    """
    def __init__(self, A, t, fs=128000, f0=500, f1=5000, Prf=10, Pl=0.01):
        super().__init__(A, t, fs)
        self.f0 = f0
        self.f1 = f1
        self.Prf = Prf
        self.Pl = Pl
        self.AM_pulse = self.calc_pulse_windows()

    def calc_pulse_windows(self): 
        """Create a binary vector (1 where the pulse is ON, 0 otherwise)"""

        # timeline
        N = int(self.t * self.fs)
        time = np.arange(N) / self.fs

        # PRF pulse repetition period
        Tprf = 1 / self.Prf

        # pulse mask: 1 during Pl, else 0
        pulse = ((time % Tprf) < self.Pl).astype(int)
        
        return pulse
    
    def calc_pulse_FM(self):
        """
        Generate a vector of instantaneous frequencies representing
        a linear FM chirp only during the pulse window.
        Outside the pulse, frequency = 0.
        """

        # timeline
        N = int(self.t * self.fs)
        time = np.arange(N) / self.fs

        # repetition period
        Tprf = 1 / self.Prf

        # vetor de saída
        freq = np.zeros(N)

        # percorre cada amostra
        for i in range(N):
            # instante dentro do período do PRF
            tau = time[i] % Tprf

            # se estamos dentro da duração do pulso
            if tau < self.Pl:
                # posição normalizada dentro do pulso (0 → 1)
                alpha = tau / self.Pl
                # barrido linear f0 → f1
                freq[i] = self.f0 + (self.f1 - self.f0) * alpha
            else:
                freq[i] = 0.0

        return freq

    def modulate(self):
        freq = self.calc_pulse_FM()

        # fase = integral discreta de 2*pi*f/fs
        phase = np.cumsum(2 * np.pi * freq / self.fs)

        # modulação AM (pulso) + FM (fase)
        signal = self.A * np.cos(phase) * self.AM_pulse

        return signal



import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    pulse = FM_radar_pulse(
        A=1,
        t=0.5,         # duração total
        fs=128000,      # taxa de amostragem
        Prf=3,
        Pl=0.05
    )

    AM = pulse.AM_pulse
    freq = pulse.calc_pulse_FM()
    signal = pulse.modulate()   
    N = len(AM)
    ts = np.arange(N) / pulse.fs

    plt.figure(figsize=(10, 3))
    plt.plot(ts, AM, linewidth=1)
    plt.title("Máscara de Pulso (AM_pulse)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Valor (0 ou 1)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(ts, freq)
    plt.title("Frequência Instantânea do Pulso FM")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Frequência (Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(ts, signal)
    plt.title("Sinal Modulado FM")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

