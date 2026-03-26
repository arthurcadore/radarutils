# TODO: implementar funções para sinais

import numpy as np
from scipy.signal import butter, cheby1, cheby2, ellip, filtfilt
from ..visualization.plotter import create_figure, save_figure, TimePlot, FrequencyPlot, FrequencyResponsePlot

class HighPassFilter:
    def __init__(self, order, cutoff_freq, fs, filter_type="butter", rp=1, rs=40):
        """
        filter_type : str  -> 'butter', 'cheby1', 'cheby2', 'ellip'
        order       : int  -> ordem do filtro
        cutoff_freq : float -> frequência de corte (Hz)
        fs          : float -> frequência de amostragem (Hz)
        rp, rs      : parâmetros adicionais para Chebyshev e Elíptico
        """
        self.filter_type = filter_type.lower()
        self.order = order
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.rp = rp  # ripple passband (usado em cheby1 e ellip)
        self.rs = rs  # ripple stopband (usado em cheby2 e ellip)

        self.b, self.a = self._design_filter()

    def _design_filter(self):
        """Gera a função de transferência (b, a) para o tipo especificado."""
        wc = self.cutoff_freq / (self.fs / 2)  # normaliza para Nyquist

        if self.filter_type == "butter":
            return butter(self.order, wc, btype="highpass")

        elif self.filter_type == "cheby1":
            return cheby1(self.order, self.rp, wc, btype="highpass")

        elif self.filter_type == "cheby2":
            return cheby2(self.order, self.rs, wc, btype="highpass")

        elif self.filter_type == "ellip":
            return ellip(self.order, self.rp, self.rs, wc, btype="highpass")

        else:
            raise ValueError(
                "filter_type inválido. Use: 'butter', 'cheby1', 'cheby2', 'ellip'"
            )

    def apply(self, x):
        """Aplica o filtro a um vetor x e retorna o resultado."""
        return filtfilt(self.b, self.a, x)

class coherent_radar():
    def __init__(self, As, Ac, t, fs=100, fc=1000, es=0, ec=0):

        self.As = As
        self.Ac = Ac
        self.fs = fs
        self.fc = fc
        self.es = es
        self.ec = ec
        self.t = t

        self.Os = self.stalo()
        self.Oc = self.coho()

        self.Om = self.Os * self.Oc

    def stalo(self):

        Os = self.As * np.cos(2 * np.pi * self.fs * self.t + self.es)
        return Os

    def coho(self):

        Oc = self.Ac * np.cos(2 * np.pi * self.fc * self.t + self.ec)
        return Oc

# STALO



# COHO 

# RECT

def rect(t, tp):
    r"""
    if |t| < tp, return 1
    if |t| = tp, return 0.5
    if |t| > tp, return 0
    """

    if np.abs(t) < tp:
        return 1
    elif np.abs(t) == tp:
        return 0.5
    else:
        return 0


# MOD / DEMOD IQ


if __name__ == "__main__":
    
    t = np.linspace(0, 0.1, 10000)
    a1 = 1
    a2 = 2

    y = coherent_radar(a1, a2, t)

    print(y.Om) 

    fig, grid = create_figure(3,1, figsize=(16, 9))
    TimePlot(
        fig, grid, (0,0), 
        t,
        signals =[y.Os],
        labels=[r"Os"]

    ).plot()
    TimePlot(
        fig, grid, (1,0), 
        t,
        signals =[y.Oc],
        labels=[r"Oc"]

    ).plot()
    TimePlot(
        fig, grid, (2,0), 
        t,
        signals =[y.Om],
        labels=[r"Om"]

    ).plot()

    fig.tight_layout()
    save_figure(fig, "transmitter_formatter_time.pdf")


    fig, grid = create_figure(3,1, figsize=(16, 9))
    FrequencyPlot(
        fig, grid, (0,0), 
        fs=10000,
        signal=y.Os,
        fc=100,
        labels=[r"Os"]
    ).plot()
    FrequencyPlot(
        fig, grid, (1,0), 
        fs=10000,
        signal=y.Oc,
        fc=100,
        labels=[r"Oc"],
        xlim=[-500,500]
    ).plot()
    FrequencyPlot(
        fig, grid, (2,0), 
        fs=10000,
        signal=y.Om,
        fc=100,
        labels=[r"Om"],
        xlim=[-500,500]
    ).plot()
    fig.tight_layout()
    save_figure(fig, "transmitter_formatter_freq.pdf")

    HPA = HighPassFilter(4, 200, 10000, filter_type="ellip")
    y_hp = HPA.apply(y.Om)
    print(y_hp)

    freq_response, grid_freq_response = create_figure(1, 1, figsize=(16,6))
    FrequencyResponsePlot(
            freq_response, grid_freq_response, (0,0), 
            HPA.b, HPA.a, 
            fs=HPA.fs, 
            f_cut=HPA.cutoff_freq, 
            xlim=(0, 3*HPA.cutoff_freq),
            title="HPA Frequency Response"    
        ).plot()
    save_figure(freq_response, "transmitter_formatter_freq_hpa_response.pdf")

    fig, grid = create_figure(2,1, figsize=(16, 9))
    FrequencyPlot(
        fig, grid, (0,0), 
        fs=10000,
        signal=y.Om,
        fc=100,
        labels=[r"Om"],
        xlim=[-500,500]
    ).plot()
    FrequencyPlot(
        fig, grid, (1,0), 
        fs=10000,
        signal=y_hp,
        fc=100,
        labels=[r"Om (HPA)"],
        xlim=[-500,500]
    ).plot()
    fig.tight_layout()
    save_figure(fig, "transmitter_formatter_freq_hpa.pdf")
