from scipy.signal import butter, cheby1, cheby2, ellip, filtfilt

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