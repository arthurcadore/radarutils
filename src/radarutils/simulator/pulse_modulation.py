r"""
pulse_modulation.py — Widget de visualização do pulso radar LFM (coluna do meio).

Contém a classe ``PulseWidget``, um QSplitter vertical com:
  - HeaderPanel : tabela HTML com parâmetros globais da simulação.
  - TxPlot      : chirp LFM transmitido, normalizado a ±1 (ciano).
  - RxPlot      : sinal banda base recebido + AWGN (laranja).
  - MFPlot      : saída do Filtro Casado / Pulse Compression (magenta).

O método público ``update_pulse(detections)`` reconstrói o sinal RX a
partir das detecções do PPI, aplica ruído AWGN e calcula o filtro casado.
Retorna um dict com os sinais processados para uso no ``ProcessingWidget``.

As operações matemáticas (geração de chirp LFM, eco de alvo com Doppler,
AWGN calibrado por SNR e filtro casado) residem em:
    radarutils.core.waveform
"""

import numpy as np
import pyqtgraph as pg

from PySide6 import QtCore, QtWidgets

from radarutils.simulator.ppi import PPI
from radarutils.simulator.detection import DetectionRecord
from radarutils.simulator.constants import (
    C, F_C, B, N_SAMPLES, DEFAULT_SNR_DB,
)
from radarutils.simulator.html_contents import get_pulse_header_html
from radarutils.core.clutter import Clutter, clutter_from_str
from radarutils.core.waveform import (
    generate_lfm_chirp,
    build_rx_echo,
    apply_awgn,
    matched_filter,
    doppler_frequency,
)


class PulseWidget(QtWidgets.QSplitter):
    r"""
    Painel de visualização do pulso radar em banda base (chirp LFM).

    Parâmetros de forma de onda derivados automaticamente de ``r_max``::

        T_PRI = 2·r_max / c    (período de repetição de pulso)
        T_P   = T_PRI / 7      (duração do pulso transmitido)
        fs    = N_SAMPLES/T_PRI (taxa de amostragem)
        k     = B / T_P         (taxa de varredura do chirp)

    Uso::

        pw = PulseWidget(ppi=ppi, snr_db=20, coherent_integration=True)
        pulse_data = pw.update_pulse(detections)
        # pulse_data → {'comp_disp', 'comp_complex', 'azimuth_deg'}
    """

    def __init__(
        self,
        ppi: PPI = None,
        snr_db: float = None,
        integrator_type: str = "noncoherent",
        clutter_type: str = "None",
        normalize_plots: bool = True,
    ):
        r"""
        Args:
            ppi (PPI): Instância de PPI com radar e targets configurados.
            snr_db (float): SNR da adição de AWGN em dB. Padrão: DEFAULT_SNR_DB.
            coherent_integration (bool): Se True, passa sinal IQ complexo normalizado no retorno.
            clutter_type (str): Tipo de clutter a injetar ('None', 'Rayleigh', 'Rice' ou 'Weibull').
            normalize_plots (bool): Se True, normaliza eixo Y do MF para [0, 1].
        """
        super().__init__(QtCore.Qt.Vertical)

        self.ppi                  = ppi
        self.snr_db               = snr_db if snr_db is not None else DEFAULT_SNR_DB
        self.integrator_type      = integrator_type
        self.normalize_plots      = normalize_plots

        # Instancia o modelo de clutter (ou None se desativado)
        self._clutter: Clutter | None = None
        self._clutter_name: str = "None"
        self._set_clutter(clutter_type)

        self.setStyleSheet("QSplitter::handle { background-color: #555555; height: 3px; }")

        # ── Parâmetros da forma de onda ───────────────────────────────────
        r_max      = ppi.r_max if ppi else 1000.0
        self.T_PRI = 2.0 * r_max / C          # Período de repetição de pulso (s)
        self.T_P   = self.T_PRI / 7.0         # Duração do pulso TX (s)
        self.fs    = N_SAMPLES / self.T_PRI   # Taxa de amostragem (Hz)
        self.t     = np.linspace(0, self.T_PRI, N_SAMPLES, endpoint=False)  # eixo tempo (s)
        self.t_us  = self.t * 1e6             # eixo tempo em microsegundos (para plot)
        self.k     = B / self.T_P             # Taxa de varredura do chirp LFM (Hz/s)
        self.n_p   = int(self.T_P * self.fs)  # Número de amostras do pulso TX
        self.N_rx  = N_SAMPLES - self.n_p     # Amostras do período de escuta (após pulso)

        # Potência TX em dBm (usada para calcular amplitude relativa do eco)
        if ppi and ppi.radar:
            self.P_tx_dbm = 10.0 * np.log10(ppi.radar.pt * 1e3)
        else:
            self.P_tx_dbm = 60.0

        # Sinal TX: chirp LFM gerado por core.waveform
        self._tx = generate_lfm_chirp(N_SAMPLES, self.n_p, self.k, self.t)

        # Marcador de início do período de escuta (µs) para linhas verticais nos plots
        rx_start_us = self.T_P * 1e6

        # ── Construção dos widgets filhos ─────────────────────────────────
        self._build_header()
        self._build_plots(rx_start_us)

        # Proporções das faixas verticais dentro do GraphicsLayoutWidget
        self._glw.ci.layout.setRowStretchFactor(0, 1)   # TX
        self._glw.ci.layout.setRowStretchFactor(1, 1)   # RX
        self._glw.ci.layout.setRowStretchFactor(2, 1.5) # MF (um pouco maior)

        self.setSizes([200, 800])

    # ──────────────────────────────────────────────────────────────────────
    #  Construção dos sub-widgets
    # ──────────────────────────────────────────────────────────────────────

    def _set_clutter(self, clutter_type: str) -> None:
        """
        Instancia o modelo de clutter a partir do nome e armazena o nome
        formatado para exibição no cabeçalho.

        Args:
            clutter_type: Nome do modelo ('None', 'Rayleigh', 'Rice', 'Weibull').
        """
        try:
            self._clutter = clutter_from_str(clutter_type, N_SAMPLES, amplitude=1e-6)
        except ValueError:
            self._clutter = None

        if self._clutter is None:
            self._clutter_name = "None"
        else:
            self._clutter_name = type(self._clutter).__name__.replace("Clutter", "")

    def set_clutter(self, clutter_type: str) -> None:
        """
        Altera o modelo de clutter em tempo de execução.

        Args:
            clutter_type: Nome do modelo ('None', 'Rayleigh', 'Rice', 'Weibull').
        """
        self._set_clutter(clutter_type)

    def _build_header(self) -> None:
        r"""Cria o QLabel de cabeçalho com parâmetros da simulação."""
        self._header_label = QtWidgets.QLabel()
        self._header_label.setStyleSheet("background-color: black;")
        self._header_label.setAlignment(QtCore.Qt.AlignCenter)
        self.addWidget(self._header_label)

    def _build_plots(self, rx_start_us: float) -> None:
        r"""
        Cria o GraphicsLayoutWidget com os três sub-plots (TX, RX, MF).

        Args:
            rx_start_us (float): Posição (µs) da linha vertical que marca início do RX.
        """
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground('k')
        self._glw.ci.layout.setSpacing(12)
        self.addWidget(self._glw)

        # ── Plot TX ──────────────────────────────────────────────────────
        self._tx_plot = self._glw.addPlot(row=0, col=0)
        self._tx_plot.setLabel('left', 'TX  Pulse')
        self._tx_plot.getAxis('left').setWidth(65)
        self._tx_plot.showGrid(x=True, y=True, alpha=0.22)
        self._tx_plot.setYRange(-1.2, 1.2)
        self._tx_plot.setMouseEnabled(x=False, y=False)
        self._tx_curve = self._tx_plot.plot(
            self.t_us, self._tx, pen=pg.mkPen((0, 200, 255), width=1),
        )
        self._tx_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen((0, 80, 100), width=1, style=QtCore.Qt.DotLine),
        ))
        self._tx_plot.addItem(pg.InfiniteLine(
            pos=rx_start_us, angle=90,
            pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine),
        ))

        # ── Plot RX ──────────────────────────────────────────────────────
        self._rx_plot = self._glw.addPlot(row=1, col=0)
        self._rx_plot.setLabel('left', 'RX Baseband')
        self._rx_plot.getAxis('left').setWidth(65)
        self._rx_plot.showGrid(x=True, y=True, alpha=0.22)
        self._rx_plot.setYRange(-1.2, 1.2)
        self._rx_plot.setMouseEnabled(x=False, y=False)
        self._rx_plot.setXLink(self._tx_plot)
        self._rx_curve = self._rx_plot.plot(
            self.t_us, np.zeros(N_SAMPLES),
            pen=pg.mkPen((255, 140, 0), width=1),
        )
        self._rx_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen((80, 50, 0), width=1, style=QtCore.Qt.DotLine),
        ))
        self._rx_plot.addItem(pg.InfiniteLine(
            pos=rx_start_us, angle=90,
            pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine),
        ))

        # ── Plot Matched Filter ──────────────────────────────────────────
        self._mf_plot = self._glw.addPlot(row=2, col=0)
        self._mf_plot.setLabel('left', 'Matched Filter Out')
        self._mf_plot.getAxis('left').setWidth(65)
        self._mf_plot.setLabel('bottom', 'Tempo (µs)')
        self._mf_plot.showGrid(x=True, y=True, alpha=0.22)
        self._mf_plot.setYRange(0, 100)
        self._mf_plot.setMouseEnabled(x=False, y=False)
        self._mf_plot.setXLink(self._tx_plot)
        self._mf_curve = self._mf_plot.plot(
            self.t_us, np.zeros(N_SAMPLES),
            pen=pg.mkPen((255, 0, 255), width=1),
        )
        self._mf_plot.addItem(pg.InfiniteLine(
            pos=rx_start_us, angle=90,
            pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine),
        ))

    # ──────────────────────────────────────────────────────────────────────
    #  Atualização por frame
    # ──────────────────────────────────────────────────────────────────────

    def update_pulse(self, detections: list[DetectionRecord], active_regional: list = None) -> dict:
        r"""
        Reconstrói o sinal RX banda base a partir das detecções do PPI,
        adiciona ruído AWGN, calcula o Filtro Casado e atualiza os plots.

        Pipeline:
          1. Para cada detecção: calcula atraso τ = 2R/c, amplitude relativa
             e frequência Doppler f_d = 2·v_r·F_C/c (via ``core.waveform``).
          2. Soma contribuições de eco ao sinal real rx[] e complexo rx_complex[].
          3. Injeta clutter regional (``active_regional``) nas amostras de range
             correspondentes ao atraso de ida e volta da região iluminada.
          4. Injeta clutter Rayleigh/Rice/Weibull ambiental (opcional).
          5. Normaliza sinal (ganho fixo independente de SNR).
          6. Adiciona AWGN gaussiano calibrado pelo snr_db (via ``core.waveform``).
          7. Calcula correlação cruzada (matched filter) real e complexa (via ``core.waveform``).
          8. Atualiza plots e cabeçalho HTML.

        Args:
            detections (list[DetectionRecord]): Lista de DetectionRecord do passo atual.
            active_regional (list | None): Lista de (RegionalClutter, r_near, r_far) retornada
                pelo PPI.update() para regiões de clutter iluminadas pelo feixe.

        Returns:
            pulse_data (dict): Dicionário contendo os seguintes dados com chaves:
              'comp_disp'    : np.ndarray — saída real do MF (para plot e MTI).
              'comp_complex' : np.ndarray — saída complexa do MF (para integração coerente).
              'azimuth_deg'  : float      — azimute atual do radar (graus).
        """
        # Sinais de saída: real (para plot) e complexo IQ (para integração coerente)
        rx         = np.zeros(N_SAMPLES)
        rx_complex = np.zeros(N_SAMPLES, dtype=complex)

        # ── 1. Construção do sinal RX a partir das detecções ─────────────
        if detections and self.ppi and self.ppi.radar:
            self.P_tx_dbm = 10.0 * np.log10(self.ppi.radar.pt * 1e3)

            for rec in detections:
                # Atraso de ida e volta (s) e índice de amostra correspondente
                tau   = 2.0 * rec.range_m / C
                n_del = int(tau * self.fs)
                if n_del >= N_SAMPLES:
                    continue  # eco chegaria após o fim do PRI — descarta

                # Amplitude relativa: razão entre P_rx e P_tx (escala linear de tensão)
                a = 10.0 ** ((rec.rx_power_dbm - self.P_tx_dbm) / 20.0)

                # Fase da portadora no instante do retardo
                phi = (2.0 * np.pi * F_C * tau) % (2.0 * np.pi)

                # Velocidade radial do alvo (projeção da velocidade no eixo radar-alvo)
                tgt = self.ppi.targets[rec.target_idx]
                if rec.range_m > 0 and tgt.velocity > 0:
                    vx  = tgt.velocity * np.cos(tgt.theta)
                    vy  = tgt.velocity * np.sin(tgt.theta)
                    v_r = (vx * tgt.x + vy * tgt.y) / rec.range_m
                else:
                    v_r = 0.0

                # Desvio Doppler (Hz) via core.waveform
                f_d = doppler_frequency(v_r, F_C, C)

                # Eco do alvo (chirp LFM atrasado + Doppler) via core.waveform
                echo_real, echo_cplx = build_rx_echo(
                    t=self.t,
                    n_samples=N_SAMPLES,
                    n_pulse=self.n_p,
                    chirp_rate=self.k,
                    amplitude=a,
                    delay_samples=n_del,
                    carrier_phase=phi,
                    doppler_hz=f_d,
                    tau=tau,
                )
                rx         += echo_real
                rx_complex += echo_cplx

        # ── 2. Clutter regional (regiões circulares no PPI) ──────────────
        # Cada entrada de active_regional é (RegionalClutter, r_near, r_far).
        # Convertemos o intervalo de range para índices de amostra e geramos
        # o sinal de clutter nesse segmento do vetor rx.
        if active_regional:
            for rc, r_near, r_far in active_regional:
                # Índices de amostra correspondentes ao intervalo de range [r_near, r_far]
                n_near = int(2.0 * r_near / C * self.fs)
                n_far  = int(2.0 * r_far  / C * self.fs)
                n_near = max(0, min(n_near, N_SAMPLES - 1))
                n_far  = max(0, min(n_far,  N_SAMPLES - 1))

                n_span = n_far - n_near
                if n_span <= 0:
                    continue  # intervalo degenerado — descarta

                # Gera amostras IQ complexas para o segmento iluminado
                clutter_samples = rc.generate_samples(n_span)

                rx[n_near:n_far]         += np.real(clutter_samples)
                rx_complex[n_near:n_far] += clutter_samples

        # ── 3. Clutter ambiental (Rayleigh / Rice / Weibull) ─────────────
        if self._clutter is not None:
            c_noise     = self._clutter.generate()
            rx_complex += c_noise
            rx         += np.real(c_noise)

        # ── 3. Atualização do cabeçalho HTML ─────────────────────────────
        self._update_header()

        # ── 4. Normalização ───────────────────────────────────────────────
        # Normaliza para que o pico do sinal real fique em ±0.88, independente
        # da potência absoluta. Isso permite um SNR controlável via snr_db.
        peak = float(np.max(np.abs(rx)))
        rx_norm = (rx / peak * 0.88) if peak > 1e-30 else rx.copy()

        peak_cplx       = float(np.max(np.abs(rx_complex)))
        rx_complex_norm = (rx_complex / peak_cplx * 0.88) if peak_cplx > 1e-30 else rx_complex.copy()

        # ── 5. AWGN (via core.waveform) ──────────────────────────────────
        rx_noisy = apply_awgn(rx_norm, self.snr_db)

        # Plot do sinal RX com ruído
        self._rx_curve.setData(self.t_us, rx_noisy)

        # ── 6. Filtro Casado (via core.waveform) ─────────────────────────
        tx_pulse = self._tx[:self.n_p]
        comp_disp, comp_complex = matched_filter(rx_noisy, rx_complex_norm, tx_pulse, self.n_p)

        # ── 7. Plot do MF ──────────────────────────────────────────────
        peak_comp = float(np.max(comp_disp))
        if self.normalize_plots:
            mf_disp = (comp_disp / peak_comp) if peak_comp > 1e-30 else comp_disp
            self._mf_curve.setData(self.t_us, mf_disp)
            self._mf_plot.setYRange(0, 1.05)
        else:
            self._mf_curve.setData(self.t_us, comp_disp)
            self._mf_plot.setYRange(0, max(peak_comp + 20, 100))

        # Retorna sinais processados para o ProcessingWidget
        az = self.ppi.radar.theta if (self.ppi and self.ppi.radar) else 0.0
        return {
            'comp_disp':    comp_disp,
            'comp_complex': comp_complex,
            'azimuth_deg':  az,
        }

    def _update_header(self) -> None:
        r"""Atualiza a tabela HTML de parâmetros da simulação no cabeçalho."""
        T_us    = self.T_P   * 1e6
        PRI_us  = self.T_PRI * 1e6
        B_MHz   = B / 1e6
        r_min   = self.ppi.r_max / 7.0         if self.ppi           else 0.0
        bw      = self.ppi.radar.beamwidth      if (self.ppi and self.ppi.radar) else 0.0
        c_time  = self.ppi.elapsed_time         if self.ppi           else 0.0
        t_total = self.ppi.t                    if self.ppi           else 0.0
        r_max   = self.ppi.r_max                if self.ppi           else 0.0
        c_str        = self._clutter_name
        int_mode_str = "Coherent" if self.integrator_type == "coherent" else "Non-Coherent"

        html = get_pulse_header_html(
            PRI_us=PRI_us,
            T_us=T_us,
            F_C_GHz=F_C / 1e9,
            B_MHz=B_MHz,
            snr_db=self.snr_db,
            c_str=c_str,
            r_min=r_min,
            r_max=r_max,
            bw=bw,
            int_mode_str=int_mode_str,
            c_time=c_time,
            t_total=t_total,
        )
        self._header_label.setText(html)
