"""
plots.py — Widgets de visualização do simulador de radar.

Contém todos os widgets de plot agrupados em três seções:

1. Série temporal (coluna esquerda da UI):
   - DetectionPlot  : Range (m) vs. Tempo (s)
   - AmplitudePlot  : Potência RX (dBm) vs. Tempo (s)
   - PhasePlot      : Fase teórica (rad) vs. Tempo (s)

   Todos usam a paleta Jet mapeada pelo erro angular (deg_error):
     vermelho = centro do feixe (deg_error ≈ 0)
     azul     = borda do feixe  (|deg_error| ≈ beamwidth/2)

2. Pipeline de processamento de sinal (coluna direita da UI):
   - MTIWidget       : Saída do filtro MTI (cancelamento de clutter estático)
   - IntegratorWidget: Acumulação de PRIs (coerente ou não-coerente)
   - CfarWidget      : Threshold adaptativo CA-CFAR + detecção de picos

3. Banda base (coluna central da UI):
   - PipelineFrontendWidget: TX / RX Baseband / Matched Filter + cabeçalho
"""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from radarutils.simulator.detection import DetectionRecord
from radarutils.simulator.constants import (
    WAVELENGTH_M, N_SAMPLES, N_INT, K_CFAR,
    MIN_Y_MTI, MIN_Y_INT, MIN_Y_CFAR,
    N_GUARD, N_TRAIN, MIN_CFAR_ABS,
    B, F_C,
)
from radarutils.simulator.html_contents import get_pulse_header_html
from radarutils.core.mti import MTI
from radarutils.core.integrator import (
    PulseIntegrator, CoherentIntegrator, integrator_from_str,
)

# ──────────────────────────────────────────────────────────────
#  Paleta Jet compartilhada (Blue→Cyan→Green→Yellow→Red)
# ──────────────────────────────────────────────────────────────
_JET_COLORMAP = pg.ColorMap(
    pos=[0.0, 0.25, 0.5, 0.75, 1.0],
    color=[
        (0,   0,   255, 255),
        (0,   255, 255, 255),
        (0,   255, 0,   255),
        (255, 255, 0,   255),
        (255, 0,   0,   255),
    ],
)

# Símbolos disponíveis por índice de alvo (round-robin)
_SYMBOLS = ['o', 's', 't', 'd', '+', 'x', 'star', 'p', 'h']

# Janela deslizante padrão dos gráficos de série temporal (segundos)
_WINDOW_SIZE = 15.0


def _norm_deg_error(deg_error: float, beamwidth: float) -> float:
    """
    Normaliza um erro angular para [0, 1] em relação à meia-beamwidth.

    Args:
        deg_error:  Erro angular em graus (centro do feixe = 0).
        beamwidth:  Largura de feixe total em graus.

    Returns:
        Valor normalizado ∈ [0, 1].  0 = centro, 1 = borda.
    """
    bw_half = max(beamwidth / 2.0, 0.001)
    return min(abs(deg_error) / bw_half, 1.0)


# ──────────────────────────────────────────────────────────────
#  DetectionPlot — Range vs. Tempo
# ──────────────────────────────────────────────────────────────

class DetectionPlot(pg.PlotWidget):
    """
    Gráfico de dispersão: Range (m) vs. Tempo (s).

    Cada ponto representa uma detecção registrada pelo radar.
    A cor segue a paleta Jet indexada pelo erro angular normalizado:
      vermelho → detecção no centro do feixe (máximo SNR),
      azul     → detecção na borda do feixe  (mínimo SNR).

    Uso::

        plot = DetectionPlot(ppi=ppi)
        plot.add_detections(t=sim.elapsed_time, detection_list=detections)
    """

    def __init__(self, ppi=None):
        """
        Args:
            ppi: Instância de PPI. Usada para ler a beamwidth do radar.
                 Pode ser None (usa beamwidth padrão de 10°).
        """
        super().__init__()
        self.ppi = ppi

        # Configuração visual base
        self.setBackground('k')
        self.setLabel('left', 'Range', units='m')
        self.getAxis('left').setWidth(65)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setYRange(0, 1000)

        # Item de dispersão
        self.plot_data = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None))
        self.addItem(self.plot_data)

        # Histórico acumulado (cresce ao longo da simulação)
        self._times:   list[float] = []
        self._ranges:  list[float] = []
        self._brushes: list        = []
        self._symbols: list[str]   = []
        self._current_time: float  = 0.0

    def add_detections(self, t: float, detection_list: list[DetectionRecord]) -> None:
        """
        Adiciona novas detecções ao histórico e redesenha o gráfico.

        A janela X desliza automaticamente para mostrar os últimos
        ``_WINDOW_SIZE`` segundos quando t > _WINDOW_SIZE.

        Args:
            t:               Tempo atual da simulação (s).
            detection_list:  Lista de DetectionRecord do passo atual.
        """
        self._current_time = t
        bw = self.ppi.radar.beamwidth if (self.ppi and self.ppi.radar) else 10.0

        for rec in detection_list:
            self._times.append(rec.time)
            self._ranges.append(rec.range_m)
            self._symbols.append(_SYMBOLS[rec.target_idx % len(_SYMBOLS)])

            # Cor: vermelho = centro (norm=0), azul = borda (norm=1)
            norm  = _norm_deg_error(rec.deg_error, bw)
            color = _JET_COLORMAP.mapToQColor(1.0 - norm)
            color.setAlpha(180)
            self._brushes.append(pg.mkBrush(color))

        if self._times:
            self.plot_data.setData(
                x=self._times,
                y=self._ranges,
                brush=self._brushes,
                symbol=self._symbols,
            )

        # Janela X deslizante
        if t > _WINDOW_SIZE:
            self.setXRange(t - _WINDOW_SIZE, t)
        else:
            self.setXRange(0, _WINDOW_SIZE)


# ──────────────────────────────────────────────────────────────
#  AmplitudePlot — Potência RX (dBm) vs. Tempo
# ──────────────────────────────────────────────────────────────

class AmplitudePlot(pg.PlotWidget):
    """
    Gráfico de dispersão: Potência RX (dBm) vs. Tempo (s).

    Exibe a potência recebida calculada pela Equação do Radar para cada
    detecção, colorida pelo erro angular (mesma paleta Jet do DetectionPlot).

    O eixo Y é auto-dimensionado para os pontos visíveis na janela atual,
    com margem de ±5 dBm. Isso torna fácil observar variações de amplitude
    ao longo do tempo (ex.: alvo se aproximando ou emergindo da borda do feixe).
    """

    def __init__(self, ppi=None):
        """
        Args:
            ppi: Instância de PPI. Usada para ler a beamwidth do radar.
        """
        super().__init__()
        self.ppi = ppi

        self.setBackground('k')
        self.setLabel('left', 'Amplitude', units='dBm')
        self.getAxis('left').setWidth(65)
        self.showGrid(x=True, y=True, alpha=0.3)

        self.plot_data = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None))
        self.addItem(self.plot_data)

        self._times:   list[float] = []
        self._powers:  list[float] = []
        self._brushes: list        = []
        self._symbols: list[str]   = []
        self._current_time: float  = 0.0

    def add_detections(self, t: float, detection_list: list[DetectionRecord]) -> None:
        """
        Adiciona potências RX das novas detecções ao histórico.

        O eixo X desliza com a simulação. O eixo Y se ajusta ao intervalo
        visível (janela deslizante) com margem de ±5 dBm.

        Args:
            t:               Tempo atual da simulação (s).
            detection_list:  Lista de DetectionRecord do passo atual.
        """
        self._current_time = t
        bw = self.ppi.radar.beamwidth if (self.ppi and self.ppi.radar) else 10.0

        for rec in detection_list:
            self._times.append(rec.time)
            self._powers.append(rec.rx_power_dbm)
            self._symbols.append(_SYMBOLS[rec.target_idx % len(_SYMBOLS)])

            norm  = _norm_deg_error(rec.deg_error, bw)
            color = _JET_COLORMAP.mapToQColor(1.0 - norm)
            color.setAlpha(200)
            self._brushes.append(pg.mkBrush(color))

        if self._times:
            self.plot_data.setData(
                x=self._times,
                y=self._powers,
                brush=self._brushes,
                symbol=self._symbols,
            )

        # Ajuste do eixo X (janela deslizante)
        if t > _WINDOW_SIZE:
            x_min = t - _WINDOW_SIZE
            self.setXRange(x_min, t)
            visible = [p for tt, p in zip(self._times, self._powers) if tt >= x_min]
        else:
            self.setXRange(0, _WINDOW_SIZE)
            visible = self._powers

        # Auto-scale do eixo Y para os dados visíveis
        if visible:
            self.setYRange(min(visible) - 5, max(visible) + 5)


# ──────────────────────────────────────────────────────────────
#  PhasePlot — Fase teórica RX (rad) vs. Tempo
# ──────────────────────────────────────────────────────────────

class PhasePlot(pg.PlotWidget):
    """
    Gráfico de dispersão: Fase teórica RX (rad) vs. Tempo (s).

    A fase é calculada como::

        φ = (4π · R / λ) mod 2π,  mapeada para (-π, π]

    onde λ = 0.03 m (banda X, 10 GHz).  Isso corresponde ao retardo de
    fase de ida e volta do pulso refletido.  O padrão de dispersão em
    torno da fase indica coerência do sinal ao longo do tempo.
    """

    def __init__(self, ppi=None):
        """
        Args:
            ppi: Instância de PPI. Usada para ler a beamwidth do radar.
        """
        super().__init__()
        self.ppi = ppi

        self.setBackground('k')
        self.setLabel('bottom', 'Time', units='s')
        self.setLabel('left', 'Phase', units='rad')
        self.getAxis('left').setWidth(65)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setYRange(-np.pi * 1.1, np.pi * 1.1)

        self.plot_data = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None))
        self.addItem(self.plot_data)

        self._times:   list[float] = []
        self._phases:  list[float] = []
        self._brushes: list        = []
        self._symbols: list[str]   = []
        self._current_time: float  = 0.0

    @staticmethod
    def _calc_phase(range_m: float) -> float:
        """
        Calcula a fase de retardo de ida e volta para um dado range.

        φ = (4π·R/λ) mod 2π, mapeada para (-π, π].

        Args:
            range_m: Distância radar-alvo (m).

        Returns:
            Fase em radianos ∈ (-π, π].
        """
        phi = (4.0 * np.pi * range_m / WAVELENGTH_M) % (2.0 * np.pi)
        if phi > np.pi:
            phi -= 2.0 * np.pi
        return phi

    def add_detections(self, t: float, detection_list: list[DetectionRecord]) -> None:
        """
        Adiciona fases calculadas das novas detecções ao histórico.

        Args:
            t:               Tempo atual da simulação (s).
            detection_list:  Lista de DetectionRecord do passo atual.
        """
        self._current_time = t
        bw = self.ppi.radar.beamwidth if (self.ppi and self.ppi.radar) else 10.0

        for rec in detection_list:
            self._times.append(rec.time)
            self._phases.append(self._calc_phase(rec.range_m))
            self._symbols.append(_SYMBOLS[rec.target_idx % len(_SYMBOLS)])

            norm  = _norm_deg_error(rec.deg_error, bw)
            color = _JET_COLORMAP.mapToQColor(1.0 - norm)
            color.setAlpha(200)
            self._brushes.append(pg.mkBrush(color))

        if self._times:
            self.plot_data.setData(
                x=self._times,
                y=self._phases,
                brush=self._brushes,
                symbol=self._symbols,
            )

        # Janela X deslizante
        if t > _WINDOW_SIZE:
            self.setXRange(t - _WINDOW_SIZE, t)
        else:
            self.setXRange(0, _WINDOW_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline de processamento de sinal
# ══════════════════════════════════════════════════════════════════════════════

class MTIWidget(pg.PlotWidget):
    """
    Widget de plot para a saída do filtro MTI.

    Herda de ``pg.PlotWidget`` e delega o cálculo ao ``MTI`` de
    ``radarutils.core.mti``.

    A cada chamada de ``update_plot()``, exibe o sinal pós-MTI
    (normalizado 0→1 ou em unidades absolutas).

    Uso::

        w = MTIWidget(t_us=pulse_widget.t_us)
        w.update_plot(mti_out, normalize=True)
    """

    def __init__(self, t_us: np.ndarray, link_x_to=None):
        """
        Args:
            t_us:       Eixo de tempo em µs (compartilhado com os demais plots).
            link_x_to: PlotItem ao qual sincronizar o eixo X (opcional).
        """
        super().__init__()

        self._t_us = t_us

        self.setBackground('k')
        self.setLabel('left', 'MTI')
        self.getAxis('left').setWidth(65)
        self.showGrid(x=True, y=True, alpha=0.22)
        self.setYRange(0, 10)
        self.setMouseEnabled(x=False, y=False)

        if link_x_to is not None:
            self.setXLink(link_x_to)

        # Curva de dados (amarelo — realce de alvos móveis)
        self._curve = self.plot(
            t_us, np.zeros(N_SAMPLES),
            pen=pg.mkPen((255, 255, 0), width=1),
        )

    def update_plot(self, mti_out: np.ndarray, normalize: bool = True) -> None:
        """
        Atualiza o plot com o sinal pós-MTI já computado pelo pipeline.

        Args:
            mti_out:   Sinal pós-MTI calculado (valor absoluto da diferença).
            normalize: Se True, normaliza o eixo Y para [0, 1].
        """
        peak_mti = float(np.max(mti_out)) if mti_out.any() else 0.0

        if normalize:
            disp = (mti_out / peak_mti) if peak_mti > 1e-30 else mti_out
            self._curve.setData(self._t_us, disp)
            self.setYRange(0, 1.05)
        else:
            self._curve.setData(self._t_us, mti_out)
            self.setYRange(0, max(peak_mti * 1.15, MIN_Y_MTI))


class IntegratorWidget(pg.PlotWidget):
    """
    Widget de plot para a saída do Integrador de Pulsos.

    Herda de ``pg.PlotWidget`` e delega o cálculo a uma instância de
    ``PulseIntegrator`` (CoherentIntegrator ou NonCoherentIntegrator).

    O modo de integração é fixado na construção via ``integrator_type``
    ou, alternativamente, pelo flag booleano legado ``coherent``.
    Uma legenda no canto superior esquerdo indica o modo ativo.

    Uso::

        w = IntegratorWidget(t_us, integrator_type='coherent', n_int=8)
        w.update_plot(integrated, normalize=True)
    """

    def __init__(
        self,
        t_us: np.ndarray,
        coherent: bool = False,
        n_int: int = N_INT,
        link_x_to=None,
        integrator_type: str | None = None,
    ):
        """
        Args:
            t_us:             Eixo de tempo em µs (compartilhado com os demais plots).
            coherent:         Flag legado — se True, equivale a integrator_type='coherent'.
                              Ignorado quando ``integrator_type`` for fornecido explicitamente.
            n_int:            Número de PRIs a integrar. Padrão: N_INT de constants.py.
            link_x_to:        PlotItem ao qual sincronizar o eixo X (opcional).
            integrator_type:  Nome do integrador: 'noncoherent' ou 'coherent'.
                              Se None, usa o flag ``coherent`` para compatibilidade.
        """
        super().__init__()

        self._t_us = t_us

        # integrator_type tem precedência sobre o flag booleano legado
        if integrator_type is not None:
            self._integrator: PulseIntegrator = integrator_from_str(integrator_type, n_int)
        else:
            mode = "coherent" if coherent else "noncoherent"
            self._integrator = integrator_from_str(mode, n_int)

        self._is_coherent = isinstance(self._integrator, CoherentIntegrator)

        self.setBackground('k')
        self.setLabel('left', 'Pulse Integrator')
        self.getAxis('left').setWidth(65)
        self.showGrid(x=True, y=True, alpha=0.22)
        self.setYRange(0, 10)
        self.setMouseEnabled(x=False, y=False)

        if link_x_to is not None:
            self.setXLink(link_x_to)

        self._curve = self.plot(
            t_us, np.zeros(N_SAMPLES),
            pen=pg.mkPen((210, 210, 210), width=1),
        )

        legend = self.addLegend(colCount=1)
        legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(0, 0))
        mode_str = "Coherent" if self._is_coherent else "Non-Coherent"
        legend.addItem(
            pg.PlotDataItem(pen=pg.mkPen((210, 210, 210), width=1)),
            f"Mode: {mode_str}",
        )

    def update_plot(self, integrated: np.ndarray, normalize: bool = True) -> None:
        """
        Atualiza o plot com o sinal integrado já computado pelo pipeline.

        Args:
            integrated: Sinal de potência acumulada computado.
            normalize:  Se True, normaliza o eixo Y para [0, 1].
        """
        peak_int = float(np.max(integrated)) if integrated.any() else 0.0

        if normalize:
            disp = (integrated / peak_int) if peak_int > 1e-30 else integrated
            self._curve.setData(self._t_us, disp)
            self.setYRange(0, 1.05)
        else:
            self._curve.setData(self._t_us, integrated)
            self.setYRange(0, max(peak_int * 1.15, MIN_Y_INT))


class CfarWidget(pg.PlotWidget):
    """
    Widget de plot para o detector CA-CFAR.

    Herda de ``pg.PlotWidget`` e exibe três elementos sobrepostos:
      - Curva ciano    : sinal integrado (entrada do CFAR).
      - Curva vermelha : threshold adaptativo CA-CFAR.
      - Pontos amarelos: picos que ultrapassaram o threshold (detecções).

    Uso::

        w = CfarWidget(t_us, fs)
        w.update_plot(integrated, effective_thresh, peaks, normalize=True)
    """

    def __init__(
        self,
        t_us: np.ndarray,
        fs: float,
        n_guard: int = N_GUARD,
        n_train: int = N_TRAIN,
        alpha: float = K_CFAR,
        min_cfar_abs: float = MIN_CFAR_ABS,
        link_x_to=None,
    ):
        """
        Args:
            t_us:         Eixo de tempo em µs (compartilhado com os demais plots).
            fs:           Taxa de amostragem (Hz).
            n_guard:      Células de guarda de cada lado da célula sob teste.
            n_train:      Células de treinamento de cada lado.
            alpha:        Fator multiplicativo do threshold.
            min_cfar_abs: Threshold absoluto mínimo (suprime FA em AWGN puro).
            link_x_to:    PlotItem ao qual sincronizar o eixo X (opcional).
        """
        super().__init__()

        self._t_us         = t_us
        self._fs           = fs
        self._n_guard      = n_guard
        self._n_train      = n_train
        self._alpha        = alpha
        self._min_cfar_abs = min_cfar_abs

        self.setBackground('k')
        self.setLabel('left', 'CA-CFAR')
        self.getAxis('left').setWidth(65)
        self.setLabel('bottom', 'Tempo (µs)')
        self.showGrid(x=True, y=True, alpha=0.22)
        self.setYRange(0, 10)
        self.setMouseEnabled(x=False, y=False)

        if link_x_to is not None:
            self.setXLink(link_x_to)

        # Curva do sinal integrado (ciano)
        self._sig_curve = self.plot(
            t_us, np.zeros(N_SAMPLES),
            pen=pg.mkPen((0, 190, 255), width=1),
            name="Sinal (Rx)",
        )
        # Curva do threshold CFAR (vermelho tracejado)
        self._thr_curve = self.plot(
            t_us, np.zeros(N_SAMPLES),
            pen=pg.mkPen((255, 80, 80), width=1, style=QtCore.Qt.DashLine),
            name="Threshold CFAR",
        )
        # Pontos de pico detectado (amarelo)
        self._spots = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 220), symbol='o',
        )
        self.addItem(self._spots)

        legend = self.addLegend(colCount=2)
        legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(0, 0))

    def update_plot(
        self,
        integrated: np.ndarray,
        effective_thresh: np.ndarray,
        peaks: np.ndarray,
        normalize: bool = True,
    ) -> None:
        """
        Atualiza o plot CFAR com os dados já computados pelo pipeline.

        Args:
            integrated:       Sinal de potência acumulada.
            effective_thresh: Threshold adaptativo (limiar + piso absoluto).
            peaks:            Índices dos picos detectados.
            normalize:        Se True, normaliza sinal e threshold para [0, 1].
        """
        peak_int         = float(np.max(integrated)) if integrated.any() else 0.0
        cfar_norm_factor = max(peak_int, float(np.max(effective_thresh)))

        if normalize:
            if cfar_norm_factor > 1e-30:
                sig_disp = integrated       / cfar_norm_factor
                thr_disp = effective_thresh / cfar_norm_factor
            else:
                sig_disp = integrated
                thr_disp = effective_thresh
            self._sig_curve.setData(self._t_us, sig_disp)
            self._thr_curve.setData(self._t_us, thr_disp)
            self.setYRange(0, 1.05)
        else:
            self._sig_curve.setData(self._t_us, integrated)
            self._thr_curve.setData(self._t_us, effective_thresh)
            self.setYRange(0, max(cfar_norm_factor * 1.2, MIN_Y_CFAR))

        # Plot dos picos (scatter amarelo)
        if len(peaks) > 0:
            if normalize and cfar_norm_factor > 1e-30:
                spots_y = integrated[peaks] / cfar_norm_factor
            else:
                spots_y = integrated[peaks]
            self._spots.setData(self._t_us[peaks], spots_y)
        else:
            self._spots.setData([], [])


# ══════════════════════════════════════════════════════════════════════════════
#  Banda base — TX / RX / Matched Filter
# ══════════════════════════════════════════════════════════════════════════════

class PipelineFrontendWidget(QtWidgets.QSplitter):
    r"""
    Painel de visualização da banda base gerada pelo Pipeline.

    Exibe três plots empilhados verticalmente:
      - TX Pulse         : pulso transmitido (chirp normalizado).
      - RX Baseband      : sinal recebido com ruído AWGN + clutter.
      - Matched Filter   : saída do filtro casado (envelope de potência).

    Um cabeçalho HTML no topo resume os principais parâmetros do radar
    (PRI, largura de pulso, frequência, SNR, clutter, modo de integração).

    Uso::

        w = PipelineFrontendWidget(pipeline)
        w.update_plot(rx_noisy, comp_disp, azimuth_deg)
    """

    def __init__(self, pipeline) -> None:
        """
        Args:
            pipeline (RadarPipeline): Instância com configurações e WaveformParams.
        """
        super().__init__(QtCore.Qt.Vertical)

        self.pipeline = pipeline
        self.wp = pipeline.wp

        self.setStyleSheet("QSplitter::handle { background-color: #555555; height: 3px; }")

        rx_start_us = self.wp.T_P * 1e6

        self._build_header()
        self._build_plots(rx_start_us)

        self._glw.ci.layout.setRowStretchFactor(0, 1)
        self._glw.ci.layout.setRowStretchFactor(1, 1)
        self._glw.ci.layout.setRowStretchFactor(2, 1.5)

        self.setSizes([200, 800])

    def _build_header(self) -> None:
        self._header_label = QtWidgets.QLabel()
        self._header_label.setStyleSheet("background-color: black;")
        self._header_label.setAlignment(QtCore.Qt.AlignCenter)
        self.addWidget(self._header_label)

    def _build_plots(self, rx_start_us: float) -> None:
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground('k')
        self._glw.ci.layout.setSpacing(12)
        self.addWidget(self._glw)

        # ── Plot TX ────────────────────────────────────────────────────────
        self._tx_plot = self._glw.addPlot(row=0, col=0)
        self._tx_plot.setLabel('left', 'TX  Pulse')
        self._tx_plot.getAxis('left').setWidth(65)
        self._tx_plot.showGrid(x=True, y=True, alpha=0.22)
        self._tx_plot.setYRange(-1.2, 1.2)
        self._tx_plot.setMouseEnabled(x=False, y=False)
        self._tx_curve = self._tx_plot.plot(
            self.wp.t_us, self.wp.tx, pen=pg.mkPen((0, 200, 255), width=1),
        )
        self._tx_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen((0, 80, 100), width=1, style=QtCore.Qt.DotLine),
        ))
        self._tx_plot.addItem(pg.InfiniteLine(
            pos=rx_start_us, angle=90, pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine),
        ))

        # ── Plot RX ────────────────────────────────────────────────────────
        self._rx_plot = self._glw.addPlot(row=1, col=0)
        self._rx_plot.setLabel('left', 'RX Baseband')
        self._rx_plot.getAxis('left').setWidth(65)
        self._rx_plot.showGrid(x=True, y=True, alpha=0.22)
        self._rx_plot.setYRange(-1.2, 1.2)
        self._rx_plot.setMouseEnabled(x=False, y=False)
        self._rx_plot.setXLink(self._tx_plot)
        self._rx_curve = self._rx_plot.plot(
            self.wp.t_us, np.zeros(N_SAMPLES), pen=pg.mkPen((255, 140, 0), width=1),
        )
        self._rx_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen((80, 50, 0), width=1, style=QtCore.Qt.DotLine),
        ))
        self._rx_plot.addItem(pg.InfiniteLine(
            pos=rx_start_us, angle=90, pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine),
        ))

        # ── Plot MF (Matched Filter) ───────────────────────────────────────
        self._mf_plot = self._glw.addPlot(row=2, col=0)
        self._mf_plot.setLabel('left', 'Matched Filter Out')
        self._mf_plot.getAxis('left').setWidth(65)
        self._mf_plot.setLabel('bottom', 'Tempo (µs)')
        self._mf_plot.showGrid(x=True, y=True, alpha=0.22)
        self._mf_plot.setYRange(0, 100)
        self._mf_plot.setMouseEnabled(x=False, y=False)
        self._mf_plot.setXLink(self._tx_plot)
        self._mf_curve = self._mf_plot.plot(
            self.wp.t_us, np.zeros(N_SAMPLES), pen=pg.mkPen((255, 0, 255), width=1),
        )
        self._mf_plot.addItem(pg.InfiniteLine(
            pos=rx_start_us, angle=90, pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine),
        ))

    def update_plot(
        self,
        rx_noisy: np.ndarray,
        comp_disp: np.ndarray,
        azimuth_deg: float,
    ) -> None:
        """
        Atualiza os plots RX e Matched Filter com os dados do passo atual.

        Args:
            rx_noisy:    Sinal RX com ruído (para o plot RX Baseband).
            comp_disp:   Saída do filtro casado (envelope real).
            azimuth_deg: Azimute atual (apenas usado para acionar _update_header).
        """
        self._rx_curve.setData(self.wp.t_us, rx_noisy)
        self._update_header()

        peak_comp = float(np.max(comp_disp))
        if self.pipeline.config.get('normalize_plots', True):
            mf_disp = (comp_disp / peak_comp) if peak_comp > 1e-30 else comp_disp
            self._mf_curve.setData(self.wp.t_us, mf_disp)
            self._mf_plot.setYRange(0, 1.05)
        else:
            self._mf_curve.setData(self.wp.t_us, comp_disp)
            self._mf_plot.setYRange(0, max(peak_comp + 20, 100))

    def _update_header(self) -> None:
        """Recalcula e renderiza o cabeçalho HTML com os parâmetros do radar."""
        T_us    = self.wp.T_P   * 1e6
        PRI_us  = self.wp.T_PRI * 1e6
        B_MHz   = B / 1e6
        ppi     = self.pipeline.ppi
        r_min   = ppi.r_max / 7.0 if ppi else 0.0
        bw      = ppi.radar.beamwidth if (ppi and ppi.radar) else 0.0
        c_time  = ppi.elapsed_time if ppi else 0.0
        t_total = ppi.t if ppi else 0.0
        r_max   = ppi.r_max if ppi else 0.0
        c_str   = (
            type(self.pipeline.clutter).__name__.replace("Clutter", "")
            if self.pipeline.clutter else "None"
        )
        int_mode_str = (
            "Coherent"
            if self.pipeline.config.get('integrator_type') == "coherent"
            else "Non-Coherent"
        )

        html = get_pulse_header_html(
            PRI_us=PRI_us, T_us=T_us, F_C_GHz=F_C / 1e9, B_MHz=B_MHz,
            snr_db=self.pipeline.snr_db, c_str=c_str, r_min=r_min, r_max=r_max,
            bw=bw, int_mode_str=int_mode_str, c_time=c_time, t_total=t_total,
        )
        self._header_label.setText(html)
