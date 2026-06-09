"""
plots.py — Gráficos de séries temporais das detecções do radar.

Contém três widgets de dispersão (coluna esquerda da UI):
  - DetectionPlot  : Range (m) vs. Tempo (s)
  - AmplitudePlot  : Potência RX (dBm) vs. Tempo (s)
  - PhasePlot      : Fase teórica (rad) vs. Tempo (s)

Todos usam a mesma paleta Jet mapeada pelo erro angular (deg_error):
  vermelho = centro do feixe (deg_error ≈ 0)
  azul     = borda do feixe  (|deg_error| ≈ beamwidth/2)
"""

import numpy as np
import pyqtgraph as pg

from radarutils.simulator.detection import DetectionRecord
from radarutils.simulator.constants import WAVELENGTH_M

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
