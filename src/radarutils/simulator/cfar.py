"""
cfar.py — Widget de visualização do detector CA-CFAR.

Encapsula a função ``ca_cfar()`` de ``radarutils.core.cfar`` num widget
PyQtGraph com três elementos visuais sobrepostos:
  - Curva ciano   : sinal integrado (entrada do CFAR).
  - Curva vermelha: threshold adaptativo calculado pelo CA-CFAR.
  - Pontos amarelos: células que ultrapassaram o threshold (detecções).

O CA-CFAR (Cell-Averaging Constant False Alarm Rate) estima o piso de
ruído local usando células de treinamento vizinhas e define um threshold
multiplicativo α acima desse piso, garantindo taxa de falso alarme (FA)
aproximadamente constante independentemente do nível de ruído.

A implementação matemática reside em:
    radarutils.core.cfar.ca_cfar
"""

import numpy as np
import pyqtgraph as pg
import scipy.signal

from PySide6 import QtCore

from radarutils.core.cfar import ca_cfar
from radarutils.simulator.constants import (
    N_SAMPLES, N_GUARD, N_TRAIN, K_CFAR, MIN_CFAR_ABS, MIN_Y_CFAR,
)


class CfarWidget(pg.PlotWidget):
    """
    Widget de plot para o detector CA-CFAR.

    Herda de ``pg.PlotWidget`` e chama ``ca_cfar()`` de
    ``radarutils.core.processing``.

    Além de calcular e plotar o threshold adaptativo, detecta os picos
    que superam o threshold e os exibe como scatter (pontos amarelos).
    Retorna os índices dos picos detectados para uso no PPI Estimado.

    Uso::

        w = CfarWidget(t_us, fs, link_x_to=mti_plot)
        peaks = w.update(integrated, normalize=True)
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
            fs:           Taxa de amostragem (Hz), usada para calcular distância
                          mínima entre picos em find_peaks.
            n_guard:      Células de guarda de cada lado da célula sob teste.
            n_train:      Células de treinamento de cada lado.
            alpha:        Fator multiplicativo do threshold (limiar).
            min_cfar_abs: Threshold absoluto mínimo (suprime FA em AWGN puro).
            link_x_to:   PlotItem ao qual sincronizar o eixo X (opcional).
        """
        super().__init__()

        self._t_us         = t_us
        self._fs           = fs
        self._n_guard      = n_guard
        self._n_train      = n_train
        self._alpha        = alpha
        self._min_cfar_abs = min_cfar_abs

        # Configuração visual
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

        # Legenda em duas colunas (sinal + threshold)
        legend = self.addLegend(colCount=2)
        legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(0, 0))

    def update(self, integrated: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Calcula o threshold CA-CFAR, detecta picos e atualiza o plot.

        Pipeline interno:
          1. Chama ``ca_cfar()`` para obter o threshold adaptativo para
             cada célula do sinal integrado.
          2. Aplica piso mínimo absoluto (``min_cfar_abs``) para evitar
             falsos alarmes em regiões com apenas AWGN puro.
          3. Identifica amostras onde ``integrated > threshold``.
          4. Usa ``scipy.signal.find_peaks`` para separar picos distintos
             (distância mínima ≈ 4% da taxa de amostragem).
          5. Atualiza curvas e scatter.

        Args:
            integrated: Sinal de potência acumulada (saída do Integrador).
            normalize:  Se True, normaliza sinal e threshold para [0, 1]
                        usando o mesmo fator (preserva proporção visual).

        Returns:
            np.ndarray de inteiros — índices dos picos detectados no sinal.
        """
        # Threshold adaptativo por célula
        cfar_thresh      = ca_cfar(integrated, self._n_guard, self._n_train, self._alpha)
        effective_thresh = np.maximum(cfar_thresh, self._min_cfar_abs)

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

        # Detecção dos picos que superam o threshold
        binary   = (integrated > effective_thresh).astype(float) * integrated
        min_dist = max(1, int(0.04 * self._fs))
        peaks, _ = scipy.signal.find_peaks(binary, distance=min_dist)

        # Plot dos picos (scatter amarelo)
        if normalize and cfar_norm_factor > 1e-30:
            spots_y = integrated[peaks] / cfar_norm_factor
        else:
            spots_y = integrated[peaks]
        self._spots.setData(self._t_us[peaks], spots_y)

        return peaks
