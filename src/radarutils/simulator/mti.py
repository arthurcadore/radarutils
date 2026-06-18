"""
mti.py — Widget de visualização do filtro MTI.

Encapsula o algoritmo MTI (Moving Target Indicator) de ``radarutils.core.mti``
num widget PyQtGraph pronto para uso no painel de processamento.

O MTI (cancelador delay-line de 1 atraso) subtrai o pulso comprimido
do PRI anterior do atual, cancelando ecos de alvos fixos (clutter
estático) e realçando retornos de alvos em movimento.

A implementação matemática reside em:
    radarutils.core.mti.MTI
"""

import numpy as np
import pyqtgraph as pg

from radarutils.core.mti import MTI
from radarutils.simulator.constants import N_SAMPLES, MIN_Y_MTI


class MTIWidget(pg.PlotWidget):
    """
    Widget de plot para a saída do filtro MTI.

    Herda de ``pg.PlotWidget`` e delega o cálculo ao ``MTI`` de
    ``radarutils.core.processing``.

    Internamente mantém o estado do pulso anterior (necessário para a
    subtração delay-line).  A cada chamada de ``update()``:
      1. Calcula |MF_atual − MF_anterior|.
      2. Atualiza o plot (normalizado 0→1 ou em unidades absolutas).
      3. Retorna o sinal MTI para a etapa seguinte (Integrador).

    Uso::

        w = MTIWidget(t_us=pulse_widget.t_us, link_x_to=other_plot)
        mti_out = w.update(comp_disp, normalize=True)
    """

    def __init__(self, t_us: np.ndarray, link_x_to=None):
        """
        Args:
            t_us:       Eixo de tempo em µs (compartilhado com os demais plots).
            link_x_to: PlotItem ao qual sincronizar o eixo X (opcional).
        """
        super().__init__()

        self._t_us = t_us
        self._mti  = MTI(N_SAMPLES)

        # Configuração visual
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

    def update(self, comp_disp: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Aplica o filtro MTI ao sinal atual e atualiza o plot.

        O filtro opera como cancelador delay-line de 1 atraso:
          mti_out = |comp_disp − comp_disp_anterior|

        Isso cancela retornos estacionários (mesma amplitude e fase em
        PRIs consecutivos) e preserva retornos de alvos em movimento.

        Args:
            comp_disp: Saída do filtro casado (envelope real) do PRI atual.
            normalize: Se True, normaliza o eixo Y para [0, 1].

        Returns:
            np.ndarray — sinal pós-MTI (valor absoluto da diferença).
        """
        mti_out  = self._mti.process(comp_disp)
        peak_mti = float(np.max(mti_out)) if mti_out.any() else 0.0

        if normalize:
            disp = (mti_out / peak_mti) if peak_mti > 1e-30 else mti_out
            self._curve.setData(self._t_us, disp)
            self.setYRange(0, 1.05)
        else:
            self._curve.setData(self._t_us, mti_out)
            self.setYRange(0, max(peak_mti * 1.15, MIN_Y_MTI))

        return mti_out
