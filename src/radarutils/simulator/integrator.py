"""
integrator.py — Widget de visualização do Integrador de Pulsos.

Encapsula o ``PulseIntegrator`` de ``radarutils.core.integrator`` num widget
PyQtGraph pronto para uso no painel de processamento.

Dois modos de integração:
  - **Não-Coerente**: soma as potências (|mti|²) dos últimos N_INT PRIs.
    Ganho de SNR ≈ √N_INT (10·log₁₀(N_INT) dB).
  - **Coerente**: soma as amplitudes complexas IQ dos últimos N_INT PRIs
    e extrai o envelope de potência |soma|².
    Ganho de SNR ≈ N_INT (20·log₁₀(N_INT) dB), porém requer coerência
    de fase entre PRIs.

A implementação matemática reside em:
    radarutils.core.integrator.PulseIntegrator
"""

import numpy as np
import pyqtgraph as pg

from PySide6 import QtCore

from radarutils.core.integrator import PulseIntegrator
from radarutils.simulator.constants import N_SAMPLES, N_INT, MIN_Y_INT


class IntegratorWidget(pg.PlotWidget):
    """
    Widget de plot para a saída do Integrador de Pulsos.

    Herda de ``pg.PlotWidget`` e delega o cálculo ao ``PulseIntegrator``
    de ``radarutils.core.processing``.

    O modo (coerente vs. não-coerente) é fixado na construção.
    Uma legenda no canto superior esquerdo indica o modo ativo.

    Uso::

        w = IntegratorWidget(t_us, coherent=True, link_x_to=mti_plot)
        integrated = w.update(mti_out, comp_complex, normalize=True)
    """

    def __init__(
        self,
        t_us: np.ndarray,
        coherent: bool = False,
        n_int: int = N_INT,
        link_x_to=None,
    ):
        """
        Args:
            t_us:       Eixo de tempo em µs (compartilhado com os demais plots).
            coherent:   Se True, usa integração coerente (IQ). Padrão: não-coerente.
            n_int:      Número de PRIs a integrar. Padrão: N_INT de constants.py.
            link_x_to: PlotItem ao qual sincronizar o eixo X (opcional).
        """
        super().__init__()

        self._t_us       = t_us
        self._integrator = PulseIntegrator(n_int=n_int, coherent=coherent)
        self._coherent   = coherent

        # Configuração visual
        self.setBackground('k')
        self.setLabel('left', 'Pulse Integrator')
        self.getAxis('left').setWidth(65)
        self.showGrid(x=True, y=True, alpha=0.22)
        self.setYRange(0, 10)
        self.setMouseEnabled(x=False, y=False)

        if link_x_to is not None:
            self.setXLink(link_x_to)

        # Curva de dados (cinza claro)
        self._curve = self.plot(
            t_us, np.zeros(N_SAMPLES),
            pen=pg.mkPen((210, 210, 210), width=1),
        )

        # Legenda de modo no canto superior esquerdo
        legend = self.addLegend(colCount=1)
        legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(0, 0))
        mode_str = "Coherent" if coherent else "Non-Coherent"
        legend.addItem(
            pg.PlotDataItem(pen=pg.mkPen((210, 210, 210), width=1)),
            f"Mode: {mode_str}",
        )

    def update(
        self,
        mti_out: np.ndarray,
        comp_complex: np.ndarray = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Integra o sinal MTI nos últimos N_INT PRIs e atualiza o plot.

        Modo não-coerente::

            integrated = Σ |mti_i|²   (i = PRI atual − N_INT até atual)

        Modo coerente::

            integrated = |Σ iq_i|²    (soma vetorial → melhora SNR linear)

        Args:
            mti_out:     Sinal pós-MTI do PRI atual (real, positivo).
            comp_complex: Sinal complexo do MF do PRI atual (necessário se coerente=True).
            normalize:   Se True, normaliza o eixo Y para [0, 1].

        Returns:
            np.ndarray — sinal integrado (potência acumulada).
        """
        integrated = self._integrator.process(
            mti_out, comp_complex if self._coherent else None
        )
        peak_int = float(np.max(integrated)) if integrated.any() else 0.0

        if normalize:
            disp = (integrated / peak_int) if peak_int > 1e-30 else integrated
            self._curve.setData(self._t_us, disp)
            self.setYRange(0, 1.05)
        else:
            self._curve.setData(self._t_us, integrated)
            self.setYRange(0, max(peak_int * 1.15, MIN_Y_INT))

        return integrated
