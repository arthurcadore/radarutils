"""
integrator.py — Widget de visualização do Integrador de Pulsos.

Encapsula um ``PulseIntegrator`` de ``radarutils.core.integrator`` num widget
PyQtGraph pronto para uso no painel de processamento.

Tipos suportados (via ``integrator_from_str``):
  - **'noncoherent'**: soma as potências (|mti|²) dos últimos N_INT PRIs.
    Ganho de SNR ≈ √N_INT (10·log₁₀(N_INT) dB).
  - **'coherent'**: soma as amplitudes complexas IQ dos últimos N_INT PRIs
    e extrai o envelope de potência |soma|².
    Ganho de SNR ≈ N_INT (20·log₁₀(N_INT) dB), porém requer coerência
    de fase entre PRIs.

A implementação matemática reside em:
    radarutils.core.integrator
"""

import numpy as np
import pyqtgraph as pg
from radarutils.core.integrator import (
    PulseIntegrator,
    CoherentIntegrator,
    integrator_from_str,
    VALID_INTEGRATOR_TYPES,
)
from radarutils.simulator.constants import N_SAMPLES, N_INT, MIN_Y_INT


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
        integrated = w.update(mti_out, comp_complex, normalize=True)
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

        # Resolução do tipo: integrator_type tem precedência sobre o flag booleano
        if integrator_type is not None:
            self._integrator: PulseIntegrator = integrator_from_str(integrator_type, n_int)
        else:
            mode = "coherent" if coherent else "noncoherent"
            self._integrator = integrator_from_str(mode, n_int)

        self._is_coherent = isinstance(self._integrator, CoherentIntegrator)

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
        mode_str = "Coherent" if self._is_coherent else "Non-Coherent"
        legend.addItem(
            pg.PlotDataItem(pen=pg.mkPen((210, 210, 210), width=1)),
            f"Mode: {mode_str}",
        )

    def update_plot(
        self,
        integrated: np.ndarray,
        normalize: bool = True,
    ) -> None:
        """
        Atualiza o plot com o sinal integrado já computado pelo pipeline.

        Args:
            integrated:   Sinal de potência acumulada computado.
            normalize:    Se True, normaliza o eixo Y para [0, 1].
        """
        peak_int = float(np.max(integrated)) if integrated.any() else 0.0

        if normalize:
            disp = (integrated / peak_int) if peak_int > 1e-30 else integrated
            self._curve.setData(self._t_us, disp)
            self.setYRange(0, 1.05)
        else:
            self._curve.setData(self._t_us, integrated)
            self.setYRange(0, max(peak_int * 1.15, MIN_Y_INT))
