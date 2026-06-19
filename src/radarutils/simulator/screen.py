"""
screen.py — Orquestração da interface gráfica do simulador de radar.

Responsabilidades deste módulo:
  - ``prepare_output_file()`` : cria diretório e retorna caminho do vídeo MP4.
  - ``ProcessingWidget``      : QSplitter vertical que agrupa os widgets de
                                processamento (MTI → Integrador → CFAR → PPI Estimado).
  - ``MainWindow``            : janela principal Qt com layout de 3 colunas e
                                timer de atualização em 30 ms.

Os widgets individuais são importados de arquivos especializados:
  plots.py            → DetectionPlot, AmplitudePlot, PhasePlot
  pulse_modulation.py → PulseWidget
  mti.py              → MTIWidget
  integrator.py       → IntegratorWidget
  cfar.py             → CfarWidget
  ppi.py              → PPIViewer, PPIEstimatedViewer, PPIEstimatedTracker
"""

import math
import os
import numpy as np
import pyqtgraph as pg
import imageio

from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui

# ── Lógica de simulação ──────────────────────────────────────────────────
from radarutils.simulator.ppi import (
    PPI, PPIViewer, PPIEstimatedTracker, PPIEstimatedViewer,
)
from radarutils.simulator.constants import (
    C, N_SAMPLES, K_CFAR, N_INT, MAX_MATCH_DIST,
)

# ── Widgets visuais especializados ───────────────────────────────────────
from radarutils.simulator.pipeline import RadarPipeline, PipelineFrontendWidget
from radarutils.core.mti import MTI
from radarutils.core.integrator import integrator_from_str, CoherentIntegrator

# ── Widgets visuais especializados ───────────────────────────────────────
from radarutils.simulator.plots import DetectionPlot, AmplitudePlot, PhasePlot
from radarutils.simulator.mti import MTIWidget
from radarutils.simulator.integrator import IntegratorWidget
from radarutils.simulator.cfar import CfarWidget


# ──────────────────────────────────────────────────────────────────────────
#  Utilitário de saída de vídeo
# ──────────────────────────────────────────────────────────────────────────

def prepare_output_file(file_name: str = "simulation.mp4") -> str:
    r"""
    Garante que o diretório ``data/`` existe e retorna o caminho completo
    do arquivo de vídeo, removendo qualquer versão anterior.

    Args:
        file_name (str): Nome do arquivo MP4 (somente o nome, sem diretório).

    Returns:
        output_path (str): Caminho absoluto resolvido do arquivo de saída.
    """
    base_dir  = Path(__file__).resolve().parent
    data_dir  = (base_dir / "../../../data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / file_name
    if output_path.exists():
        output_path.unlink()

    return str(output_path)


class ProcessingWidget(QtWidgets.QSplitter):
    r"""
    Pipeline de processamento de sinal radar — coluna direita da UI.

    O método ``update(pulse_data)`` encadeia os quatro estágios e
    atualiza o PPI Estimado com as detecções do CFAR.

    Parâmetros de processamento (de ``constants.py``):
      N_GUARD, N_TRAIN → janelas do CA-CFAR
      K_CFAR           → fator multiplicativo do threshold
      N_INT            → número de PRIs integrados
      MAX_MATCH_DIST   → raio (m) para associar detecção a alvo real
    """

    def __init__(
        self,
        pipeline: RadarPipeline,
        baseband_widget: PipelineFrontendWidget,
    ):
        r"""
        Args:
            pipeline (RadarPipeline): Instância do pipeline.
            baseband_widget (PipelineFrontendWidget): Widget da coluna do meio.
        """
        super().__init__(QtCore.Qt.Vertical)

        self._pipeline  = pipeline
        self._ppi       = pipeline.ppi
        self._pw        = baseband_widget
        self._normalize = pipeline.config.get('normalize_plots', True)
        self._integrator_type = pipeline.config.get('integrator_type', 'noncoherent')

        self.setStyleSheet("QSplitter::handle { background-color: #555555; height: 3px; }")

        # Taxa de amostragem
        fs    = self._pw.wp.fs
        t_us  = self._pw.wp.t_us
        r_max = self._ppi.r_max

        # ── Widgets de processamento de sinal ──────────────────────────────
        # Os três widgets de sinal são pg.PlotWidget, portanto adicionados
        # num QSplitter interno (não num GraphicsLayoutWidget).
        self._sig_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._sig_splitter.setStyleSheet("QSplitter::handle { background-color: #444444; height: 2px; }")
        self.addWidget(self._sig_splitter)

        # MTIWidget — cancela clutter estático (delay-line 1 atraso)
        self._mti_w = MTIWidget(t_us=t_us)
        self._sig_splitter.addWidget(self._mti_w)

        # IntegratorWidget — acumula PRIs (coerente ou não-coerente)
        self._int_w = IntegratorWidget(
            t_us=t_us,
            integrator_type=self._integrator_type,
            n_int=N_INT,
        )
        self._int_w.setXLink(self._mti_w.getPlotItem())
        self._sig_splitter.addWidget(self._int_w)

        # CfarWidget — threshold adaptativo CA-CFAR + detecção de picos
        self._cfar_w = CfarWidget(t_us=t_us, fs=fs, alpha=K_CFAR)
        self._cfar_w.setXLink(self._mti_w.getPlotItem())
        self._sig_splitter.addWidget(self._cfar_w)

        # Proporções iguais entre os três plots de sinal
        self._sig_splitter.setSizes([1000, 1000, 1000])

        # ── PPI Estimado ───────────────────────────────────────────────────
        # Tracker: mantém estado de detecções verdadeiras e falsos alarmes
        self._tracker = PPIEstimatedTracker(
            h_hits=4096,
            max_match_dist=MAX_MATCH_DIST,
        )
        # Viewer: renderiza o rastro no display tipo PPI (tema azul-escuro)
        self._ppi_est_viewer = PPIEstimatedViewer(r_max=r_max)
        self.addWidget(self._ppi_est_viewer)

        # Divisão inicial: metade para plots de sinal, metade para PPI Estimado
        self.setSizes([1000, 1000])

        # ── PRI state: MTI e integrador instâncias puras do pipeline ───────
        self._mti      = MTI(N_SAMPLES)
        self._integrator = integrator_from_str(self._integrator_type, N_INT)
        self._is_coherent = isinstance(self._integrator, CoherentIntegrator)

        # Buffer do sinal complexo do MF anterior (para estimativa de velocidade radial)
        self._mf_prev_complex = np.zeros(N_SAMPLES, dtype=complex)

    # ──────────────────────────────────────────────────────────────────────
    #  Atualização por frame
    # ──────────────────────────────────────────────────────────────────────

    def update(self, pulse_data: dict) -> None:
        r"""
        Executa o pipeline completo de processamento para um PRI.

        Toda a computação numérica é delegada às funções de
        ``radarutils.simulator.pipeline``. Os widgets são responsáveis
        apenas pela visualização dos resultados.

        Args:
            pulse_data (dict): dict retornado por ``PulseWidget.update_pulse()``.
        """
        comp_disp    = pulse_data.get('comp_disp',    np.zeros(N_SAMPLES))
        comp_complex = pulse_data.get('comp_complex', np.zeros(N_SAMPLES, dtype=complex))
        azimuth_deg  = pulse_data.get('azimuth_deg',  0.0)
        r_max        = self._ppi.r_max
        az_rad       = math.radians(azimuth_deg)

        # ── 1. MTI (pipeline) → widget visualiza ──────────────────────────
        from radarutils.simulator.pipeline import mti_stage, integration_stage, cfar_stage, estimated_ppi_stage
        
        mti_out = mti_stage(comp_disp, self._mti)
        self._mti_w.update_plot(mti_out, normalize=self._normalize)

        # ── 2. Integração (pipeline) → widget visualiza ───────────────────
        mf_cplx_for_int = comp_complex if self._is_coherent else None
        integrated = integration_stage(mti_out, mf_cplx_for_int, self._integrator)
        self._int_w.update_plot(integrated, normalize=self._normalize)

        # ── 3. CA-CFAR (pipeline) → widget visualiza ─────────────────────
        peaks_cfar, threshold = cfar_stage(integrated, self._pw.wp.fs)
        self._cfar_w.update_plot(integrated, threshold, peaks_cfar, normalize=self._normalize)

        # ── 4. PPI Estimado (pipeline) → tracker + viewer ─────────────────
        direction = -1 if (self._ppi and self._ppi.radar and self._ppi.radar.clockwise) else 1
        self._tracker.update_sweep(azimuth_deg, direction)

        lam       = C / 10e9
        delta_phi = np.angle(np.conj(self._mf_prev_complex) * comp_complex)
        vr_map    = delta_phi / (2.0 * np.pi * self._pw.wp.T_PRI) * lam / 2.0
        self._mf_prev_complex = comp_complex.copy()

        real_targets = [(tgt.x, tgt.y) for tgt in self._ppi.targets] if self._ppi else []

        detections_xy = estimated_ppi_stage(
            peaks_cfar, azimuth_deg, self._pw.wp.t, r_max, self._pw.wp.n_p,
        )

        new_true_vrs: list[float] = []
        _mf_range_offset = self._pw.wp.n_p - 1
        
        for x, y, range_est in detections_xy:
            # Recupera o índice de pico mais próximo para vr_map
            vr_est = 0.0
            for p in peaks_cfar:
                p_c = max(0, p - _mf_range_offset)
                if abs(C * self._pw.wp.t[p_c] / 2.0 - range_est) < 1.0:
                    vr_est = float(vr_map[p])
                    break

            is_true, vr = self._tracker.add_detection(
                x, y, azimuth_deg, vr_est, real_targets
            )
            if is_true:
                new_true_vrs.append(vr)

        if new_true_vrs:
            self._tracker.last_detected_vrs = new_true_vrs.copy()

        self._ppi_est_viewer.update_view(self._tracker, az_rad)


# ──────────────────────────────────────────────────────────────────────────
#  MainWindow — Janela principal Qt
# ──────────────────────────────────────────────────────────────────────────

class MainWindow(QtWidgets.QMainWindow):
    r"""
    Janela principal da interface gráfica do simulador de radar.

    Layout de 3 colunas:
      Coluna 1 (esq.): PPIViewer real + DetectionPlot + AmplitudePlot + PhasePlot
      Coluna 2 (mid.): PulseWidget (TX / RX / Matched Filter + cabeçalho)
      Coluna 3 (dir.): ProcessingWidget (MTI / Integrador / CFAR / PPI Estimado)

    A simulação avança a cada 30 ms via QTimer.  Quando ``elapsed_time``
    atinge ``t``, o timer para e a janela fecha automaticamente.

    Se ``output_file`` for fornecido, cada frame é capturado e gravado num
    vídeo MP4 (H.264 / yuv420p) compatível com WhatsApp.  A gravação para
    automaticamente quando ``max_video_mb`` é atingido (se definido).
    """

    def __init__(
        self,
        pipeline: RadarPipeline,
        show_vectors: bool = True,
        output_file: str = None,
        max_video_mb: float = None,
        video_quality: int = 8,
    ):
        r"""
        Args:
            pipeline (RadarPipeline): Instância com configurações e PPI.
            show_vectors (bool): Se True, exibe vetores de velocidade no PPIViewer.
            output_file (str): Caminho do arquivo MP4 de saída. None = sem gravação.
            max_video_mb (float): Tamanho máximo do vídeo em MB. Encerra ao atingir.
            video_quality (int): Qualidade de compressão do vídeo (0–10). Padrão: 8.
        """
        super().__init__()
        self.setWindowTitle('PPI RADAR SIMULATOR')
        self.resize(2208, 992)

        self._pipeline      = pipeline
        self._ppi           = pipeline.ppi
        self._show_vectors  = show_vectors
        self._output_file   = output_file
        self._max_video_mb  = max_video_mb
        self._video_writer  = None
        self._video_size    = (2208, 992)

        # Inicializa o gravador de vídeo se solicitado
        if self._output_file:
            self._video_writer = imageio.get_writer(
                self._output_file,
                fps=30,
                codec="libx264",
                audio_codec="aac",
                pixelformat="yuv420p",
                quality=video_quality,
            )

        # ── Layout principal: 3 colunas ───────────────────────────────────
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # ─── COLUNA 1: PPI Real + plots de detecção ───────────────────────
        left_panel  = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self._viewer     = PPIViewer(self._ppi, show_vectors=self._show_vectors)
        self._det_plot   = DetectionPlot(ppi=self._ppi)
        self._det_plot.setYRange(0, self._ppi.r_max)
        self._amp_plot   = AmplitudePlot(ppi=self._ppi)
        self._phase_plot = PhasePlot(ppi=self._ppi)

        left_layout.addWidget(self._viewer,     stretch=6)
        left_layout.addWidget(self._det_plot,   stretch=2)
        left_layout.addWidget(self._amp_plot,   stretch=2)
        left_layout.addWidget(self._phase_plot, stretch=2)

        # ─── COLUNA 2: PipelineFrontendWidget ─────────────────────────────
        mid_panel  = QtWidgets.QWidget()
        mid_layout = QtWidgets.QVBoxLayout(mid_panel)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(4)

        self._baseband_w = PipelineFrontendWidget(pipeline=self._pipeline)
        mid_layout.addWidget(self._baseband_w, stretch=1)

        # ─── COLUNA 3: ProcessingWidget ───────────────────────────────────
        proc_panel  = QtWidgets.QWidget()
        proc_layout = QtWidgets.QVBoxLayout(proc_panel)
        proc_layout.setContentsMargins(0, 0, 0, 0)
        proc_layout.setSpacing(4)

        self._proc_widget = ProcessingWidget(
            pipeline=self._pipeline,
            baseband_widget=self._baseband_w,
        )
        proc_layout.addWidget(self._proc_widget, stretch=1)

        # Monta o layout principal (proporções relativas das 3 colunas)
        main_layout.addWidget(left_panel,  stretch=8)
        main_layout.addWidget(mid_panel,   stretch=7)
        main_layout.addWidget(proc_panel,  stretch=7)

        # Timer de atualização: 30 ms ≈ 33 fps (limitado pela renderização Qt)
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    # ──────────────────────────────────────────────────────────────────────
    #  Loop de simulação
    # ──────────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        r"""
        Chamado a cada disparo do QTimer (≈30 ms).

        Avança um passo de simulação, atualiza todos os widgets visuais e,
        se gravação estiver ativa, captura o frame atual.
        Encerra a simulação quando ``elapsed_time >= t``.
        """
        if self._ppi.elapsed_time >= self._ppi.t:
            self._timer.stop()
            print(f"Simulation finished at t={self._ppi.elapsed_time:.2f}s")
            self.close()
            return

        # Avança simulação e coleta todos os dados do frontend
        pulse_data = self._pipeline.run_step()
        detections = pulse_data['detections']
        rx_noisy = pulse_data['rx_noisy']
        comp_disp = pulse_data['comp_disp']
        comp_complex = pulse_data['comp_complex']
        azimuth_deg = pulse_data['azimuth_deg']

        # Atualiza coluna 1: PPI Real e plots de série temporal
        self._viewer.redraw()
        self._viewer.viewport().update()
        self._det_plot.add_detections(self._ppi.elapsed_time, detections)
        self._amp_plot.add_detections(self._ppi.elapsed_time, detections)
        self._phase_plot.add_detections(self._ppi.elapsed_time, detections)

        # Atualiza widgets frontend (coluna 2)
        self._baseband_w.update_plot(rx_noisy, comp_disp, azimuth_deg)

        # Atualiza coluna 3: pipeline MTI → Integrador → CFAR → PPI Estimado
        self._proc_widget.update(pulse_data)

        # Captura frame para vídeo (se gravação ativa)
        self._capture_video_frame()

    def _capture_video_frame(self) -> None:
        r"""
        Captura o widget central como imagem e grava no vídeo MP4.

        Verifica o tamanho do arquivo após cada frame; encerra a gravação
        e fecha a simulação se ``max_video_mb`` for excedido.
        """
        if not self._video_writer:
            return

        pixmap = self.centralWidget().grab()
        pixmap = pixmap.scaled(
            self._video_size[0],
            self._video_size[1],
            QtCore.Qt.IgnoreAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        img = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
        ptr = img.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.width(), 3)
        self._video_writer.append_data(arr.copy())

        # Verificação do tamanho máximo do vídeo
        if self._max_video_mb is not None:
            try:
                size_mb = os.path.getsize(self._output_file) / (1024 * 1024)
                if size_mb >= self._max_video_mb:
                    print(
                        f"Video size limit reached ({size_mb:.1f} MB "
                        f">= {self._max_video_mb} MB). Stopping simulation."
                    )
                    self._timer.stop()
                    self.close()
            except FileNotFoundError:
                pass

    def closeEvent(self, event) -> None:
        r"""Finaliza o gravador de vídeo ao fechar a janela."""
        if self._video_writer:
            self._video_writer.close()
            print(f"Video saved to {self._output_file}")
        super().closeEvent(event)