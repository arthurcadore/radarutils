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
from radarutils.simulator.plots import DetectionPlot, AmplitudePlot, PhasePlot
from radarutils.simulator.pulse_modulation import PulseWidget
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
        ppi: PPI,
        pulse_widget: PulseWidget,
        coherent_integration: bool = False,
        normalize_plots: bool = True,
    ):
        r"""
        Args:
            ppi (PPI): Instância de PPI com radar e targets.
            pulse_widget (PulseWidget): PulseWidget da coluna do meio (fornece t_us, T_PRI, etc.).
            coherent_integration (bool): Se True, usa integração coerente no IntegratorWidget.
            normalize_plots (bool): Se True, normaliza eixos Y de todos os plots para [0, 1].
        """
        super().__init__(QtCore.Qt.Vertical)

        self._ppi       = ppi
        self._pw        = pulse_widget
        self._normalize = normalize_plots

        self.setStyleSheet("QSplitter::handle { background-color: #555555; height: 3px; }")

        # Taxa de amostragem: derivada do PulseWidget para garantir consistência
        fs    = N_SAMPLES / pulse_widget.T_PRI
        t_us  = pulse_widget.t_us
        r_max = ppi.r_max

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
            coherent=coherent_integration,
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

        # Buffer do sinal complexo do MF anterior (para estimativa de velocidade radial)
        self._mf_prev_complex = np.zeros(N_SAMPLES, dtype=complex)

    # ──────────────────────────────────────────────────────────────────────
    #  Atualização por frame
    # ──────────────────────────────────────────────────────────────────────

    def update(self, pulse_data: dict) -> None:
        r"""
        Executa o pipeline completo de processamento para um PRI.

        Sequência:
          1. **MTI**        : cancela ecos de alvos fixos.
          2. **Integrador** : acumula SNR em N_INT PRIs.
          3. **CA-CFAR**    : define threshold adaptativo e detecta picos.
          4. **PPI Estimado**: converte picos em coordenadas cartesianas,
                               classifica (verdadeiro vs. FA) e atualiza display.

        Args:
            pulse_data (dict): dict retornado por ``PulseWidget.update_pulse()``:
                          'comp_disp'    : np.ndarray — saída real do MF.
                          'comp_complex' : np.ndarray — saída complexa IQ do MF.
                          'azimuth_deg'  : float      — azimute atual do radar.
        """
        comp_disp    = pulse_data.get('comp_disp',    np.zeros(N_SAMPLES))
        comp_complex = pulse_data.get('comp_complex', np.zeros(N_SAMPLES, dtype=complex))
        azimuth_deg  = pulse_data.get('azimuth_deg',  0.0)
        r_max        = self._ppi.r_max
        az_rad       = math.radians(azimuth_deg)

        # ── 1. MTI ────────────────────────────────────────────────────────
        mti_out = self._mti_w.update(comp_disp, normalize=self._normalize)

        # ── 2. Integração ─────────────────────────────────────────────────
        integrated = self._int_w.update(mti_out, comp_complex, normalize=self._normalize)

        # ── 3. CA-CFAR ────────────────────────────────────────────────────
        peaks_cfar = self._cfar_w.update(integrated, normalize=self._normalize)

        # ── 4. PPI Estimado ───────────────────────────────────────────────
        direction = -1 if (self._ppi and self._ppi.radar and self._ppi.radar.clockwise) else 1
        self._tracker.update_sweep(azimuth_deg, direction)

        # Estimativa de velocidade radial por desvio de fase Doppler instantâneo
        # Δφ = arg(MF*_anterior · MF_atual)  →  v_r = Δφ·λ / (4π·T_PRI)
        lam       = C / 10e9   # comprimento de onda: banda X (10 GHz)
        delta_phi = np.angle(np.conj(self._mf_prev_complex) * comp_complex)
        vr_map    = delta_phi / (2.0 * np.pi * self._pw.T_PRI) * lam / 2.0
        self._mf_prev_complex = comp_complex.copy()

        # Posições reais (ground truth) dos alvos para classificação TP/FA
        real_targets = [(tgt.x, tgt.y) for tgt in self._ppi.targets] if self._ppi else []

        # Offset de bias introduzido pela cadeia correlate + roll no PulseWidget:
        #   correlate(mode='same') → desloca (n_p-1)//2 amostras
        #   roll(+n_p//2)         → adiciona n_p//2 amostras
        #   Total: n_p - 1 amostras → erro ≈ r_max/7 sem correção
        _mf_range_offset = self._pw.n_p - 1

        new_true_vrs: list[float] = []

        if len(peaks_cfar) > 0:
            r_min_blind = r_max * 0.07   # zona cega (< R_min do radar)

            for p in peaks_cfar:
                # Índice corrigido para compensar o bias de range
                p_corrected = max(0, p - _mf_range_offset)
                range_est   = C * self._pw.t[p_corrected] / 2.0

                # Ignora detecções fora da faixa válida de range
                if not (r_min_blind < range_est < r_max):
                    continue

                # Converte range + azimute em coordenadas cartesianas
                det_x  = range_est * math.cos(az_rad)
                det_y  = range_est * math.sin(az_rad)
                vr_est = float(vr_map[p])

                # Classifica: True Positive (próximo de alvo real) ou Falso Alarme
                is_true, vr = self._tracker.add_detection(
                    det_x, det_y, azimuth_deg, vr_est, real_targets
                )
                if is_true:
                    new_true_vrs.append(vr)

        # Atualiza lista de velocidades radiais da última detecção verdadeira
        if new_true_vrs:
            self._tracker.last_detected_vrs = new_true_vrs.copy()

        # Renderiza o PPI Estimado
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
        ppi: PPI,
        show_vectors: bool = True,
        output_file: str = None,
        coherent_integration: bool = False,
        clutter_type: str = "None",
        normalize_plots: bool = True,
        max_video_mb: float = None,
        video_quality: int = 8,
    ):
        r"""
        Args:
            ppi (PPI): Instância de PPI já configurada com radar e targets.
            show_vectors (bool): Se True, exibe vetores de velocidade no PPIViewer.
            output_file (str): Caminho do arquivo MP4 de saída. None = sem gravação.
            coherent_integration (bool): Passa para ProcessingWidget (modo de integração).
            clutter_type (str): Tipo de clutter ('None' ou 'Rayleigh').
            normalize_plots (bool): Se True, normaliza eixos Y dos plots de processamento.
            max_video_mb (float): Tamanho máximo do vídeo em MB. Encerra ao atingir.
            video_quality (int): Qualidade de compressão do vídeo (0–10). Padrão: 8.
        """
        super().__init__()
        self.setWindowTitle('PPI RADAR SIMULATOR')
        self.resize(2208, 992)

        self._ppi           = ppi
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

        self._viewer     = PPIViewer(ppi, show_vectors=show_vectors)
        self._det_plot   = DetectionPlot(ppi=ppi)
        self._det_plot.setYRange(0, ppi.r_max)
        self._amp_plot   = AmplitudePlot(ppi=ppi)
        self._phase_plot = PhasePlot(ppi=ppi)

        left_layout.addWidget(self._viewer,     stretch=6)
        left_layout.addWidget(self._det_plot,   stretch=2)
        left_layout.addWidget(self._amp_plot,   stretch=2)
        left_layout.addWidget(self._phase_plot, stretch=2)

        # ─── COLUNA 2: PulseWidget ────────────────────────────────────────
        mid_panel  = QtWidgets.QWidget()
        mid_layout = QtWidgets.QVBoxLayout(mid_panel)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(4)

        self._pulse_widget = PulseWidget(
            ppi=ppi,
            coherent_integration=coherent_integration,
            clutter_type=clutter_type,
            normalize_plots=normalize_plots,
        )
        mid_layout.addWidget(self._pulse_widget, stretch=1)

        # ─── COLUNA 3: ProcessingWidget ───────────────────────────────────
        proc_panel  = QtWidgets.QWidget()
        proc_layout = QtWidgets.QVBoxLayout(proc_panel)
        proc_layout.setContentsMargins(0, 0, 0, 0)
        proc_layout.setSpacing(4)

        self._proc_widget = ProcessingWidget(
            ppi=ppi,
            pulse_widget=self._pulse_widget,
            coherent_integration=coherent_integration,
            normalize_plots=normalize_plots,
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

        # Avança simulação e coleta detecções do passo atual
        detections = self._ppi.update()

        # Atualiza coluna 1: PPI Real e plots de série temporal
        self._viewer.redraw()
        self._viewer.viewport().update()
        self._det_plot.add_detections(self._ppi.elapsed_time, detections)
        self._amp_plot.add_detections(self._ppi.elapsed_time, detections)
        self._phase_plot.add_detections(self._ppi.elapsed_time, detections)

        # Atualiza coluna 2: PulseWidget → retorna sinais processados
        pulse_data = self._pulse_widget.update_pulse(detections)

        # Atualiza coluna 3: pipeline MTI → Integrador → CFAR → PPI Estimado
        if pulse_data:
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