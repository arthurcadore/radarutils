r"""
pipeline.py — Funções puras, orquestração e UI do pipeline de radar.

Este módulo centraliza:
1. As funções matemáticas puras de cada estágio do pipeline.
2. A classe `RadarPipeline` que instancia o PPI e orquestra a simulação numérico-temporal.
3. A classe `PipelineFrontendWidget` que exibe a banda base (fusão solicitada).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import scipy.signal
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from radarutils.core.clutter import Clutter, clutter_from_str
from radarutils.core.cfar import ca_cfar
from radarutils.core.mti import MTI
from radarutils.core.integrator import PulseIntegrator
from radarutils.core.waveform import (
    build_rx_echo,
    apply_awgn,
    matched_filter,
    doppler_frequency,
)
from radarutils.simulator.constants import (
    C, F_C, B, N_SAMPLES, DEFAULT_SNR_DB,
    N_GUARD, N_TRAIN, K_CFAR, MIN_CFAR_ABS,
)
from radarutils.simulator.detection import DetectionRecord
from radarutils.simulator.pulse import WaveformParams, waveform_params_from_rmax
from radarutils.simulator.ppi import PPI
from radarutils.simulator.html_contents import get_pulse_header_html

class RadarPipeline:
    r"""
    Instancia o PPI a partir da configuração e orquestra a geração de dados.
    """
    def __init__(self, config: dict):
        self.config = config
        
        # 1. Instancia o PPI
        self.ppi = PPI(
            dimensions=config.get('dimensions', (2000, 2000)),
            dt=config.get('dt', 0.03),
            t=config.get('t', 10.0)
        )
        self.ppi.r_max = config.get('r_max', 1000.0)
        
        # Configura PPI
        if config.get('radar'):
            self.ppi.add_radar(**config['radar'])
        for t in config.get('targets', []):
            self.ppi.add_target(**t)
        for t in config.get('orbital_targets', []):
            self.ppi.add_orbital_target(**t)
        for t in config.get('nested_orbital_targets', []):
            self.ppi.add_nested_orbital_target(**t)
        for c in config.get('regional_clutter', []):
            self.ppi.add_regional_clutter(**c)
            
        # Parâmetros de Pipeline
        self.snr_db = config.get('snr_db', DEFAULT_SNR_DB)
        clutter_type = config.get('clutter_type', 'None')
        try:
            self.clutter = clutter_from_str(clutter_type, N_SAMPLES, amplitude=1e-6)
        except ValueError:
            self.clutter = None
            
        pt = self.ppi.radar.pt if self.ppi.radar else 1000.0
        self.wp = waveform_params_from_rmax(self.ppi.r_max, pt)

    def run_step(self) -> dict:
        r"""
        Avança 1 passo no PPI e gera a saída da banda base (frontend).
        """
        detections, active_regional = self.ppi.update()
        rx_noisy, comp_disp, comp_complex = frontend_stage(
            detections, self.ppi.targets, self.wp, self.clutter, active_regional, self.snr_db
        )
        
        azimuth_deg = self.ppi.radar.theta if self.ppi.radar else 0.0
        
        return {
            'detections': detections,
            'rx_noisy': rx_noisy,
            'comp_disp': comp_disp,
            'comp_complex': comp_complex,
            'azimuth_deg': azimuth_deg
        }


def build_rx_signal(
    detections: list[DetectionRecord],
    wp: WaveformParams,
    targets: list,
) -> tuple[np.ndarray, np.ndarray]:
    rx         = np.zeros(N_SAMPLES)
    rx_complex = np.zeros(N_SAMPLES, dtype=complex)

    for rec in detections:
        tau   = 2.0 * rec.range_m / C
        n_del = int(tau * wp.fs)
        if n_del >= N_SAMPLES:
            continue

        a = 10.0 ** ((rec.rx_power_dbm - wp.P_tx_dbm) / 20.0)
        phi = (2.0 * np.pi * F_C * tau) % (2.0 * np.pi)

        tgt = targets[rec.target_idx]
        if rec.range_m > 0 and tgt.velocity > 0:
            vx  = tgt.velocity * np.cos(tgt.theta)
            vy  = tgt.velocity * np.sin(tgt.theta)
            v_r = (vx * tgt.x + vy * tgt.y) / rec.range_m
        else:
            v_r = 0.0

        f_d = doppler_frequency(v_r, F_C, C)

        echo_real, echo_cplx = build_rx_echo(
            t=wp.t, n_samples=N_SAMPLES, n_pulse=wp.n_p,
            chirp_rate=wp.k, amplitude=a, delay_samples=n_del,
            carrier_phase=phi, doppler_hz=f_d, tau=tau,
        )
        rx         += echo_real
        rx_complex += echo_cplx

    return rx, rx_complex

def add_awgn_stage(rx: np.ndarray, rx_complex: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    peak = float(np.max(np.abs(rx)))
    rx_norm = (rx / peak * 0.88) if peak > 1e-30 else rx.copy()

    peak_cplx       = float(np.max(np.abs(rx_complex)))
    rx_complex_norm = (rx_complex / peak_cplx * 0.88) if peak_cplx > 1e-30 else rx_complex.copy()

    rx_noisy = apply_awgn(rx_norm, snr_db)
    return rx_noisy, rx_complex_norm

def apply_clutter(
    rx: np.ndarray,
    rx_complex: np.ndarray,
    clutter_obj: Optional[Clutter],
    active_regional: Optional[list],
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    rx         = rx.copy()
    rx_complex = rx_complex.copy()

    if active_regional:
        for rc, r_near, r_far in active_regional:
            n_near = int(2.0 * r_near / C * fs)
            n_far  = int(2.0 * r_far  / C * fs)
            n_near = max(0, min(n_near, N_SAMPLES - 1))
            n_far  = max(0, min(n_far,  N_SAMPLES - 1))

            n_span = n_far - n_near
            if n_span <= 0:
                continue

            clutter_samples          = rc.generate_samples(n_span)
            rx[n_near:n_far]         += np.real(clutter_samples)
            rx_complex[n_near:n_far] += clutter_samples

    if clutter_obj is not None:
        c_noise     = clutter_obj.generate()
        rx_complex += c_noise
        rx         += np.real(c_noise)

    return rx, rx_complex

def matched_filter_stage(
    rx_noisy: np.ndarray,
    rx_complex: np.ndarray,
    tx_pulse: np.ndarray,
    n_pulse: int,
) -> tuple[np.ndarray, np.ndarray]:
    return matched_filter(rx_noisy, rx_complex, tx_pulse, n_pulse)

def frontend_stage(
    detections: list[DetectionRecord],
    targets: list,
    wp: WaveformParams,
    clutter_obj: Optional[Clutter],
    active_regional: Optional[list],
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rx, rx_complex = build_rx_signal(detections, wp, targets)
    rx_c, rx_complex_c = apply_clutter(rx, rx_complex, clutter_obj, active_regional, wp.fs)
    rx_noisy, rx_complex_norm = add_awgn_stage(rx_c, rx_complex_c, snr_db)
    comp_disp, comp_complex = matched_filter_stage(rx_noisy, rx_complex_norm, wp.tx[:wp.n_p], wp.n_p)
    return rx_noisy, comp_disp, comp_complex

def mti_stage(mf_disp: np.ndarray, mti: MTI) -> np.ndarray:
    return mti.process(mf_disp)

def integration_stage(mti_out: np.ndarray, mf_complex: np.ndarray, integrator: PulseIntegrator) -> np.ndarray:
    return integrator.process(mti_out, mf_complex)

def cfar_stage(
    integrated: np.ndarray,
    fs: float,
    n_guard: int = N_GUARD,
    n_train: int = N_TRAIN,
    alpha: float = K_CFAR,
    min_cfar_abs: float = MIN_CFAR_ABS,
) -> tuple[np.ndarray, np.ndarray]:
    cfar_thresh      = ca_cfar(integrated, n_guard, n_train, alpha)
    effective_thresh = np.maximum(cfar_thresh, min_cfar_abs)

    binary   = (integrated > effective_thresh).astype(float) * integrated
    min_dist = max(1, int(0.04 * fs))
    peaks, _ = scipy.signal.find_peaks(binary, distance=min_dist)

    return peaks, effective_thresh

def estimated_ppi_stage(
    peaks: np.ndarray,
    azimuth_deg: float,
    t: np.ndarray,
    r_max: float,
    n_pulse: int,
) -> list[tuple[float, float, float]]:
    az_rad       = math.radians(azimuth_deg)
    r_min_blind  = r_max * 0.07
    mf_bias      = n_pulse - 1
    detections   = []

    for p in peaks:
        p_corrected = max(0, p - mf_bias)
        range_est   = C * t[p_corrected] / 2.0

        if not (r_min_blind < range_est < r_max):
            continue

        x = range_est * math.cos(az_rad)
        y = range_est * math.sin(az_rad)
        detections.append((x, y, range_est))

    return detections


class PipelineFrontendWidget(QtWidgets.QSplitter):
    r"""
    Painel de visualização da banda base gerada pelo Pipeline.
    """
    def __init__(self, pipeline: RadarPipeline):
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

        # Plot TX
        self._tx_plot = self._glw.addPlot(row=0, col=0)
        self._tx_plot.setLabel('left', 'TX  Pulse')
        self._tx_plot.getAxis('left').setWidth(65)
        self._tx_plot.showGrid(x=True, y=True, alpha=0.22)
        self._tx_plot.setYRange(-1.2, 1.2)
        self._tx_plot.setMouseEnabled(x=False, y=False)
        self._tx_curve = self._tx_plot.plot(
            self.wp.t_us, self.wp.tx, pen=pg.mkPen((0, 200, 255), width=1),
        )
        self._tx_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen((0, 80, 100), width=1, style=QtCore.Qt.DotLine)))
        self._tx_plot.addItem(pg.InfiniteLine(pos=rx_start_us, angle=90, pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine)))

        # Plot RX
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
        self._rx_plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen((80, 50, 0), width=1, style=QtCore.Qt.DotLine)))
        self._rx_plot.addItem(pg.InfiniteLine(pos=rx_start_us, angle=90, pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine)))

        # Plot MF
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
        self._mf_plot.addItem(pg.InfiniteLine(pos=rx_start_us, angle=90, pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine)))

    def update_plot(self, rx_noisy: np.ndarray, comp_disp: np.ndarray, azimuth_deg: float) -> None:
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
        T_us    = self.wp.T_P   * 1e6
        PRI_us  = self.wp.T_PRI * 1e6
        B_MHz   = B / 1e6
        ppi     = self.pipeline.ppi
        r_min   = ppi.r_max / 7.0 if ppi else 0.0
        bw      = ppi.radar.beamwidth if (ppi and ppi.radar) else 0.0
        c_time  = ppi.elapsed_time if ppi else 0.0
        t_total = ppi.t if ppi else 0.0
        r_max   = ppi.r_max if ppi else 0.0
        c_str        = type(self.pipeline.clutter).__name__.replace("Clutter", "") if self.pipeline.clutter else "None"
        int_mode_str = "Coherent" if self.pipeline.config.get('integrator_type') == "coherent" else "Non-Coherent"

        html = get_pulse_header_html(
            PRI_us=PRI_us, T_us=T_us, F_C_GHz=F_C / 1e9, B_MHz=B_MHz,
            snr_db=self.pipeline.snr_db, c_str=c_str, r_min=r_min, r_max=r_max,
            bw=bw, int_mode_str=int_mode_str, c_time=c_time, t_total=t_total,
        )
        self._header_label.setText(html)

