r"""
pipeline.py — Funções puras e orquestração do pipeline de radar.

Este módulo centraliza:
1. As funções matemáticas puras de cada estágio do pipeline.
2. A classe `RadarPipeline` que instancia o PPI e orquestra a simulação numérico-temporal.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import scipy.signal

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
    C, F_C, N_SAMPLES, DEFAULT_SNR_DB,
    N_GUARD, N_TRAIN, K_CFAR, MIN_CFAR_ABS,
)
from radarutils.simulator.detection import DetectionRecord
from radarutils.simulator.pulse import WaveformParams, waveform_params_from_rmax
from radarutils.simulator.ppi import PPI

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

