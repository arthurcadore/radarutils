import math
import sys
import numpy as np
import pyqtgraph as pg
import imageio

from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui


from radarutils.simulator.mti import MTI
from radarutils.simulator.cfar import ca_cfar
from radarutils.simulator.integrator import PulseIntegrator
from radarutils.simulator.ppi import PPI, PPIViewer, PPIEstimatedTracker, PPIEstimatedViewer
from radarutils.simulator.constants import (
    C, F_C, B, N_SAMPLES,
    DEFAULT_SNR_DB, N_GUARD, N_TRAIN, K_CFAR, N_INT,
    MIN_CFAR_ABS, MIN_Y_MTI, MIN_Y_INT, MIN_Y_CFAR, MAX_MATCH_DIST
)

from radarutils.simulator.detection import DetectionRecord

def prepare_output_file(file_name: str = "simulation.mp4") -> str:
    """Garante que o diretório de saída existe e retorna o caminho completo."""
    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / "../../../data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / file_name

    if output_path.exists():
        output_path.unlink()

    return str(output_path)

class DetectionPlot(pg.PlotWidget):
    """Gráfico de dispersão Range vs. Tempo das detecções do radar."""

    def __init__(self, ppi: PPI = None):
        super().__init__()
        self.ppi = ppi  # usado para normalizar deg_error no colormap
        self.setBackground('k')
        self.setLabel('left', 'Range', units='m')
        self.getAxis('left').setWidth(65)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setYRange(0, 1000)



        self.plot_data = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None))
        self.addItem(self.plot_data)

        self.available_symbols = ['o', 's', 't', 'd', '+', 'x', 'star', 'p', 'h']

        # Manual Jet-like colormap (Blue → Cyan → Green → Yellow → Red)
        # Vermelho = centro do feixe (deg_error ≈ 0), Azul = borda (|deg_error| ≈ bw/2)
        self.colormap = pg.ColorMap(
            pos=[0.0, 0.25, 0.5, 0.75, 1.0],
            color=[
                (0, 0, 255, 255),
                (0, 255, 255, 255),
                (0, 255, 0, 255),
                (255, 255, 0, 255),
                (255, 0, 0, 255),
            ]
        )

        self.times: list[float] = []
        self.ranges: list[float] = []
        self.brushes: list = []
        self.symbols: list[str] = []
        self._current_time: float = 0.0

    def _norm_deg_error(self, deg_error: float) -> float:
        """Normaliza deg_error para [0, 1] usando beamwidth do radar.
        0 = centro do feixe (vermelho), 1 = borda (azul).
        """
        bw_half = 5.0  # fallback: beamwidth 10°
        if self.ppi and self.ppi.radar:
            bw_half = max(self.ppi.radar.beamwidth / 2, 0.001)
        return min(abs(deg_error) / bw_half, 1.0)

    def add_detections(self, t: float, detection_list: list[DetectionRecord]):
        """
        Adiciona novas detecções ao histórico visual do gráfico.

        Args:
            t:               Tempo atual da simulação (s).
            detection_list:  Lista de DetectionRecord do passo atual.
        """
        self._current_time = t

        for rec in detection_list:
            self.times.append(rec.time)
            self.ranges.append(rec.range_m)

            sym = self.available_symbols[rec.target_idx % len(self.available_symbols)]
            self.symbols.append(sym)

            # 1 - norm: vermelho = centro (0°), azul = borda (bw/2°)
            norm = self._norm_deg_error(rec.deg_error)
            color = self.colormap.mapToQColor(1.0 - norm)
            color.setAlpha(180)
            self.brushes.append(pg.mkBrush(color))

        if self.times:
            self.plot_data.setData(
                x=self.times,
                y=self.ranges,
                brush=self.brushes,
                symbol=self.symbols,
            )

        # Janela deslizante de 15 s
        window_size = 15
        current_time = self._current_time
        if current_time > window_size:
            self.setXRange(current_time - window_size, current_time)
        else:
            self.setXRange(0, window_size)


class AmplitudePlot(pg.PlotWidget):
    """Gráfico de dispersão Amplitude (dBm) vs. Tempo das detecções do radar."""

    WINDOW_SIZE = 15  # segundos

    def __init__(self, ppi: PPI = None):
        super().__init__()
        self.ppi = ppi
        self.setBackground('k')
        self.setLabel('left', 'Amplitude', units='dBm')
        self.getAxis('left').setWidth(65)
        self.showGrid(x=True, y=True, alpha=0.3)

        # Mesma paleta Jet da DetectionPlot
        self.colormap = pg.ColorMap(
            pos=[0.0, 0.25, 0.5, 0.75, 1.0],
            color=[
                (0, 0, 255, 255),
                (0, 255, 255, 255),
                (0, 255, 0, 255),
                (255, 255, 0, 255),
                (255, 0, 0, 255),
            ]
        )

        self.plot_data = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None))
        self.addItem(self.plot_data)

        self.available_symbols = ['o', 's', 't', 'd', '+', 'x', 'star', 'p', 'h']

        self.times:        list[float] = []
        self.powers:       list[float] = []
        self.brushes:      list        = []
        self.symbols:      list[str]   = []
        self._current_time: float      = 0.0

    def _norm_deg_error(self, deg_error: float) -> float:
        bw_half = 5.0
        if self.ppi and self.ppi.radar:
            bw_half = max(self.ppi.radar.beamwidth / 2, 0.001)
        return min(abs(deg_error) / bw_half, 1.0)

    def add_detections(self, t: float, detection_list: list[DetectionRecord]):
        """
        Adiciona as potências RX das novas detecções ao histórico visual.

        Sempre avança a janela de tempo (mesmo sem detecções) para manter
        sincronismo com o DetectionPlot.
        """
        self._current_time = t

        for rec in detection_list:
            self.times.append(rec.time)
            self.powers.append(rec.rx_power_dbm)

            sym = self.available_symbols[rec.target_idx % len(self.available_symbols)]
            self.symbols.append(sym)

            norm  = self._norm_deg_error(rec.deg_error)
            color = self.colormap.mapToQColor(1.0 - norm)
            color.setAlpha(200)
            self.brushes.append(pg.mkBrush(color))

        if self.times:
            self.plot_data.setData(
                x=self.times,
                y=self.powers,
                brush=self.brushes,
                symbol=self.symbols,
            )

        # Janela deslizante — sempre sincroniza pelo tempo atual da simulação
        current_time = self._current_time
        if current_time > self.WINDOW_SIZE:
            x_min = current_time - self.WINDOW_SIZE
            self.setXRange(x_min, current_time)
            visible = [
                p for tt, p in zip(self.times, self.powers)
                if tt >= x_min
            ]
        else:
            self.setXRange(0, self.WINDOW_SIZE)
            visible = self.powers

        if visible:
            y_min = min(visible) - 5
            y_max = max(visible) + 5
            self.setYRange(y_min, y_max)


class PhasePlot(pg.PlotWidget):
    """Gráfico de dispersão Fase (radianos) vs. Tempo das detecções do radar."""

    WINDOW_SIZE = 15  # segundos

    def __init__(self, ppi: PPI = None):
        super().__init__()
        self.ppi = ppi
        self.setBackground('k')
        self.setLabel('bottom', 'Time', units='s')
        self.setLabel('left', 'Phase', units='rad')
        self.getAxis('left').setWidth(65)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setYRange(-np.pi * 1.1, np.pi * 1.1)

        # Mesma paleta Jet da DetectionPlot
        self.colormap = pg.ColorMap(
            pos=[0.0, 0.25, 0.5, 0.75, 1.0],
            color=[
                (0, 0, 255, 255),
                (0, 255, 255, 255),
                (0, 255, 0, 255),
                (255, 255, 0, 255),
                (255, 0, 0, 255),
            ]
        )

        self.plot_data = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None))
        self.addItem(self.plot_data)

        self.available_symbols = ['o', 's', 't', 'd', '+', 'x', 'star', 'p', 'h']

        self.times:        list[float] = []
        self.phases:       list[float] = []
        self.brushes:      list        = []
        self.symbols:      list[str]   = []
        self._current_time: float      = 0.0

    def _norm_deg_error(self, deg_error: float) -> float:
        bw_half = 5.0
        if self.ppi and self.ppi.radar:
            bw_half = max(self.ppi.radar.beamwidth / 2, 0.001)
        return min(abs(deg_error) / bw_half, 1.0)

    def add_detections(self, t: float, detection_list: list[DetectionRecord]):
        self._current_time = t

        for rec in detection_list:
            self.times.append(rec.time)
            
            # Cálculo da fase teórica RX (retardo ida e volta)
            # phi = 4 * pi * R / lambda
            # Usando banda X genérica (lambda = 0.03m / freq 10GHz)
            lam_m = 0.03
            phi = (4.0 * np.pi * rec.range_m / lam_m) % (2.0 * np.pi)
            if phi > np.pi:
                phi -= 2.0 * np.pi
                
            self.phases.append(phi)

            sym = self.available_symbols[rec.target_idx % len(self.available_symbols)]
            self.symbols.append(sym)

            norm  = self._norm_deg_error(rec.deg_error)
            color = self.colormap.mapToQColor(1.0 - norm)
            color.setAlpha(200)
            self.brushes.append(pg.mkBrush(color))

        if self.times:
            self.plot_data.setData(
                x=self.times,
                y=self.phases,
                brush=self.brushes,
                symbol=self.symbols,
            )

        # Janela deslizante — sempre sincroniza pelo tempo atual da simulação
        current_time = self._current_time
        if current_time > self.WINDOW_SIZE:
            x_min = current_time - self.WINDOW_SIZE
            self.setXRange(x_min, current_time)
        else:
            self.setXRange(0, self.WINDOW_SIZE)


class PulseWidget(QtWidgets.QSplitter):
    """
    Visualização de pulso radar em banda base (chirp LFM) com AWGN.

    Sub-plots verticais:
      TX  (ciano)  : chirp LFM transmitido, normalizado a ±1.
      RX  (laranja): ecos compostos + AWGN com SNR configurável.
                     Efeitos: atraso 2R/c, amplitude, fase portadora, Doppler.
      FFT (verde)  : FFT do sinal RX da escuta.
      MF  (rosa)   : Filtro Casado (Pulse Compression) no tempo.
    """


    def __init__(self, ppi: PPI = None, snr_db: float = None, coherent_integration: bool = False, clutter_type: str = "None", normalize_plots: bool = True):
        super().__init__(QtCore.Qt.Vertical)
        self.ppi        = ppi
        self.snr_db     = snr_db if snr_db is not None else DEFAULT_SNR_DB
        self.coherent_integration = coherent_integration
        self.clutter_type = clutter_type if clutter_type else "None"
        self.normalize_plots = normalize_plots
        
        self.setStyleSheet("QSplitter::handle { background-color: #555555; height: 3px; }")

        self.header_label = QtWidgets.QLabel()
        self.header_label.setStyleSheet("background-color: black;")
        self.header_label.setAlignment(QtCore.Qt.AlignCenter)
        
        self.bottom_glw = pg.GraphicsLayoutWidget()
        self.bottom_glw.setBackground('k')
        self.bottom_glw.ci.layout.setSpacing(12)  # Adiciona espaçamento entre os plots

        self.addWidget(self.header_label)
        self.addWidget(self.bottom_glw)

        # 
        # Parâmetros de forma de onda
        # 
        r_max       = ppi.r_max if ppi else 1000.0
        self.T_PRI  = 2.0 * r_max / C
        self.T_P    = self.T_PRI / 7.0
        self.fs     = N_SAMPLES / self.T_PRI
        self.t      = np.linspace(0, self.T_PRI, N_SAMPLES, endpoint=False)
        self.t_us   = self.t * 1e6
        self.k      = B / self.T_P
        self.n_p    = int(self.T_P * self.fs)       # amostras do pulso TX
        self.N_rx   = N_SAMPLES - self.n_p             # amostras do período de escuta

        if ppi and ppi.radar:
            self.P_tx_dbm = 10.0 * np.log10(ppi.radar.pt * 1e3)
        else:
            self.P_tx_dbm = 60.0

        # Sinal TX: cos(π·k·t²) nos primeiros n_p samples
        self._tx            = np.zeros(N_SAMPLES)
        t_chirp             = self.t[:self.n_p]
        self._tx[:self.n_p] = np.cos(np.pi * self.k * t_chirp ** 2)

        # Eixo de frequência para FFT do período de escuta (rfft → [0, fs/2])
        self._f_fft_mhz = np.fft.rfftfreq(self.N_rx, d=1.0 / self.fs) / 1e6
        self._win_hann  = np.hanning(self.N_rx)

        rx_start_us = self.T_P * 1e6   # posição do marcador de início de RX

        # 
        # Label de Informações Globais (Topo) - Já instanciado como QLabel
        #

        # ── Plot 0 — TX Chirp ───────────────────────────────────────────────
        self.tx_plot = self.bottom_glw.addPlot(row=0, col=0)
        self.tx_plot.setLabel('left', 'TX  Pulse')
        self.tx_plot.getAxis('left').setWidth(65)
        self.tx_plot.showGrid(x=True, y=True, alpha=0.22)
        self.tx_plot.setYRange(-1.2, 1.2)
        self.tx_plot.setMouseEnabled(x=False, y=False)
        self.tx_curve = self.tx_plot.plot(
            self.t_us, self._tx, pen=pg.mkPen((0, 200, 255), width=1),
        )
        self.tx_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen((0, 80, 100), width=1, style=QtCore.Qt.DotLine),
        ))
        # Linha vertical marcando início do período de escuta
        self.tx_plot.addItem(pg.InfiniteLine(
            pos=rx_start_us, angle=90,
            pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine),
        ))

        # ── Plot 1 — RX Baseband com AWGN ────────────────────────────────────
        self.rx_plot = self.bottom_glw.addPlot(row=1, col=0)
        self.rx_plot.setLabel('left', 'RX Baseband')
        self.rx_plot.getAxis('left').setWidth(65)
        self.rx_plot.showGrid(x=True, y=True, alpha=0.22)
        self.rx_plot.setYRange(-1.2, 1.2)   # Y idêntico ao TX
        self.rx_plot.setMouseEnabled(x=False, y=False)
        self.rx_plot.setXLink(self.tx_plot)
        self.rx_curve = self.rx_plot.plot(
            self.t_us, np.zeros(N_SAMPLES),
            pen=pg.mkPen((255, 140, 0), width=1),
        )
        self.rx_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen((80, 50, 0), width=1, style=QtCore.Qt.DotLine),
        ))
        self.rx_plot.addItem(pg.InfiniteLine(
            pos=rx_start_us, angle=90,
            pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine),
        ))

        # ── Plot 2 — Matched Filter (Pulse Compression) ──────────────────────
        self.mf_plot = self.bottom_glw.addPlot(row=2, col=0)
        self.mf_plot.setLabel('left', 'Matched Filter Out')
        self.mf_plot.getAxis('left').setWidth(65)
        self.mf_plot.setLabel('bottom', 'Tempo (µs)')
        self.mf_plot.showGrid(x=True, y=True, alpha=0.22)
        self.mf_plot.setYRange(0, 100) # Inicial configurável
        self.mf_plot.setMouseEnabled(x=False, y=False)
        self.mf_plot.setXLink(self.tx_plot)
        self.mf_curve = self.mf_plot.plot(
            self.t_us, np.zeros(N_SAMPLES),
            pen=pg.mkPen((255, 0, 255), width=1),
        )
        self.mf_plot.addItem(pg.InfiniteLine(
            pos=rx_start_us, angle=90,
            pen=pg.mkPen((180, 80, 0), width=1, style=QtCore.Qt.DashLine),
        ))

        # ── Proporções ──────────────────────────────────────────────────────
        self.bottom_glw.ci.layout.setRowStretchFactor(0, 1)
        self.bottom_glw.ci.layout.setRowStretchFactor(1, 1)
        self.bottom_glw.ci.layout.setRowStretchFactor(2, 1.5)
        
        self.setSizes([200, 800])

    # 

    def update_pulse(self, detections: list) -> None:
        """
        Reconstrói RX, adiciona AWGN e delega FFT para o fft_widget se existir.
        """
        rx         = np.zeros(N_SAMPLES)          # sinal real (para plot)
        rx_complex = np.zeros(N_SAMPLES, dtype=complex)  # sinal complexo (para integ. coerente)
        status_parts : list[str] = []

        if detections and self.ppi and self.ppi.radar:
            self.P_tx_dbm = 10.0 * np.log10(self.ppi.radar.pt * 1e3)

            for rec in detections:
                tau   = 2.0 * rec.range_m / C
                n_del = int(tau * self.fs)
                if n_del >= N_SAMPLES:
                    continue

                a   = 10.0 ** ((rec.rx_power_dbm - self.P_tx_dbm) / 20.0)
                phi = (2.0 * np.pi * F_C * tau) % (2.0 * np.pi)

                tgt = self.ppi.targets[rec.target_idx]
                if rec.range_m > 0 and tgt.velocity > 0:
                    vx  = tgt.velocity * np.cos(tgt.theta)
                    vy  = tgt.velocity * np.sin(tgt.theta)
                    v_r = (vx * tgt.x + vy * tgt.y) / rec.range_m
                else:
                    v_r = 0.0
                f_d = 2.0 * v_r * F_C / C

                end      = min(n_del + self.n_p, N_SAMPLES)
                n_actual = end - n_del
                if n_actual <= 0:
                    continue
                t_local = self.t[n_del:end] - tau
                chirp_phase = np.pi * self.k * t_local ** 2 + 2.0 * np.pi * f_d * t_local + phi
                # Sinal real (para display)
                rx[n_del:end] += a * np.cos(chirp_phase)
                # Sinal complexo IQ (para integração coerente)
                rx_complex[n_del:end] += a * np.exp(1j * chirp_phase)

                status_parts.append(
                    f"T{rec.target_idx}: τ={tau*1e6:.2f}µs  "
                    f"{rec.rx_power_dbm:.1f}dBm  "
                    f"fd={f_d:+.0f}Hz  "
                    f"vr={v_r:+.1f}m/s"
                )

        # ── Clutter (Rayleigh Ambiental) ──────────────────────────────────
        if self.clutter_type.lower() == 'rayleigh':
            # Clutter ambiental (solo/chuva) pode ser modelado como ruído branco Gaussiano
            # em banda base, o envelope final após o filtro casado segue Rayleigh.
            # Aqui configurado para uma intensidade moderada relativa ao alvo.
            clutter_amp = 0.000001
            c_noise = clutter_amp * (np.random.randn(N_SAMPLES) + 1j * np.random.randn(N_SAMPLES)) / np.sqrt(2)
            rx_complex += c_noise
            rx += np.real(c_noise)

        # ── Atualização do Header ─────────────────────────────────────────
        T_us   = self.T_P   * 1e6
        PRI_us = self.T_PRI * 1e6
        B_MHz  = B / 1e6
        r_min  = self.ppi.r_max / 7.0 if self.ppi else 0.0
        bw     = self.ppi.radar.beamwidth if (self.ppi and self.ppi.radar) else 0.0
        c_time = self.ppi.elapsed_time if self.ppi else 0.0
        t_total = self.ppi.t if self.ppi else 0.0
        n_samples = N_SAMPLES
        r_max = self.ppi.r_max if self.ppi else 0.0
        c_str = self.clutter_type.capitalize() if self.clutter_type and self.clutter_type.lower() != 'none' else "None"

        int_mode_str = "Coherent" if self.coherent_integration else "Non-Coherent"
        header_html = (
            f'<div align="center">'
            f'<table align="center" cellpadding="3" cellspacing="0" style="font-family: Consolas; font-size:10pt; color: #DDDDDD;">'
            f'<tr><td colspan="8" align="center" style="font-size:14pt; color: #00C8FF; font-weight: bold; padding-bottom: 8px;">RADAR SIMULATION PARAMETERS</td></tr>'
            f'<tr>'
            f'<td align="right" style="color:#88CCFF; padding-right:4px;">PRI:</td><td align="left" style="font-weight: bold; padding-right:15px;">{PRI_us:.2f} µs</td>'
            f'<td align="right" style="color:#88CCFF; padding-right:4px;">Tp:</td><td align="left" style="font-weight: bold; padding-right:15px;">{T_us:.2f} µs</td>'
            f'<td align="right" style="color:#88CCFF; padding-right:4px;">Fc:</td><td align="left" style="font-weight: bold; padding-right:15px;">{F_C/1e9:.0f} GHz</td>'
            f'<td align="right" style="color:#88CCFF; padding-right:4px;">BW:</td><td align="left" style="font-weight: bold;">{B_MHz:.0f} MHz</td>'
            f'</tr>'
            f'<tr>'
            f'<td align="right" style="color:#88CCFF;">SNR:</td><td align="left" style="font-weight: bold;">{self.snr_db:.0f} dB</td>'
            f'<td align="right" style="color:#88CCFF;">Clutter:</td><td align="left" style="font-weight: bold;">{c_str}</td>'
            f'<td align="right" style="color:#88CCFF;">R_min:</td><td align="left" style="font-weight: bold;">{r_min:.1f} m</td>'
            f'<td align="right" style="color:#88CCFF;">R_max:</td><td align="left" style="font-weight: bold;">{r_max:.1f} m</td>'
            f'</tr>'
            f'<tr>'
            f'<td align="right" style="color:#88CCFF;">Beam:</td><td align="left" style="font-weight: bold;">{bw:.1f}°</td>'
            f'<td align="right" style="color:#88CCFF;">Int. Mode:</td><td align="left" style="font-weight: bold;">{int_mode_str}</td>'
            f'<td align="right" style="color:#88CCFF;">T_now:</td><td align="left" style="font-weight: bold;">{c_time:.2f} s</td>'
            f'<td align="right" style="color:#88CCFF; padding-right:4px;">T_max:</td><td align="left" style="font-weight: bold;">{t_total:.1f} s</td>'
            f'</tr>'
            f'</table>'
            f'</div>'
        )
        self.header_label.setText(header_html)

        # ── Normalização ─────────────────────────────────────────────────
        peak = float(np.max(np.abs(rx)))
        rx_norm = (rx / peak * 0.88) if peak > 1e-30 else rx.copy()

        # Normaliza rx_complex pela mesma escala do sinal real
        # (sem normalização o sinal complexo teria amplitude ~1e-8, ficando
        #  abaixo do threshold mínimo do CFAR e causando "sinal invisível")
        peak_cplx = float(np.max(np.abs(rx_complex)))
        rx_complex_norm = (rx_complex / peak_cplx * 0.88) if peak_cplx > 1e-30 else rx_complex.copy()

        # ── AWGN ─────────────────────────────────────────────────────────
        # σ calibrado para SNR desejado relativo ao pico normalizado
        ref = float(np.max(np.abs(rx_norm))) if peak > 1e-30 else 0.88
        sigma    = ref / (10.0 ** (self.snr_db / 20.0))
        rx_noisy = rx_norm + sigma * np.random.randn(N_SAMPLES)

        # ── Plot tempo ───────────────────────────────────────────────────
        # Y-range é fixo (definido no __init__) — sem auto-scale pelo ruído
        self.rx_curve.setData(self.t_us, rx_noisy)

        # ── Matched Filter Complexo (para integração coerente) ───────────────
        import scipy.signal
        tx_pulse = self._tx[:self.n_p]

        # Envelope real (display e integração não-coerente)
        compressed      = np.abs(scipy.signal.correlate(rx_noisy, tx_pulse, mode='same'))
        comp_disp       = np.roll(compressed, self.n_p // 2)
        comp_disp[:self.n_p // 2] = 0.0

        # Filtro casado complexo (IQ normalizado — para integração coerente)
        compressed_cplx = scipy.signal.correlate(rx_complex_norm, tx_pulse.astype(complex), mode='same')
        comp_complex    = np.roll(compressed_cplx, self.n_p // 2)
        comp_complex[:self.n_p // 2] = 0.0

        self.mf_curve.setData(self.t_us, comp_disp)
        peak_comp = float(np.max(comp_disp))
        
        if self.normalize_plots:
            # Apenas normaliza o display (0-1); comp_disp continua com valores originais
            mf_disp_norm = (comp_disp / peak_comp) if peak_comp > 1e-30 else comp_disp
            self.mf_curve.setData(self.t_us, mf_disp_norm)
            self.mf_plot.setYRange(0, 1.05)
        else:
            self.mf_curve.setData(self.t_us, comp_disp)
            self.mf_plot.setYRange(0, max(peak_comp + 20, 100))

        az = self.ppi.radar.theta if (self.ppi and self.ppi.radar) else 0.0
        return {'comp_disp': comp_disp, 'comp_complex': comp_complex, 'azimuth_deg': az}


class ProcessingWidget(QtWidgets.QSplitter):
    """
    Pipeline de processamento de sinal radar (Coluna 3).

    Blocos verticais seqüenciais:
      MTI  (amarelo) : Cancelador delay-line (1 atraso) — suprime ecos fixos.
      CFAR (ciano)   : CA-CFAR adaptativo — threshold por célula.
      INT  (branco)  : Integrador não-coerente de N_INT PRIs.
      PPI  (azul)    : PPI estimado com histórico de hits.
    """

    def __init__(self, ppi: PPI, pulse_widget, coherent_integration: bool = False, normalize_plots: bool = True):
        super().__init__(QtCore.Qt.Vertical)
        self.ppi = ppi
        self.pw  = pulse_widget
        self.coherent_integration = coherent_integration
        self.normalize_plots = normalize_plots
        
        self.setStyleSheet("QSplitter::handle { background-color: #555555; height: 3px; }")

        self.top_glw = pg.GraphicsLayoutWidget()
        self.top_glw.setBackground('k')
        self.top_glw.ci.layout.setSpacing(8)

        self.addWidget(self.top_glw)

        # Inicializa modulos
        self.mti = MTI(N_SAMPLES)
        self.integrator = PulseIntegrator(N_INT, coherent_integration)
        self.ppi_tracker = PPIEstimatedTracker(h_hits=4096, max_match_dist=MAX_MATCH_DIST)
        self.alpha = K_CFAR

        t_us = pulse_widget.t_us
        r_max = ppi.r_max

        self._mf_prev_complex = np.zeros(N_SAMPLES, dtype=complex)  # para estimativa de desvio doppler

        # ── Plot 0 — MTI ───────────────────────────────────────────────
        self.mti_plot = self.top_glw.addPlot(row=0, col=0)
        self.mti_plot.setLabel('left', 'MTI')
        self.mti_plot.getAxis('left').setWidth(65)
        self.mti_plot.showGrid(x=True, y=True, alpha=0.22)
        self.mti_plot.setYRange(0, 10)
        self.mti_plot.setMouseEnabled(x=False, y=False)
        self.mti_curve = self.mti_plot.plot(
            t_us, np.zeros(N_SAMPLES), pen=pg.mkPen((255, 255, 0), width=1)
        )

        # ── Plot 1 — Integrador Não-Coerente ──────────────────────────
        self.int_plot = self.top_glw.addPlot(row=1, col=0)
        self.int_plot.setLabel('left', 'Pulse Integrator')
        self.int_plot.getAxis('left').setWidth(65)
        self.int_plot.showGrid(x=True, y=True, alpha=0.22)
        self.int_plot.setYRange(0, 10)
        self.int_plot.setMouseEnabled(x=False, y=False)
        self.int_plot.setXLink(self.mti_plot)
        self.int_curve = self.int_plot.plot(
            t_us, np.zeros(N_SAMPLES), pen=pg.mkPen((210, 210, 210), width=1)
        )
        self.int_legend = self.int_plot.addLegend(colCount=1)
        self.int_legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        self.int_legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(0, 0))
        _mode_str = "Coherent" if self.coherent_integration else "Non-Coherent"
        _dummy_int = pg.PlotDataItem(pen=pg.mkPen((210, 210, 210), width=1))
        self.int_legend.addItem(_dummy_int, f"Mode: {_mode_str}")

        # ── Plot 2 — CA-CFAR ──────────────────────────────────────────
        self.cfar_plot = self.top_glw.addPlot(row=2, col=0)
        self.cfar_plot.setLabel('left', 'CA-CFAR')
        self.cfar_plot.getAxis('left').setWidth(65)
        self.cfar_plot.setLabel('bottom', 'Tempo (µs)')
        self.cfar_plot.showGrid(x=True, y=True, alpha=0.22)
        self.cfar_plot.setYRange(0, 10)
        self.cfar_plot.setMouseEnabled(x=False, y=False)
        self.cfar_plot.setXLink(self.mti_plot)
        self.cfar_legend = self.cfar_plot.addLegend(colCount=2)
        self.cfar_legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        self.cfar_legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(0, 0))
        
        self.cfar_sig_curve = self.cfar_plot.plot(
            t_us, np.zeros(N_SAMPLES), pen=pg.mkPen((0, 190, 255), width=1), name="Sinal (Rx)"
        )
        self.cfar_thr_curve = self.cfar_plot.plot(
            t_us, np.zeros(N_SAMPLES),
            pen=pg.mkPen((255, 80, 80), width=1, style=QtCore.Qt.DashLine), name="Threshold CFAR"
        )
        self.cfar_spots = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 220), symbol='o'
        )
        self.cfar_plot.addItem(self.cfar_spots)

        # ── Plot 3 — PPI Estimado Viewer ────────────────────────────────
        self.ppi_est_viewer = PPIEstimatedViewer(r_max=r_max)
        self.addWidget(self.ppi_est_viewer)
        
        # ── Proporções ─────────────────────────────────────────────────
        self.top_glw.ci.layout.setRowStretchFactor(0, 1)
        self.top_glw.ci.layout.setRowStretchFactor(1, 1)
        self.top_glw.ci.layout.setRowStretchFactor(2, 1)        
        self.setSizes([1000, 1000])

    def update(self, pulse_data: dict) -> None:
        import scipy.signal as ss

        comp_disp   = pulse_data.get('comp_disp', np.zeros(N_SAMPLES))
        comp_complex = pulse_data.get('comp_complex', np.zeros(N_SAMPLES, dtype=complex))
        azimuth_deg = pulse_data.get('azimuth_deg', 0.0)
        t_us        = self.pw.t_us
        fs          = (N_SAMPLES / self.pw.T_PRI)
        r_max       = self.ppi.r_max

        # 1. Processamento MTI
        mti_out = self.mti.process(comp_disp)
        peak_mti = float(np.max(mti_out)) if mti_out.any() else 0.0
        
        if self.normalize_plots:
            mti_disp = (mti_out / peak_mti) if peak_mti > 1e-30 else mti_out
            self.mti_curve.setData(t_us, mti_disp)
            self.mti_plot.setYRange(0, 1.05)
        else:
            self.mti_curve.setData(t_us, mti_out)
            self.mti_plot.setYRange(0, max(peak_mti * 1.15, MIN_Y_MTI))

        # 2. Integração
        min_dist = max(1, int(0.04 * fs))
        integrated = self.integrator.process(mti_out, comp_complex if self.coherent_integration else None)
        peak_int = float(np.max(integrated)) if integrated.any() else 0.0
        
        if self.normalize_plots:
            int_disp = (integrated / peak_int) if peak_int > 1e-30 else integrated
            self.int_curve.setData(t_us, int_disp)
            self.int_plot.setYRange(0, 1.05)
        else:
            self.int_curve.setData(t_us, integrated)
            self.int_plot.setYRange(0, max(peak_int * 1.15, MIN_Y_INT))

        # 3. CA-CFAR
        cfar_thresh = ca_cfar(integrated, N_GUARD, N_TRAIN, self.alpha)
        effective_thresh = np.maximum(cfar_thresh, MIN_CFAR_ABS)
        
        cfar_norm_factor = max(peak_int, float(np.max(effective_thresh)))
        if self.normalize_plots:
            if cfar_norm_factor > 1e-30:
                cfar_sig_disp = integrated      / cfar_norm_factor
                cfar_thr_disp = effective_thresh / cfar_norm_factor
            else:
                cfar_sig_disp = integrated
                cfar_thr_disp = effective_thresh
            self.cfar_sig_curve.setData(t_us, cfar_sig_disp)
            self.cfar_thr_curve.setData(t_us, cfar_thr_disp)
            self.cfar_plot.setYRange(0, 1.05)
        else:
            self.cfar_sig_curve.setData(t_us, integrated)
            self.cfar_thr_curve.setData(t_us, effective_thresh)
            y_top = cfar_norm_factor * 1.2
            self.cfar_plot.setYRange(0, max(y_top, MIN_Y_CFAR))

        # Peak detection
        binary = (integrated > effective_thresh).astype(float) * integrated
        peaks_cfar, _ = ss.find_peaks(binary, distance=min_dist)
        
        if self.normalize_plots:
            spots_y = (integrated[peaks_cfar] / cfar_norm_factor) if cfar_norm_factor > 1e-30 else integrated[peaks_cfar]
            self.cfar_spots.setData(t_us[peaks_cfar], spots_y)
        else:
            self.cfar_spots.setData(t_us[peaks_cfar], integrated[peaks_cfar])

        # 4. PPI Estimado Updates
        direction = -1 if (self.ppi and self.ppi.radar and self.ppi.radar.clockwise) else 1
        self.ppi_tracker.update_sweep(azimuth_deg, direction)

        lam = C / 10e9
        delta_phi = np.angle(np.conj(self._mf_prev_complex) * comp_complex)
        vr_map = delta_phi / (2.0 * np.pi * self.pw.T_PRI) * lam / 2.0
        self._mf_prev_complex = comp_complex.copy()

        real_targets = [(tgt.x, tgt.y) for tgt in self.ppi.targets] if self.ppi else []
        az_rad = math.radians(azimuth_deg)

        _mf_range_offset = self.pw.n_p - 1
        new_true_vrs = []
        if len(peaks_cfar) > 0:
            r_min_blind = r_max * 0.07
            for p in peaks_cfar:
                p_corrected = max(0, p - _mf_range_offset)
                range_est = C * self.pw.t[p_corrected] / 2.0
                if not (r_min_blind < range_est < r_max):
                    continue
                det_x = range_est * math.cos(az_rad)
                det_y = range_est * math.sin(az_rad)
                vr_est = float(vr_map[p])

                is_true, vr = self.ppi_tracker.add_detection(det_x, det_y, azimuth_deg, vr_est, real_targets)
                if is_true:
                    new_true_vrs.append(vr)

        if new_true_vrs:
            self.ppi_tracker.last_detected_vrs = new_true_vrs.copy()

        self.ppi_est_viewer.update_view(self.ppi_tracker, az_rad)

class MainWindow(QtWidgets.QMainWindow):
    """
    Janela principal da interface gráfica.

    Recebe um PPI já configurado e executa o loop de atualização via QTimer.
    Não define targets, radar nem parâmetros de simulação.
    """

    def __init__(self, ppi: PPI, show_vectors: bool = True, output_file: str = None,
                 coherent_integration: bool = False, clutter_type: str = "None", normalize_plots: bool = True,
                 max_video_mb: float = None, video_quality: int = 8):
        super().__init__()
        self.setWindowTitle('PPI RADAR SIMULATOR')
        self.resize(2208, 992)

        self.ppi = ppi
        self.output_file = output_file
        self.max_video_mb = max_video_mb
        self.video_writer = None
        self.video_size = (2208, 992)
        self.coherent_integration = coherent_integration

        if self.output_file:
            # Formato ideal para WhatsApp: MP4 com H.264 + yuv420p (e AAC para áudio, se houver)
            self.video_writer = imageio.get_writer(
                self.output_file,
                fps=30,
                codec="libx264",
                audio_codec="aac",
                pixelformat="yuv420p",
                quality=video_quality,
            )

        # Layout principal: 3 colunas
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # ─── COLUNA 1: PPI + Range + Amplitude ───
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.viewer   = PPIViewer(self.ppi, show_vectors=show_vectors)
        self.det_plot = DetectionPlot(ppi=self.ppi)
        self.det_plot.setYRange(0, self.ppi.r_max)
        self.amp_plot = AmplitudePlot(ppi=self.ppi)
        self.phase_plot = PhasePlot(ppi=self.ppi)

        left_layout.addWidget(self.viewer,   stretch=6)
        left_layout.addWidget(self.det_plot, stretch=2)
        left_layout.addWidget(self.amp_plot, stretch=2)
        left_layout.addWidget(self.phase_plot, stretch=2)

        # ─── COLUNA 2: TX / RX / Filtro Casado ───
        mid_panel = QtWidgets.QWidget()
        mid_layout = QtWidgets.QVBoxLayout(mid_panel)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(4)

        self.pulse_widget = PulseWidget(ppi=self.ppi, coherent_integration=coherent_integration, clutter_type=clutter_type, normalize_plots=normalize_plots)
        mid_layout.addWidget(self.pulse_widget, stretch=1)

        # ─── COLUNA 3: MTI / CA-CFAR / Integrador NC / PPI Estimado ───
        proc_panel = QtWidgets.QWidget()
        proc_layout = QtWidgets.QVBoxLayout(proc_panel)
        proc_layout.setContentsMargins(0, 0, 0, 0)
        proc_layout.setSpacing(4)

        self.proc_widget = ProcessingWidget(
            ppi=self.ppi,
            pulse_widget=self.pulse_widget,
            coherent_integration=coherent_integration,
            normalize_plots=normalize_plots,
        )
        proc_layout.addWidget(self.proc_widget, stretch=1)

        # Monta o layout principal
        main_layout.addWidget(left_panel,  stretch=8)
        main_layout.addWidget(mid_panel,   stretch=7)
        main_layout.addWidget(proc_panel,  stretch=7)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)

    def tick(self):
        if self.ppi.elapsed_time >= self.ppi.t:
            self.timer.stop()
            print(f"Simulation finished at t={self.ppi.elapsed_time:.2f}s")
            self.close()
            return

        detections = self.ppi.update()
        self.viewer.redraw()
        self.viewer.viewport().update()
        self.det_plot.add_detections(self.ppi.elapsed_time, detections)
        self.amp_plot.add_detections(self.ppi.elapsed_time, detections)
        self.phase_plot.add_detections(self.ppi.elapsed_time, detections)
        pulse_data = self.pulse_widget.update_pulse(detections)
        if pulse_data:
            self.proc_widget.update(pulse_data)

        if self.video_writer:
            pixmap = self.centralWidget().grab()
            pixmap = pixmap.scaled(
                self.video_size[0],
                self.video_size[1],
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            img = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
            ptr = img.bits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.width(), 3)
            self.video_writer.append_data(arr.copy())
            
            if self.max_video_mb is not None:
                import os
                try:
                    size_mb = os.path.getsize(self.output_file) / (1024 * 1024)
                    if size_mb >= self.max_video_mb:
                        print(f"Video size limit reached ({size_mb:.1f} MB >= {self.max_video_mb} MB). Stopping simulation.")
                        self.timer.stop()
                        self.close()
                except FileNotFoundError:
                    pass

    def closeEvent(self, event):
        if self.video_writer:
            self.video_writer.close()
            print(f"Video saved to {self.output_file}")
        super().closeEvent(event)