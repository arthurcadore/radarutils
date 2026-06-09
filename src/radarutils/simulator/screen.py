import math
import sys
import numpy as np
import pyqtgraph as pg
import imageio

from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui

from radarutils.simulator.ppi import PPI
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

    C      = 3e8      # velocidade da luz (m/s)
    F_C    = 10e9     # portadora (Hz) — banda X
    B      = 30e6     # largura de banda do chirp (Hz)
    N      = 2000     # amostras por PRI
    SNR_DB = 20.0     # SNR padrão (dB) — ajustável via atributo ou construtor

    def __init__(self, ppi: PPI = None, snr_db: float = None, coherent_integration: bool = False, clutter_type: str = "None", normalize_plots: bool = True):
        super().__init__(QtCore.Qt.Vertical)
        self.ppi        = ppi
        self.snr_db     = snr_db if snr_db is not None else self.SNR_DB
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
        self.T_PRI  = 2.0 * r_max / self.C
        self.T_P    = self.T_PRI / 7.0
        self.fs     = self.N / self.T_PRI
        self.t      = np.linspace(0, self.T_PRI, self.N, endpoint=False)
        self.t_us   = self.t * 1e6
        self.k      = self.B / self.T_P
        self.n_p    = int(self.T_P * self.fs)       # amostras do pulso TX
        self.N_rx   = self.N - self.n_p             # amostras do período de escuta

        if ppi and ppi.radar:
            self.P_tx_dbm = 10.0 * np.log10(ppi.radar.pt * 1e3)
        else:
            self.P_tx_dbm = 60.0

        # Sinal TX: cos(π·k·t²) nos primeiros n_p samples
        self._tx            = np.zeros(self.N)
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
            self.t_us, np.zeros(self.N),
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
            self.t_us, np.zeros(self.N),
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
        rx         = np.zeros(self.N)          # sinal real (para plot)
        rx_complex = np.zeros(self.N, dtype=complex)  # sinal complexo (para integ. coerente)
        status_parts : list[str] = []

        if detections and self.ppi and self.ppi.radar:
            self.P_tx_dbm = 10.0 * np.log10(self.ppi.radar.pt * 1e3)

            for rec in detections:
                tau   = 2.0 * rec.range_m / self.C
                n_del = int(tau * self.fs)
                if n_del >= self.N:
                    continue

                a   = 10.0 ** ((rec.rx_power_dbm - self.P_tx_dbm) / 20.0)
                phi = (2.0 * np.pi * self.F_C * tau) % (2.0 * np.pi)

                tgt = self.ppi.targets[rec.target_idx]
                if rec.range_m > 0 and tgt.velocity > 0:
                    vx  = tgt.velocity * np.cos(tgt.theta)
                    vy  = tgt.velocity * np.sin(tgt.theta)
                    v_r = (vx * tgt.x + vy * tgt.y) / rec.range_m
                else:
                    v_r = 0.0
                f_d = 2.0 * v_r * self.F_C / self.C

                end      = min(n_del + self.n_p, self.N)
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
            c_noise = clutter_amp * (np.random.randn(self.N) + 1j * np.random.randn(self.N)) / np.sqrt(2)
            rx_complex += c_noise
            rx += np.real(c_noise)

        # ── Atualização do Header ─────────────────────────────────────────
        T_us   = self.T_P   * 1e6
        PRI_us = self.T_PRI * 1e6
        B_MHz  = self.B / 1e6
        r_min  = self.ppi.r_max / 7.0 if self.ppi else 0.0
        bw     = self.ppi.radar.beamwidth if (self.ppi and self.ppi.radar) else 0.0
        c_time = self.ppi.elapsed_time if self.ppi else 0.0
        t_total = self.ppi.t if self.ppi else 0.0
        n_samples = self.N
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
            f'<td align="right" style="color:#88CCFF; padding-right:4px;">Fc:</td><td align="left" style="font-weight: bold; padding-right:15px;">{self.F_C/1e9:.0f} GHz</td>'
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
        rx_noisy = rx_norm + sigma * np.random.randn(self.N)

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


class PPIViewer(pg.PlotWidget):
    """Widget de visualização do PPI radar (Plan Position Indicator)."""

    def __init__(self, ppi: PPI, show_vectors: bool = False):
        super().__init__()
        self.ppi = ppi
        self.radius = min(ppi.dimensions) / 2
        self.show_vectors = show_vectors

        self.setBackground('k')
        self.setAspectLocked(True)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.setXRange(-self.radius, self.radius)
        self.setYRange(-self.radius, self.radius)
        self._draw_grid()

        self.sweep = pg.PlotDataItem(pen=pg.mkPen((0, 255, 0), width=2))
        self.addItem(self.sweep)

        self.legend = self.addLegend(offset=(0, 0))
        self.legend.setParentItem(self.plotItem)
        self.legend.setZValue(1000)
        self.legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        self.legend.setPen(pg.mkPen((0, 255, 0), width=1))

        self.targets_plot = pg.ScatterPlotItem(size=12, pen=None)
        self.addItem(self.targets_plot)

        self.available_symbols = ['o', 's', 't', 'd', '+', 'x', 'star', 'p', 'h']
        self.target_legend_added:  set[int] = set()
        self.target_legend_labels: dict[int, object] = {}  # idx -> LabelItem

        self.beam_fill = QtWidgets.QGraphicsPathItem()
        self.beam_fill.setBrush(pg.mkBrush(0, 255, 0, 30))
        self.beam_fill.setPen(pg.mkPen(None))
        self.addItem(self.beam_fill)

        self.beam_low = pg.PlotDataItem(pen=pg.mkPen((0, 180, 0), width=1))
        self.addItem(self.beam_low)

        self.beam_high = pg.PlotDataItem(pen=pg.mkPen((0, 180, 0), width=1))
        self.addItem(self.beam_high)

        self.info_text = pg.TextItem(anchor=(1, 0))
        self.info_text.setZValue(1001)
        self.addItem(self.info_text)
        self.info_text.setHtml(
            """
            <div style="
                font-family: Consolas;
                font-size: 12pt;
                color: #00FF00;
                font-weight: bold;
                background-color: rgba(0,0,0,160);
                padding: 6px;
            ">
            PPI REAL
            </div>
            """
        )

        self.vectors_plot = pg.PlotDataItem(pen=pg.mkPen((255, 255, 255, 150), width=1))
        self.addItem(self.vectors_plot)

    def _draw_grid(self):
        """Desenha círculos concêntricos, linhas radiais e rótulos de ângulo."""
        steps = 4

        #  Cálculo e desenho do R_min 
        # T_PRI = 2 * r_max / c 
        # T_p = T_PRI / 7.0
        # r_min = c * T_p / 2 = r_max / 7.0
        r_min = self.ppi.r_max / 7.0
        r_min_sc = self.radius / 7.0  # escalar para coordenada de tela

        c_min = QtWidgets.QGraphicsEllipseItem(-r_min_sc, -r_min_sc, 2 * r_min_sc, 2 * r_min_sc)
        # Mesma espessura da borda do radar
        c_min.setPen(pg.mkPen((0, 180, 0), width=2))
        self.addItem(c_min)
        # -

        for i, r in enumerate(np.linspace(self.radius / steps, self.radius, steps)):
            if i == steps - 1:
                pen = pg.mkPen((0, 180, 0), width=2)
            else:
                pen = pg.mkPen((0, 80, 0), width=1, style=QtCore.Qt.DashLine)

            c = QtWidgets.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            c.setPen(pen)
            self.addItem(c)

            dist_val = self.ppi.r_max * (i + 1) / steps
            txt = pg.TextItem(f" {int(dist_val)}m ", color=(0, 180, 0), anchor=(0.5, 0))
            txt.setPos(0, r - 18)
            self.addItem(txt)

        for ang in range(0, 360, 30):
            t = math.radians(ang)

            x = self.radius * math.cos(t)
            y = self.radius * math.sin(t)
            self.addItem(pg.PlotDataItem([0, x], [0, y], pen=pg.mkPen((0, 60, 0), width=1)))

            tick_in  = self.radius - 10
            tick_out = self.radius + 10
            x1 = tick_in  * math.cos(t); y1 = tick_in  * math.sin(t)
            x2 = tick_out * math.cos(t); y2 = tick_out * math.sin(t)
            self.addItem(pg.PlotDataItem([x1, x2], [y1, y2], pen=pg.mkPen((0, 180, 0), width=1)))

            label_radius = self.radius + 60
            xt = label_radius * math.cos(t)
            yt = label_radius * math.sin(t)
            angle_txt = pg.TextItem(
                html=f"""
                <div style="
                    color: rgb(0,220,0);
                    font-weight: bold;
                    font-size: 10pt;
                    font-family: Consolas;
                ">
                {ang}°
                </div>
                """,
                anchor=(0.5, 0.5),
            )
            angle_txt.setPos(xt, yt)
            self.addItem(angle_txt)

    def redraw(self):
        """Atualiza todos os elementos visuais para o estado atual do PPI."""
        ppi = self.ppi

        # Varredura (sweep line)
        th = math.radians(ppi.radar.theta)
        x  = self.radius * math.cos(th)
        y  = self.radius * math.sin(th)
        self.sweep.setData([0, x], [0, y])

        # Bordas do feixe
        th_l = math.radians(ppi.theta_low)
        th_h = math.radians(ppi.theta_high)
        xl = self.radius * math.cos(th_l); yl = self.radius * math.sin(th_l)
        xh = self.radius * math.cos(th_h); yh = self.radius * math.sin(th_h)
        self.beam_low.setData([0, xl],  [0, yl])
        self.beam_high.setData([0, xh], [0, yh])

        # Setor preenchido
        path = QtGui.QPainterPath()
        path.moveTo(0, 0)
        r = self.radius
        path.arcTo(-r, -r, 2 * r, 2 * r, -ppi.theta_low, -ppi.radar.beamwidth)
        path.lineTo(0, 0)
        self.beam_fill.setPath(path)

        # Targets e vetores de velocidade
        pts: list[dict] = []
        vec_x: list[float] = []
        vec_y: list[float] = []
        v_scale = 0.5  # metros por m/s

        for i, tgt in enumerate(ppi.targets):
            sym = self.available_symbols[i % len(self.available_symbols)]

            speed = tgt.velocity  # magnitude do vetor de velocidade (m/s)

            if i not in self.target_legend_added:
                dummy = pg.ScatterPlotItem(symbol=sym, pen=None, brush=pg.mkBrush(0, 255, 0))
                self.legend.addItem(dummy, f"{speed:.1f} m/s")
                # Guarda referência ao LabelItem para atualizações futuras
                self.target_legend_labels[i] = self.legend.items[-1][1]
                self.target_legend_added.add(i)
            else:
                # Atualiza a velocidade a cada frame
                label_item = self.target_legend_labels.get(i)
                if label_item is not None:
                    label_item.setText(f"{speed:.1f} m/s", color=(0, 255, 0))

            pts.append({'pos': (tgt.x, tgt.y), 'symbol': sym, 'brush': pg.mkBrush(0, 255, 0)})

            if self.show_vectors and tgt.velocity > 0:
                vx = tgt.velocity * math.cos(tgt.theta) * v_scale
                vy = tgt.velocity * math.sin(tgt.theta) * v_scale
                tip_x = tgt.x + vx
                tip_y = tgt.y + vy

                head_size  = 5
                head_angle = math.radians(20)
                p1x = tip_x - head_size * math.cos(tgt.theta + head_angle)
                p1y = tip_y - head_size * math.sin(tgt.theta + head_angle)
                p2x = tip_x - head_size * math.cos(tgt.theta - head_angle)
                p2y = tip_y - head_size * math.sin(tgt.theta - head_angle)

                vec_x.extend([tgt.x, tip_x, p1x, np.nan, tip_x, p2x, np.nan])
                vec_y.extend([tgt.y, tip_y, p1y, np.nan, tip_y, p2y, np.nan])

        self.targets_plot.setData(pts)
        self.vectors_plot.setData(vec_x, vec_y)

        x_range, y_range = self.getViewBox().viewRange()
        self.info_text.setPos(x_range[1] - 2, y_range[1] - 10)

        # Legenda (canto superior esquerdo)
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(10, 10))
        self.legend.setPos(x_range[0] + 10, y_range[1] - 10)


class ProcessingWidget(QtWidgets.QSplitter):
    """
    Pipeline de processamento de sinal radar (Coluna 3).

    Blocos verticais seqüenciais:
      MTI  (amarelo) : Cancelador delay-line (1 atraso) — suprime ecos fixos.
      CFAR (ciano)   : CA-CFAR adaptativo — threshold por célula.
      INT  (branco)  : Integrador não-coerente de N_INT PRIs.
      PPI  (azul)    : PPI estimado com histórico de hits.
    """

    C          = 3e8
    N_GUARD    = 48      # aumentado para evitar vazamento do pulso comprimido (auto-mascaramento do alvo)
    N_TRAIN    = 256      # células para estimar o perfil de ruído local
    # N_TRAIN    = 96      # células para estimar o perfil de ruído local
    K_CFAR     = 10     # limiar multiplicativo ajustado para sinal pós-integrador (N_INT=8)
    N_INT      = 8       # número de PRIs para integração não-coerente
    H_HITS     = 4096    # histórico máximo de detecções no PPI estimado (aumentado substancialmente)
    MIN_CFAR_ABS = 1000.0   # threshold absoluto mínimo — suprime falsos alarmes em AWGN puro
    # MIN_CFAR_ABS = 10.0   # threshold absoluto mínimo — suprime falsos alarmes em AWGN puro

    # Distância máxima (m) para considerar que uma detecção CFAR corresponde a um alvo real
    MAX_MATCH_DIST = 90.0

    # Amplitude mínima dos eixos Y (evita variação do eixo com ruído puro)
    MIN_Y_MTI  = 60.0    # unidades do filtro casado após subtração MTI
    MIN_Y_INT  = 60.0    # unidades após integração não-coerente de N_INT PRIs
    MIN_Y_CFAR = 150.0    # unidades do eixo do CA-CFAR

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

        self.bottom_glw = pg.GraphicsLayoutWidget()
        self.bottom_glw.setBackground('k')

        self.addWidget(self.top_glw)
        self.addWidget(self.bottom_glw)

        # Usando multiplicador de CFAR empírico (devido à integração de múltiplos PRIs a PDF do ruído muda para Erlang/Gamma, invalidando a velha fórmula exp)
        self.alpha = self.K_CFAR

        N    = pulse_widget.N
        t_us = pulse_widget.t_us
        r_max = ppi.r_max

        # Estado interno
        self._mf_prev = np.zeros(N)
        self._mf_prev_complex = np.zeros(N, dtype=complex)  # para estimativa de desvio doppler
        from collections import deque
        # Buffer para integração Não-Coerente (potências reais)
        self._int_buffer: deque = deque(maxlen=self.N_INT)
        # Buffer para integração Coerente (amplitudes complexas)
        self._coh_buffer: deque = deque(maxlen=self.N_INT)
        # Rastro de detecções do PPI Estimado (acumulativo)
        # True detections (red)
        self._trail_xs: deque = deque(maxlen=self.H_HITS)
        self._trail_ys: deque = deque(maxlen=self.H_HITS)
        self._trail_az: deque = deque(maxlen=self.H_HITS)
        self._trail_vr: deque = deque(maxlen=self.H_HITS)  # velocidade radial estimada por ponto
        self._last_detected_vrs: list[float] = [0.0]  # Armazena as últimas velocidades detectadas
        # False alarms (yellow)
        self._fa_xs: deque = deque(maxlen=self.H_HITS)
        self._fa_ys: deque = deque(maxlen=self.H_HITS)
        self._fa_az: deque = deque(maxlen=self.H_HITS)
        # Contadores acumulados
        self._total_true: int = 0
        self._total_fa:   int = 0
        self._prev_az_deg: float | None = None

        # ── Plot 0 — MTI ───────────────────────────────────────────────
        self.mti_plot = self.top_glw.addPlot(row=0, col=0)
        self.mti_plot.setLabel('left', 'MTI')
        self.mti_plot.getAxis('left').setWidth(65)
        self.mti_plot.showGrid(x=True, y=True, alpha=0.22)
        self.mti_plot.setYRange(0, 10)
        self.mti_plot.setMouseEnabled(x=False, y=False)
        self.mti_curve = self.mti_plot.plot(
            t_us, np.zeros(N), pen=pg.mkPen((255, 255, 0), width=1)
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
            t_us, np.zeros(N), pen=pg.mkPen((210, 210, 210), width=1)
        )
        # Legenda do modo de integração (canto superior esquerdo, mesmo estilo do CFAR)
        self.int_legend = self.int_plot.addLegend(colCount=1)
        self.int_legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        self.int_legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(0, 0))
        _mode_str = "Coherent" if self.coherent_integration else "Non-Coherent"
        _dummy_int = pg.PlotDataItem(pen=pg.mkPen((210, 210, 210), width=1))
        self.int_legend.addItem(_dummy_int, f"Mode: {_mode_str}")

        # ── Plot 2 — CA-CFAR (aplicado sobre o integrador) ────────────────
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
        # Move a legenda para o canto superior esquerdo, dividida em colunas (1 linha)
        self.cfar_legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(0, 0))
        
        self.cfar_sig_curve = self.cfar_plot.plot(
            t_us, np.zeros(N), pen=pg.mkPen((0, 190, 255), width=1), name="Sinal (Rx)"
        )
        self.cfar_thr_curve = self.cfar_plot.plot(
            t_us, np.zeros(N),
            pen=pg.mkPen((255, 80, 80), width=1, style=QtCore.Qt.DashLine), name="Threshold CFAR"
        )
        self.cfar_spots = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 220), symbol='o'
        )
        self.cfar_plot.addItem(self.cfar_spots)

        # ── Plot 3 — PPI Estimado ───────────────────────────────────────
        self.ppi_est_plot = self.bottom_glw.addPlot(row=0, col=0)
        self.ppi_est_plot.setLabel('left', 'PPI Estimado', color='#00AAFF', size='12pt')
        self.ppi_est_plot.setAspectLocked(True)
        self.ppi_est_plot.hideAxis('bottom')
        self.ppi_est_plot.hideAxis('left')
        self.ppi_est_plot.setXRange(-r_max, r_max)
        self.ppi_est_plot.setYRange(-r_max, r_max)
        self._draw_ppi_grid(r_max)
        self.est_sweep = pg.PlotDataItem(pen=pg.mkPen((0, 140, 255), width=1))
        self.ppi_est_plot.addItem(self.est_sweep)

        # Pontos azuis = alvos reais detectados (vermelho)
        self.est_spots_true = pg.ScatterPlotItem(
            size=6, pen=pg.mkPen((200, 0, 0), width=1.0), brush=pg.mkBrush(255, 50, 50, 230), symbol='o'
        )
        self.ppi_est_plot.addItem(self.est_spots_true)

        # Pontos amarelos = falsos alarmes
        self.est_spots_fa = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen((200, 160, 0), width=1.0), brush=pg.mkBrush(255, 220, 0, 200), symbol='o'
        )
        self.ppi_est_plot.addItem(self.est_spots_fa)

        # Label estático (canto superior direito)
        self.ppi_est_label = pg.TextItem(anchor=(1, 0))
        self.ppi_est_label.setZValue(1001)
        self.ppi_est_plot.addItem(self.ppi_est_label)
        self.ppi_est_label.setHtml(
            '<div style="font-family:Consolas; font-size:12pt; color:#00AAFF;'
            ' font-weight:bold; background-color:rgba(0,0,0,160); padding:6px;">'
            'PPI ESTIMADO</div>'
        )

        # Legenda de contagem e velocidade (canto superior esquerdo)
        self.vel_legend = pg.TextItem(anchor=(0, 0))
        self.vel_legend.setZValue(1002)
        self.ppi_est_plot.addItem(self.vel_legend)
        self.vel_legend.setHtml(
            '<div style="font-family:Consolas; font-size:9pt;'
            ' background-color:rgba(0,0,0,160); padding:6px;">'
            '<span style="color:#FFDD00;">&#11044; FAR Count: 0</span><br/>'
            '<span style="color:#FF3333;">&#11044; DET Count: 0</span><br/>'
            '<span style="color:#FF3333;">&#11044;</span>'
            '<span style="color:#DDDDDD;"> V_r: <b>+0.0 m/s</b></span>'
            '</div>'
        )


        # ── Proporções ─────────────────────────────────────────────────
        self.top_glw.ci.layout.setRowStretchFactor(0, 1)
        self.top_glw.ci.layout.setRowStretchFactor(1, 1)
        self.top_glw.ci.layout.setRowStretchFactor(2, 1)
        
        self.setSizes([1000, 1000])

    # 

    def _draw_ppi_grid(self, r_max: float):
        """Grid circular do PPI estimado (tema azul-escuro)."""
        #  Cálculo e desenho do R_min 
        r_min = r_max / 7.0
        c_min = QtWidgets.QGraphicsEllipseItem(-r_min, -r_min, 2 * r_min, 2 * r_min)
        c_min.setPen(pg.mkPen((0, 120, 200), width=2))
        self.ppi_est_plot.addItem(c_min)
        # -

        steps = 4
        for i, r in enumerate(np.linspace(r_max / steps, r_max, steps)):
            pen = (
                pg.mkPen((0, 120, 200), width=2)
                if i == steps - 1
                else pg.mkPen((0, 60, 110), width=1, style=QtCore.Qt.DashLine)
            )
            c = QtWidgets.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            c.setPen(pen)
            self.ppi_est_plot.addItem(c)
            txt = pg.TextItem(
                f"{int(r_max*(i+1)/steps)}m", color=(0, 140, 200), anchor=(0.5, 0)
            )
            txt.setPos(0, r * 0.96)
            self.ppi_est_plot.addItem(txt)
        for ang in range(0, 360, 30):
            t_rad = math.radians(ang)
            self.ppi_est_plot.addItem(
                pg.PlotDataItem(
                    [0, r_max * math.cos(t_rad)],
                    [0, r_max * math.sin(t_rad)],
                    pen=pg.mkPen((0, 60, 110), width=1),
                )
            )
            # Texto do angulo
            label_radius = r_max + 60
            xt = label_radius * math.cos(t_rad)
            yt = label_radius * math.sin(t_rad)
            angle_txt = pg.TextItem(
                html=f"""
                <div style="
                    color: rgb(0,140,200);
                    font-weight: bold;
                    font-size: 10pt;
                    font-family: Consolas;
                ">
                {ang}°
                </div>
                """,
                anchor=(0.5, 0.5),
            )
            angle_txt.setPos(xt, yt)
            self.ppi_est_plot.addItem(angle_txt)

    # 

    def _ca_cfar(self, signal: np.ndarray) -> np.ndarray:
        """CA-CFAR vectorizado via uniform_filter1d — threshold adaptativo por célula."""
        from scipy.ndimage import uniform_filter1d
        ng = self.N_GUARD
        nt = self.N_TRAIN
        win_t = 2 * (ng + nt) + 1
        win_g = 2 * ng + 1
        # Média móvel sobre a janela total e sobre a janela de guarda
        # Usa mode='reflect' (ou 'nearest') em vez de 'constant' (zero) para não
        # subestimar o piso de ruído nas bordas (o que causa falsos alarmes no início e fim do gráfico)
        sum_t = uniform_filter1d(signal, size=win_t, mode='reflect') * win_t
        sum_g = uniform_filter1d(signal, size=win_g, mode='reflect') * win_g
        n_tr  = max(win_t - win_g, 1)
        return np.maximum(self.alpha * (sum_t - sum_g) / n_tr, 0.0)

    # 

    def update(self, pulse_data: dict) -> None:
        """Executa o pipeline completo de processamento a partir da saída do PulseWidget."""
        import scipy.signal as ss

        comp_disp   = pulse_data.get('comp_disp', np.zeros(self.pw.N))
        azimuth_deg = pulse_data.get('azimuth_deg', 0.0)
        t_us        = self.pw.t_us
        fs          = self.pw.fs

        # ── MTI (delay-line canceller de 1 atraso) ─────────────────────
        # Subtrai o pulso MF anterior do atual — cancela ecos fixos (clutter)
        mti = np.abs(comp_disp - self._mf_prev)
        self._mf_prev = comp_disp.copy()
        peak_mti = float(np.max(mti)) if mti.any() else 0.0
        
        if self.normalize_plots:
            # Apenas normaliza o display (0-1); mti continua com valores originais
            mti_disp = (mti / peak_mti) if peak_mti > 1e-30 else mti
            self.mti_curve.setData(t_us, mti_disp)
            self.mti_plot.setYRange(0, 1.05)
        else:
            self.mti_curve.setData(t_us, mti)
            self.mti_plot.setYRange(0, max(peak_mti * 1.15, self.MIN_Y_MTI))

        # ── Integrador (Coerente ou Não-Coerente) ─────────────────────────
        min_dist = max(1, int(0.04 * fs))

        if self.coherent_integration:
            # Integração Coerente: soma amplitudes complexas → |soma| melhora SNR de N_INT × em amplitude
            comp_complex = pulse_data.get('comp_complex', np.zeros(self.pw.N, dtype=complex))
            self._coh_buffer.append(comp_complex)
            coh_sum   = np.sum(list(self._coh_buffer), axis=0)   # soma complexa
            integrated = np.abs(coh_sum) ** 2                     # envelope de potência
        else:
            # Integração Não-Coerente: soma de |mti|² dos últimos N_INT PRIs
            self._int_buffer.append(mti ** 2)
            integrated = np.sum(list(self._int_buffer), axis=0)
        peak_int = float(np.max(integrated)) if integrated.any() else 0.0
        
        if self.normalize_plots:
            # Apenas normaliza o display (0-1); integrated continua com valores originais
            int_disp = (integrated / peak_int) if peak_int > 1e-30 else integrated
            self.int_curve.setData(t_us, int_disp)
            self.int_plot.setYRange(0, 1.05)
        else:
            self.int_curve.setData(t_us, integrated)
            self.int_plot.setYRange(0, max(peak_int * 1.15, self.MIN_Y_INT))

        # ── CA-CFAR (aplicado sobre a saída do integrador) ─────────────────
        # CFAR após integração: SNR melhorado em ~10*log10(N_INT) dB
        cfar_thresh = self._ca_cfar(integrated)
        effective_thresh = np.maximum(cfar_thresh, self.MIN_CFAR_ABS)
        
        if self.normalize_plots:
            # Normaliza integrador e threshold pelo mesmo fator para manter proporção no display
            cfar_norm_factor = max(peak_int, float(np.max(effective_thresh)))
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
            y_top = max(peak_int, float(np.max(effective_thresh))) * 1.2
            self.cfar_plot.setYRange(0, max(y_top, self.MIN_Y_CFAR))

        # Picos detectados: integrador > threshold efetivo (usa arrays originais)
        binary = (integrated > effective_thresh).astype(float) * integrated
        peaks_cfar, _ = ss.find_peaks(binary, distance=min_dist)
        
        if self.normalize_plots:
            # Spots também normalizados para [0,1] pelo mesmo fator
            spots_y = (integrated[peaks_cfar] / cfar_norm_factor) if cfar_norm_factor > 1e-30 else integrated[peaks_cfar]
            self.cfar_spots.setData(t_us[peaks_cfar], spots_y)
        else:
            self.cfar_spots.setData(t_us[peaks_cfar], integrated[peaks_cfar])

        # ── PPI Estimado — rastro cumulativo e deleção pela varredura ────────
        az_deg = azimuth_deg
        az_rad = math.radians(azimuth_deg)
        r_max  = self.ppi.r_max

        # Linha de varredura (sempre visível)
        self.est_sweep.setData(
            [0, r_max * math.cos(az_rad)],
            [0, r_max * math.sin(az_rad)],
        )

        direction = -1 if (self.ppi and self.ppi.radar and self.ppi.radar.clockwise) else 1
        clear_angle = 15.0  # Apaga 15 graus à frente da antena para formar o rastro ('fade')

        # Filtra os pontos (true detections), limpando aqueles que a varredura está prestes a passar
        from collections import deque
        new_xs, new_ys, new_az, new_vr = [], [], [], []
        for x, y, pt_az, vr in zip(self._trail_xs, self._trail_ys, self._trail_az, self._trail_vr):
            diff = (pt_az - az_deg + 180) % 360 - 180
            is_ahead = (diff * direction > 0) and (abs(diff) < clear_angle)
            if not is_ahead:
                new_xs.append(x)
                new_ys.append(y)
                new_az.append(pt_az)
                new_vr.append(vr)
        self._trail_xs = deque(new_xs, maxlen=self.H_HITS)
        self._trail_ys = deque(new_ys, maxlen=self.H_HITS)
        self._trail_az = deque(new_az, maxlen=self.H_HITS)
        self._trail_vr = deque(new_vr, maxlen=self.H_HITS)

        # Filtra os pontos (false alarms)
        new_fa_xs, new_fa_ys, new_fa_az = [], [], []
        for x, y, pt_az in zip(self._fa_xs, self._fa_ys, self._fa_az):
            diff = (pt_az - az_deg + 180) % 360 - 180
            is_ahead = (diff * direction > 0) and (abs(diff) < clear_angle)
            if not is_ahead:
                new_fa_xs.append(x)
                new_fa_ys.append(y)
                new_fa_az.append(pt_az)
        self._fa_xs = deque(new_fa_xs, maxlen=self.H_HITS)
        self._fa_ys = deque(new_fa_ys, maxlen=self.H_HITS)
        self._fa_az = deque(new_fa_az, maxlen=self.H_HITS)

        # ── Estimação de velocidade radial por desvio de fase (Doppler instantâneo) ───
        comp_complex = pulse_data.get('comp_complex', np.zeros(self.pw.N, dtype=complex))
        lam = self.C / 10e9  # comprimento de onda (banda X, 10 GHz)
        delta_phi = np.angle(np.conj(self._mf_prev_complex) * comp_complex)
        vr_map = delta_phi / (2.0 * np.pi * self.pw.T_PRI) * lam / 2.0  # m/s por amostra
        self._mf_prev_complex = comp_complex.copy()

        # ── Coleta as posições reais dos alvos do PPI ──────────────────────────
        real_targets = [(tgt.x, tgt.y) for tgt in self.ppi.targets] if self.ppi else []

        # ── Adiciona nova detecção CFAR ao rastro (com classificação) ──────────
        # CORREÇÃO DE BIAS DE RANGE:
        # correlate(mode='same') desloca o pico +( n_p-1)//2 amostras.
        # np.roll(+n_p//2) em PulseWidget adiciona mais n_p//2 amostras.
        # Total: n_p - 1 amostras de offset → erro ≈ r_max/7 ≈ 171m para r_max=1200m.
        # Subtraímos esse offset ao converter índice de pico em range.
        _mf_range_offset = self.pw.n_p - 1  # amostras de offset introduzidas pela cadeia MF

        new_true_vrs: list[float] = []  # velocidades das verdadeiras detecções neste PRI
        if len(peaks_cfar) > 0:
            r_min_blind = r_max * 0.07
            for p in peaks_cfar:
                p_corrected = max(0, p - _mf_range_offset)
                range_est = self.C * self.pw.t[p_corrected] / 2.0
                if not (r_min_blind < range_est < r_max):
                    continue
                det_x = range_est * math.cos(az_rad)
                det_y = range_est * math.sin(az_rad)

                # Verifica se há alvo real próximo (dentro de MAX_MATCH_DIST)
                matched = False
                for (tx, ty) in real_targets:
                    if math.hypot(det_x - tx, det_y - ty) <= self.MAX_MATCH_DIST:
                        matched = True
                        break

                if matched:
                    # Detecção verdadeira → ponto vermelho + velocidade
                    vr_est = float(vr_map[p])
                    self._trail_xs.append(det_x)
                    self._trail_ys.append(det_y)
                    self._trail_az.append(az_deg)
                    self._trail_vr.append(vr_est)
                    new_true_vrs.append(vr_est)
                    self._total_true += 1
                else:
                    # Falso alarme → ponto amarelo, sem velocidade
                    self._fa_xs.append(det_x)
                    self._fa_ys.append(det_y)
                    self._fa_az.append(az_deg)
                    self._total_fa += 1

        # ── Renderiza o rastro separado por cores ──────────────────────────────
        if self._trail_xs:
            self.est_spots_true.setData(list(self._trail_xs), list(self._trail_ys))
        else:
            self.est_spots_true.setData([], [])

        if self._fa_xs:
            self.est_spots_fa.setData(list(self._fa_xs), list(self._fa_ys))
        else:
            self.est_spots_fa.setData([], [])

        # ── Atualiza legenda no canto superior esquerdo ────────────────────────
        # Se houve novas detecções, atualiza a lista de últimas velocidades.
        # Caso contrário, mantém o que já estava (persistente).
        if new_true_vrs:
            self._last_detected_vrs = new_true_vrs.copy()

        legend_lines = [
            f'<span style="color:#FFDD00;">&#11044; FAR Count: {self._total_fa}</span>',
            f'<span style="color:#FF3333;">&#11044; DET Count: {self._total_true}</span>',
        ]
        for vr in self._last_detected_vrs:
            legend_lines.append(
                f'<span style="color:#FF3333;">&#11044;</span>'
                f'<span style="color:#DDDDDD;"> V_r: <b>{vr:+.1f} m/s</b></span>'
            )
        legend_html = (
            '<div style="font-family:Consolas; font-size:10pt;'
            ' background-color:rgba(0,0,0,170); padding:6px;">'
            + '<br/>'.join(legend_lines)
            + '</div>'
        )
        self.vel_legend.setHtml(legend_html)

        # Atualiza as posições fixas dos dois textos
        x_range, y_range = self.ppi_est_plot.getViewBox().viewRange()
        self.ppi_est_label.setPos(x_range[1] - 10, y_range[1] - 10)
        self.vel_legend.setPos(x_range[0] + 10, y_range[1] - 10)


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