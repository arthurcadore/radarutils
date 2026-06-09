import math
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui

from .component import Radar, Target, OrbitalTarget, NestedOrbitalTarget
from .detection import DetectionLog, DetectionRecord
from .constants import WAVELENGTH_M, RCS_DEFAULT_M2, FOUR_PI_3
from .html_contents import (
    PPI_REAL_LABEL_HTML,
    get_ppi_real_angle_html,
    PPI_ESTIMATED_LABEL_HTML,
    PPI_ESTIMATED_INITIAL_LEGEND_HTML,
    get_ppi_est_angle_html,
    get_ppi_est_legend_html,
)

class PPI(): 
    def __init__(self, dimensions: tuple[int, int] = (1000, 1000), t=10, dt=0.0001):
        self.dimensions = dimensions
        self.t = t
        self.dt = dt
        self._step = 0
        self.elapsed_time = 0.0
        self.targets = []
        self.clutters = []
        self.radar = None
        self.r_max = self.dimensions[0]
        self.detection_log = DetectionLog()
    
    @property
    def theta_low(self):
        if self.radar:
            return self.radar.theta - self.radar.beamwidth / 2
        return 0

    @property
    def theta_high(self):
        if self.radar:
            return self.radar.theta + self.radar.beamwidth / 2
        return 0

    def add_target(self, x, y, vel=0, acc=0, theta=0):
        target = Target(x, y, vel, acc, theta)
        self.targets.append(target)

    def add_radar(self, pt=1000, gt=30, s_min=1e-10, beamwidth=10, irradPattern=None, theta=0, rpm=1, clockwise=False, deg_step=0.1):
        self.radar = Radar(self.r_max, pt, gt, s_min, beamwidth, irradPattern, theta=theta, rpm=rpm, clockwise=clockwise, deg_step=deg_step)

    def add_orbital_target(self, r, speed, acceleration=0, clockwise=False, alpha_start=0):
        target = OrbitalTarget(r, speed, acceleration, clockwise, alpha_start)
        self.targets.append(target)

    def add_nested_orbital_target(self, r1, speed1, acc1, r2, speed2, acc2, clockwise1=False, clockwise2=False, alpha1_start=0, alpha2_start=0):
        target = NestedOrbitalTarget(r1, speed1, acc1, r2, speed2, acc2, clockwise1, clockwise2, alpha1_start, alpha2_start)
        self.targets.append(target)

    def add_clutter():
        pass

    # 
    # Radar equation helpers
    # 

    _WAVELENGTH_M   = 0.03   # λ default: 10 GHz (banda X) em metros
    _RCS_DEFAULT_M2 = 1.0    # σ default: 1 m²  (alvo genérico)
    _4PI3           = (4 * np.pi) ** 3

    def _antenna_gain_linear(self, deg_error: float) -> float:
        """
        Retorna o ganho linear G(θ) da antena para um dado erro angular.

        Se o radar possui um ``irradPattern`` callable, ele é chamado com
        ``deg_error`` (em graus) e seu resultado é interpretado como ganho
        **linear** (não em dB).

        Caso contrário, aplica um modelo Gaussiano calibrado pela beamwidth:
            G(θ) = G_max · exp(-θ² / (2σ²))
        onde σ = beamwidth / (2 · √(2 · ln(2)))  ≈  beamwidth / 2.355
        (HPBW half-power convention).

        Args:
            deg_error: Erro angular em relação ao centro do feixe (graus).

        Returns:
            Ganho linear (adimensional, ≥ 0).
        """
        radar = self.radar
        G_max_linear = 10 ** (radar.gt / 10)   # dBi → linear

        if callable(radar.irradPattern):
            return float(radar.irradPattern(deg_error))

        # Modelo Gaussiano: σ tal que G(bw/2) = G_max/2  (−3 dB)
        bw_half = radar.beamwidth / 2.0
        if bw_half == 0:
            return G_max_linear
        sigma = bw_half / np.sqrt(2 * np.log(2))   # sigma em graus
        return G_max_linear * np.exp(-(deg_error ** 2) / (2 * sigma ** 2))

    def _calc_rx_power_dbm(self, range_m: float, deg_error: float) -> float:
        """
        Calcula a potência recebida (dBm) via equação do radar:

            P_rx = (Pt · G_tx(θ) · G_rx(θ) · λ² · σ) / ((4π)³ · R⁴)

        Para este modelo, G_tx = G_rx = G(θ) (radar monoestático com mesma
        antena para TX e RX, degradada pelo deg_error).

        Args:
            range_m:   Distância ao alvo (m).
            deg_error: Erro angular em relação ao centro do feixe (graus).

        Returns:
            Potência recebida em dBm.
        """
        if range_m <= 0:
            return -200.0

        G  = self._antenna_gain_linear(deg_error)
        Pt = self.radar.pt              # Watts
        lam = WAVELENGTH_M
        sigma = RCS_DEFAULT_M2

        P_rx_w = (Pt * G**2 * lam**2 * sigma) / (FOUR_PI_3 * range_m**4)
        P_rx_w = max(P_rx_w, 1e-30)    # evita log(0)
        return 10 * np.log10(P_rx_w * 1e3)   # W → dBm

    # 

    def update(self) -> list[DetectionRecord]:
        """
        Avança um passo de simulação.

        Retorna:
            Lista de DetectionRecord para os targets dentro do feixe neste passo.
            Também acumula os registros em self.detection_log.
        """
        # Conta steps inteiros para evitar drift de ponto flutuante
        self._step += 1
        self.elapsed_time = round(self._step * self.dt, 10)
        detections: list[DetectionRecord] = []

        for i, target in enumerate(self.targets):
            target.update(self.dt)

            if self.radar:
                # Cálculo de range e azimute do target
                r     = np.sqrt(target.x**2 + target.y**2)
                alpha = np.degrees(np.arctan2(target.y, target.x)) % 360

                # Limites do feixe normalizados para [0, 360)
                l = self.theta_low  % 360
                h = self.theta_high % 360

                # Verifica se o target está dentro do feixe (com wrap-around em 360)
                in_beam = False
                if l <= h:
                    in_beam = (l <= alpha <= h)
                else:  # caso de wrap-around (ex.: low=355, high=5)
                    in_beam = (alpha >= l or alpha <= h)

                if in_beam:
                    # Erro angular em graus em relação ao centro do feixe,
                    # quantizado pelo passo angular do radar (deg_step).
                    # diff ∈ (-bw/2, +bw/2]; 0 = centro exato.
                    radar_az  = self.radar.theta % 360
                    diff      = ((alpha - radar_az + 180) % 360) - 180  # graus
                    deg_step  = self.radar.deg_step
                    deg_error = round(diff / deg_step) * deg_step

                    # Potência RX via equação do radar
                    rx_power_dbm = self._calc_rx_power_dbm(r, deg_error)

                    record = DetectionRecord(
                        time         = self.elapsed_time,
                        target_idx   = i,
                        range_m      = r,
                        azimuth_deg  = alpha,
                        deg_error    = deg_error,
                        rx_power_dbm = rx_power_dbm,
                    )
                    detections.append(record)
                    self.detection_log.add(record)

                # Eventos de entrada/saída do feixe
                was_in_beam = getattr(target, 'in_beam', False)
                if in_beam and not was_in_beam:
                    print(f"[{self.elapsed_time:.3f}s] Target ENTER: R={r:.2f}, Az={alpha:.2f}°")
                elif not in_beam and was_in_beam:
                    print(f"[{self.elapsed_time:.3f}s] Target EXIT:  R={r:.2f}, Az={alpha:.2f}°")
                target.in_beam = in_beam

        if self.radar:
            self.radar.update(self.dt)

        return detections

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
        self.info_text.setHtml(PPI_REAL_LABEL_HTML)

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
                html=get_ppi_real_angle_html(ang),
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

from collections import deque
from .constants import H_HITS, MAX_MATCH_DIST

class PPIEstimatedTracker:
    def __init__(self, h_hits=H_HITS, max_match_dist=MAX_MATCH_DIST):
        self.H_HITS = h_hits
        self.MAX_MATCH_DIST = max_match_dist
        self.trail_xs = deque(maxlen=self.H_HITS)
        self.trail_ys = deque(maxlen=self.H_HITS)
        self.trail_az = deque(maxlen=self.H_HITS)
        self.trail_vr = deque(maxlen=self.H_HITS)
        self.fa_xs = deque(maxlen=self.H_HITS)
        self.fa_ys = deque(maxlen=self.H_HITS)
        self.fa_az = deque(maxlen=self.H_HITS)
        self.total_true = 0
        self.total_fa = 0
        self.last_detected_vrs = [0.0]

    def update_sweep(self, az_deg, direction, clear_angle=15.0):
        # Filtra true detections
        new_xs, new_ys, new_az, new_vr = [], [], [], []
        for x, y, pt_az, vr in zip(self.trail_xs, self.trail_ys, self.trail_az, self.trail_vr):
            diff = (pt_az - az_deg + 180) % 360 - 180
            is_ahead = (diff * direction > 0) and (abs(diff) < clear_angle)
            if not is_ahead:
                new_xs.append(x)
                new_ys.append(y)
                new_az.append(pt_az)
                new_vr.append(vr)
        self.trail_xs = deque(new_xs, maxlen=self.H_HITS)
        self.trail_ys = deque(new_ys, maxlen=self.H_HITS)
        self.trail_az = deque(new_az, maxlen=self.H_HITS)
        self.trail_vr = deque(new_vr, maxlen=self.H_HITS)

        # Filtra false alarms
        new_fa_xs, new_fa_ys, new_fa_az = [], [], []
        for x, y, pt_az in zip(self.fa_xs, self.fa_ys, self.fa_az):
            diff = (pt_az - az_deg + 180) % 360 - 180
            is_ahead = (diff * direction > 0) and (abs(diff) < clear_angle)
            if not is_ahead:
                new_fa_xs.append(x)
                new_fa_ys.append(y)
                new_fa_az.append(pt_az)
        self.fa_xs = deque(new_fa_xs, maxlen=self.H_HITS)
        self.fa_ys = deque(new_fa_ys, maxlen=self.H_HITS)
        self.fa_az = deque(new_fa_az, maxlen=self.H_HITS)

    def add_detection(self, det_x, det_y, az_deg, vr_est, real_targets):
        matched = False
        for (tx, ty) in real_targets:
            if math.hypot(det_x - tx, det_y - ty) <= self.MAX_MATCH_DIST:
                matched = True
                break

        if matched:
            self.trail_xs.append(det_x)
            self.trail_ys.append(det_y)
            self.trail_az.append(az_deg)
            self.trail_vr.append(vr_est)
            self.total_true += 1
            return True, vr_est
        else:
            self.fa_xs.append(det_x)
            self.fa_ys.append(det_y)
            self.fa_az.append(az_deg)
            self.total_fa += 1
            return False, 0.0


class PPIEstimatedViewer(pg.PlotWidget):
    def __init__(self, r_max):
        super().__init__()
        self.r_max = r_max
        self.setLabel('left', 'PPI Estimado', color='#00AAFF', size='12pt')
        self.setAspectLocked(True)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.setXRange(-r_max, r_max)
        self.setYRange(-r_max, r_max)
        self._draw_grid()

        self.est_sweep = pg.PlotDataItem(pen=pg.mkPen((0, 140, 255), width=1))
        self.addItem(self.est_sweep)

        self.est_spots_true = pg.ScatterPlotItem(
            size=6, pen=pg.mkPen((200, 0, 0), width=1.0), brush=pg.mkBrush(255, 50, 50, 230), symbol='o'
        )
        self.addItem(self.est_spots_true)

        self.est_spots_fa = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen((200, 160, 0), width=1.0), brush=pg.mkBrush(255, 220, 0, 200), symbol='o'
        )
        self.addItem(self.est_spots_fa)

        self.ppi_est_label = pg.TextItem(anchor=(1, 0))
        self.ppi_est_label.setZValue(1001)
        self.addItem(self.ppi_est_label)
        self.ppi_est_label.setHtml(PPI_ESTIMATED_LABEL_HTML)

        self.vel_legend = pg.TextItem(anchor=(0, 0))
        self.vel_legend.setZValue(1002)
        self.addItem(self.vel_legend)
        self.vel_legend.setHtml(PPI_ESTIMATED_INITIAL_LEGEND_HTML)

    def _draw_grid(self):
        r_min = self.r_max / 7.0
        c_min = QtWidgets.QGraphicsEllipseItem(-r_min, -r_min, 2 * r_min, 2 * r_min)
        c_min.setPen(pg.mkPen((0, 120, 200), width=2))
        self.addItem(c_min)

        steps = 4
        for i, r in enumerate(np.linspace(self.r_max / steps, self.r_max, steps)):
            pen = (
                pg.mkPen((0, 120, 200), width=2)
                if i == steps - 1
                else pg.mkPen((0, 60, 110), width=1, style=QtCore.Qt.DashLine)
            )
            c = QtWidgets.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            c.setPen(pen)
            self.addItem(c)
            txt = pg.TextItem(
                f"{int(self.r_max*(i+1)/steps)}m", color=(0, 140, 200), anchor=(0.5, 0)
            )
            txt.setPos(0, r * 0.96)
            self.addItem(txt)
        for ang in range(0, 360, 30):
            t_rad = math.radians(ang)
            self.addItem(
                pg.PlotDataItem(
                    [0, self.r_max * math.cos(t_rad)],
                    [0, self.r_max * math.sin(t_rad)],
                    pen=pg.mkPen((0, 60, 110), width=1),
                )
            )
            label_radius = self.r_max + 60
            xt = label_radius * math.cos(t_rad)
            yt = label_radius * math.sin(t_rad)
            angle_txt = pg.TextItem(
                html=get_ppi_est_angle_html(ang),
                anchor=(0.5, 0.5),
            )
            angle_txt.setPos(xt, yt)
            self.addItem(angle_txt)

    def update_view(self, tracker, az_rad):
        self.est_sweep.setData(
            [0, self.r_max * math.cos(az_rad)],
            [0, self.r_max * math.sin(az_rad)],
        )
        if tracker.trail_xs:
            self.est_spots_true.setData(list(tracker.trail_xs), list(tracker.trail_ys))
        else:
            self.est_spots_true.setData([], [])

        if tracker.fa_xs:
            self.est_spots_fa.setData(list(tracker.fa_xs), list(tracker.fa_ys))
        else:
            self.est_spots_fa.setData([], [])

        legend_html = get_ppi_est_legend_html(
            tracker.total_fa, tracker.total_true, tracker.last_detected_vrs
        )
        self.vel_legend.setHtml(legend_html)

        x_range, y_range = self.getViewBox().viewRange()
        self.ppi_est_label.setPos(x_range[1] - 10, y_range[1] - 10)
        self.vel_legend.setPos(x_range[0] + 10, y_range[1] - 10)

