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
        self.setLabel('bottom', 'Time', units='s')
        self.setLabel('left', 'Range', units='m')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setYRange(0, 1000)

        self.legend = self.addLegend(offset=(100, 10))
        self.legend.setParentItem(self.plotItem)
        self.legend.setZValue(1000)
        self.legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        self.legend.setPen(pg.mkPen(200, 200, 200, width=1))

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
        self.target_legend_added: set[int] = set()

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
        if not detection_list:
            self.times.append(t)
            self.ranges.append(0)
            self.brushes.append(pg.mkBrush(100, 100, 100, 50))
            self.symbols.append('o')
        else:
            for rec in detection_list:
                self.times.append(rec.time)
                self.ranges.append(rec.range_m)

                sym = self.available_symbols[rec.target_idx % len(self.available_symbols)]
                self.symbols.append(sym)

                if rec.target_idx not in self.target_legend_added:
                    dummy = pg.ScatterPlotItem(symbol=sym, pen=None, brush=pg.mkBrush(200, 200, 200))
                    self.legend.addItem(dummy, f"Target {rec.target_idx}")
                    self.target_legend_added.add(rec.target_idx)

                # 1 - norm: vermelho = centro (0°), azul = borda (bw/2°)
                norm = self._norm_deg_error(rec.deg_error)
                color = self.colormap.mapToQColor(1.0 - norm)
                color.setAlpha(180)
                self.brushes.append(pg.mkBrush(color))

        self.plot_data.setData(
            x=self.times,
            y=self.ranges,
            brush=self.brushes,
            symbol=self.symbols,
        )

        # Janela deslizante de 15 s
        if self.times:
            window_size = 15
            current_time = self.times[-1]
            if current_time > window_size:
                self.setXRange(current_time - window_size, current_time)
            else:
                self.setXRange(0, window_size)


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
        self.target_legend_added: set[int] = set()

        self.beam_fill = QtWidgets.QGraphicsPathItem()
        self.beam_fill.setBrush(pg.mkBrush(0, 255, 0, 30))
        self.beam_fill.setPen(pg.mkPen(None))
        self.addItem(self.beam_fill)

        self.beam_low = pg.PlotDataItem(pen=pg.mkPen((0, 180, 0), width=1))
        self.addItem(self.beam_low)

        self.beam_high = pg.PlotDataItem(pen=pg.mkPen((0, 180, 0), width=1))
        self.addItem(self.beam_high)

        self.info_text = pg.TextItem(color=(0, 255, 0), anchor=(1, 0))
        self.info_text.setZValue(1001)
        self.addItem(self.info_text)

        self.vectors_plot = pg.PlotDataItem(pen=pg.mkPen((255, 255, 255, 150), width=1))
        self.addItem(self.vectors_plot)

    def _draw_grid(self):
        """Desenha círculos concêntricos, linhas radiais e rótulos de ângulo."""
        steps = 5

        for i, r in enumerate(np.linspace(self.radius / steps, self.radius, steps)):
            if i == steps - 1:
                pen = pg.mkPen((0, 180, 0), width=2)
            else:
                pen = pg.mkPen((0, 80, 0), width=1, style=QtCore.Qt.DashLine)

            c = QtWidgets.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            c.setPen(pen)
            self.addItem(c)

            dist_val = self.ppi.r_max * (i + 1) / steps
            txt = pg.TextItem(f"{int(dist_val)}m", color=(0, 180, 0), anchor=(0.5, 0))
            txt.setPos(0, r - 18)
            self.addItem(txt)

        for ang in range(0, 360, 10):
            t = math.radians(ang)

            if ang % 30 == 0:
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

            if i not in self.target_legend_added:
                dummy = pg.ScatterPlotItem(symbol=sym, pen=None, brush=pg.mkBrush(0, 255, 0))
                self.legend.addItem(dummy, f"Target {i}")
                self.target_legend_added.add(i)

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

        # Texto de status (canto superior direito)
        info_str = f"""
        <div style="
            font-family: Consolas;
            font-size: 10pt;
            color: rgb(0,255,0);
            background-color: rgba(0,0,0,160);
            padding: 6px;
            line-height: 1.35;
        ">
        <b>RADAR SIMULATION STATUS</b><br>
        Current Time    : {ppi.elapsed_time:6.2f} s<br>
        Beam Width : {ppi.radar.beamwidth:6.2f}°
        </div>
        """
        self.info_text.setHtml(info_str)

        x_range, y_range = self.getViewBox().viewRange()
        self.info_text.setPos(x_range[1] - 25, y_range[1] - 25)

        # Legenda (canto superior esquerdo)
        self.legend.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=(10, 10))
        self.legend.setPos(x_range[0] + 10, y_range[1] - 10)

class MainWindow(QtWidgets.QMainWindow):
    """
    Janela principal da interface gráfica.

    Recebe um PPI já configurado e executa o loop de atualização via QTimer.
    Não define targets, radar nem parâmetros de simulação.
    """

    def __init__(self, ppi: PPI, show_vectors: bool = True, output_file: str = None):
        super().__init__()
        self.setWindowTitle('PPI RADAR SIMULATOR')
        self.resize(1200, 800)

        self.ppi = ppi
        self.output_file = output_file
        self.video_writer = None
        self.video_size = (1200, 800)

        if self.output_file:
            self.video_writer = imageio.get_writer(
                self.output_file,
                fps=30,
                codec="libx264",
                quality=8,
            )

        # Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)

        self.viewer   = PPIViewer(self.ppi, show_vectors=show_vectors)
        self.det_plot = DetectionPlot(ppi=self.ppi)
        self.det_plot.setYRange(0, self.ppi.r_max)

        layout.addWidget(self.viewer,   stretch=10)
        layout.addWidget(self.det_plot, stretch=8)

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

    def closeEvent(self, event):
        if self.video_writer:
            self.video_writer.close()
            print(f"Video saved to {self.output_file}")
        super().closeEvent(event)