import sys
import math
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import imageio
from radarutils.simulator.ppi import PPI

from radarutils.simulator.detection import DetectionPlot
import os
from pathlib import Path

def prepare_output_file(file_name="simulation.mp4"):
    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / "../../../data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / file_name

    # remove arquivo antigo
    if output_path.exists():
        output_path.unlink()

    return str(output_path)

class PPIViewer(pg.PlotWidget):
    def __init__(self, sim: PPI, show_vectors: bool = False):
        super().__init__()
        self.sim = sim
        self.radius = min(sim.dimensions) / 2
        self.show_vectors = show_vectors

        self.setBackground('k')
        self.setAspectLocked(True)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.setXRange(-self.radius, self.radius)
        self.setYRange(-self.radius, self.radius)
        self._draw_grid()
        self.sweep = pg.PlotDataItem(pen=pg.mkPen((0,255,0), width=2))
        self.addItem(self.sweep)

        self.legend = self.addLegend(offset=(0, 0))
        self.legend.setParentItem(self.plotItem)
        self.legend.setZValue(1000)
        self.legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        self.legend.setPen(pg.mkPen((0, 255, 0), width=1))
        self.targets_plot = pg.ScatterPlotItem(size=12, pen=None)
        self.addItem(self.targets_plot)
        
        self.available_symbols = ['o', 's', 't', 'd', '+', 'x', 'star', 'p', 'h']
        self.target_legend_added = set()

        self.beam_fill = QtWidgets.QGraphicsPathItem()
        self.beam_fill.setBrush(pg.mkBrush(0, 255, 0, 30))  # Semi-transparent green
        self.beam_fill.setPen(pg.mkPen(None))
        self.addItem(self.beam_fill)

        self.beam_low = pg.PlotDataItem(pen=pg.mkPen((0, 180, 0), width=1))
        self.addItem(self.beam_low)

        self.beam_high = pg.PlotDataItem(pen=pg.mkPen((0, 180, 0), width=1))
        self.addItem(self.beam_high)

        # Info text in top-right
        self.info_text = pg.TextItem(color=(0, 255, 0), anchor=(1, 0))
        self.info_text.setZValue(1001)
        self.addItem(self.info_text)

        # Velocity vectors setup (coordinate-based segments)
        self.vectors_plot = pg.PlotDataItem(pen=pg.mkPen((255, 255, 255, 150), width=1))
        self.addItem(self.vectors_plot)

    def _draw_grid(self):
        # Concentric circles and distance labels
        steps = 5

        for i, r in enumerate(np.linspace(self.radius/steps, self.radius, steps)):

            # último anel = borda principal
            if i == steps - 1:
                pen = pg.mkPen((0, 180, 0), width=2)
            else:
                pen = pg.mkPen((0, 80, 0), width=1, style=QtCore.Qt.DashLine)

            c = QtWidgets.QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
            c.setPen(pen)
            self.addItem(c)

            dist_val = self.sim.r_max * (i + 1) / steps
            txt = pg.TextItem(
                f"{int(dist_val)}m",
                color=(0, 180, 0),
                anchor=(0.5, 0)
            )
            txt.setPos(0, r - 18)
            self.addItem(txt)

        # Radial lines and angle labels
        for ang in range(0, 360, 10):
            t = math.radians(ang)

            # linha radial principal a cada 30°
            if ang % 30 == 0:
                x = self.radius * math.cos(t)
                y = self.radius * math.sin(t)

                self.addItem(
                    pg.PlotDataItem(
                        [0, x], [0, y],
                        pen=pg.mkPen((0, 60, 0), width=1)
                    )
                )

            # pequeno tick sobre a borda a cada 10°
            tick_in = self.radius - 10
            tick_out = self.radius + 10

            x1 = tick_in * math.cos(t)
            y1 = tick_in * math.sin(t)

            x2 = tick_out * math.cos(t)
            y2 = tick_out * math.sin(t)

            self.addItem(
                pg.PlotDataItem(
                    [x1, x2], [y1, y2],
                    pen=pg.mkPen((0, 180, 0), width=1)
                )
            )

            # label do grau
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
                anchor=(0.5, 0.5)
            )
            angle_txt.setPos(xt, yt)
            self.addItem(angle_txt)

    def redraw(self):
        th = math.radians(self.sim.radar.theta)
        x = self.radius * math.cos(th)
        y = self.radius * math.sin(th)
        self.sweep.setData([0,x],[0,y])

        th_l = math.radians(self.sim.theta_low)
        th_h = math.radians(self.sim.theta_high)
        xl = self.radius * math.cos(th_l)
        yl = self.radius * math.sin(th_l)
        xh = self.radius * math.cos(th_h)
        yh = self.radius * math.sin(th_h)
        self.beam_low.setData([0, xl], [0, yl])
        self.beam_high.setData([0, xh], [0, yh])

        path = QtGui.QPainterPath()
        path.moveTo(0, 0)
        r = self.radius
        path.arcTo(-r, -r, 2*r, 2*r, -self.sim.theta_low, -self.sim.radar.beamwidth)
        path.lineTo(0, 0)
        self.beam_fill.setPath(path)

        # alvos update
        pts = []
        vec_x = []
        vec_y = []
        heads = []
        v_scale = 0.5 # Coordinate-based scale (meters per m/s)
        
        for i, tgt in enumerate(self.sim.targets):
            sym = self.available_symbols[i % len(self.available_symbols)]
            
            # Add to legend if first time
            if i not in self.target_legend_added:
                dummy = pg.ScatterPlotItem(symbol=sym, pen=None, brush=pg.mkBrush(0, 255, 0))
                self.legend.addItem(dummy, f"Alvo {i}")
                self.target_legend_added.add(i)
            
            # Target color
            is_in_beam = getattr(tgt, 'in_beam', False)
            brush = pg.mkBrush(0, 255, 0) if not is_in_beam else pg.mkBrush(255, 255, 0)
            
            pts.append({
                'pos': (tgt.x, tgt.y), 
                'symbol': sym,
                'brush': brush
            })
            
            # Velocity vectors
            if self.show_vectors and tgt.velocity > 0:
                v_scale = 0.5 
                vx = tgt.velocity * math.cos(tgt.theta) * v_scale
                vy = tgt.velocity * math.sin(tgt.theta) * v_scale
                
                tip_x = tgt.x + vx
                tip_y = tgt.y + vy
                
                # Manual arrowhead segments (coordinate-based)
                head_size = 5 # meters
                head_angle = math.radians(20) # 20 degrees half-angle
                
                p1x = tip_x - head_size * math.cos(tgt.theta + head_angle)
                p1y = tip_y - head_size * math.sin(tgt.theta + head_angle)
                p2x = tip_x - head_size * math.cos(tgt.theta - head_angle)
                p2y = tip_y - head_size * math.sin(tgt.theta - head_angle)
                
                # Tail line + head lines
                vec_x.extend([tgt.x, tip_x, p1x, np.nan, tip_x, p2x, np.nan])
                vec_y.extend([tgt.y, tip_y, p1y, np.nan, tip_y, p2y, np.nan])

        self.targets_plot.setData(pts)
        self.vectors_plot.setData(vec_x, vec_y)

        # Update info text
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
        Current Time    : {self.sim.elapsed_time:6.2f} s<br>
        Beam Width : {self.sim.radar.beamwidth:6.2f}°
        </div>
        """

        self.info_text.setHtml(info_str)

        padding_top = 25
        padding_right = 25

        # limites atuais visíveis do gráfico
        x_range, y_range = self.getViewBox().viewRange()

        x_max = x_range[1]
        y_max = y_range[1]

        # anchor=(1,0) => canto superior direito do texto
        x = x_max - padding_right
        y = y_max - padding_top

        self.info_text.setPos(x, y)

        #alvos 

        padding_top = 10
        padding_left = 10

        x_range, y_range = self.getViewBox().viewRange()

        x_min = x_range[0]
        y_max = y_range[1]

        self.legend.anchor(
            itemPos=(0, 0),
            parentPos=(0, 0),
            offset=(padding_left, padding_top)
        )

        self.legend.setPos(x_min + padding_left, y_max - padding_top)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, show_vectors: bool = True, output_file: str = None):
        super().__init__()
        self.setWindowTitle('PPI RADAR SIMULATOR')
        self.resize(1200, 800)

        self.output_file = output_file
        self.video_writer = None
        self.video_size = (1200, 800)  # largura, altura fixas

        if self.output_file:
            self.video_writer = imageio.get_writer(
                self.output_file,
                fps=30,
                codec="libx264",
                quality=8
            )

        self.sim = PPI(dimensions=(2000,2000), dt=0.03, t=10)
        self.sim.r_max = 1000
        self.sim.add_radar(theta=0, rpm=15, clockwise=True)

        # alvos
        self.sim.add_target(x=200, y=200, vel=0, acc=0, theta=0)
        self.sim.add_target(x=-200, y=-200, vel=0, acc=0, theta=0)

        self.sim.add_orbital_target(r=400, speed=60, clockwise=True, alpha_start=0)
        self.sim.add_orbital_target(r=200, speed=60, clockwise=False, alpha_start=np.pi)
        

        self.sim.add_nested_orbital_target(r1=700, speed1=80, acc1=0, r2=200, speed2=100, acc2=0, clockwise1=True, clockwise2=False, alpha1_start=(0), alpha2_start=0)
        self.sim.add_nested_orbital_target(r1=200, speed1=20, acc1=0, r2=440, speed2=120, acc2=0, clockwise1=False, clockwise2=True, alpha1_start=np.pi/2, alpha2_start=0)


        # UI Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)

        self.viewer = PPIViewer(self.sim, show_vectors=show_vectors)
        self.det_plot = DetectionPlot()
        self.det_plot.setYRange(0, self.sim.r_max)

        layout.addWidget(self.viewer, stretch=10)
        layout.addWidget(self.det_plot, stretch=8)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)

    def tick(self):
        if self.sim.elapsed_time >= self.sim.t:
            self.timer.stop()
            print(f"Simulation finished at t={self.sim.elapsed_time:.2f}s")
            self.close()
            return

        detections = self.sim.update()
        self.viewer.redraw()
        self.viewer.viewport().update()
        self.det_plot.add_detections(self.sim.elapsed_time, detections)

        # Grab screen to MP4 if enabled
        if self.video_writer:
            # captura apenas área interna útil
            pixmap = self.centralWidget().grab()

            # redimensiona SEMPRE para tamanho fixo
            pixmap = pixmap.scaled(
                self.video_size[0],
                self.video_size[1],
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation
            )

            img = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)

            width = img.width()
            height = img.height()

            ptr = img.bits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, width, 3)

            self.video_writer.append_data(arr.copy())

        for t in self.sim.targets:
            lim = self.viewer.radius - 20
            if abs(t.x) > lim:
                t.theta = math.pi - t.theta
            if abs(t.y) > lim:
                t.theta = -t.theta

def closeEvent(self, event):
    if self.video_writer:
        self.video_writer.close()
        print(f"Video saved to {self.output_file}")
        super().closeEvent(event)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)

    output_file = prepare_output_file("simulation.mp4")

    w = MainWindow(output_file=output_file)
    w.show()

    sys.exit(app.exec())