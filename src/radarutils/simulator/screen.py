import sys
import math
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg


from radarutils.simulator.ppi import PPI


class PPIViewer(pg.PlotWidget):
    def __init__(self, sim: PPI):
        super().__init__()
        self.sim = sim
        self.radius = min(sim.dimensions) / 2

        self.setBackground('k')
        self.setAspectLocked(True)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.setXRange(-self.radius, self.radius)
        self.setYRange(-self.radius, self.radius)

        self._draw_grid()

        self.sweep = pg.PlotDataItem(pen=pg.mkPen((0,255,0), width=2))
        self.addItem(self.sweep)

        self.targets_plot = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(0,255,0), pen=None)
        self.addItem(self.targets_plot)

    def _draw_grid(self):
        for r in np.linspace(self.radius/4, self.radius, 4):
            c = QtWidgets.QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
            c.setPen(pg.mkPen((0,70,0), width=1))
            self.addItem(c)
        for ang in range(0, 360, 30):
            t = math.radians(ang)
            x = self.radius * math.cos(t)
            y = self.radius * math.sin(t)
            self.addItem(pg.PlotDataItem([0,x],[0,y], pen=pg.mkPen((0,50,0), width=1)))

    def redraw(self):
        # linha do radar
        th = math.radians(self.sim.radar.theta)
        x = self.radius * math.cos(th)
        y = self.radius * math.sin(th)
        self.sweep.setData([0,x],[0,y])

        # alvos
        pts = []
        for tgt in self.sim.targets:
            pts.append({'pos': (tgt.x, tgt.y)})
        self.targets_plot.setData(pts)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PPI Demo')
        self.resize(900, 900)

        self.sim = PPI(dimensions=(1000,1000), dt=0.03)
        self.sim.r_max = 500
        self.sim.add_radar(theta=0, rpm=10)
        self.sim.add_orbital_target(r=300, speed=50, acceleration=10, clockwise=False, alpha_start=np.pi/2)
        self.sim.add_orbital_target(r=200, speed=50, acceleration=10, clockwise=True, alpha_start=np.pi/2)

        self.viewer = PPIViewer(self.sim)
        self.setCentralWidget(self.viewer)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)

    def tick(self):
        self.sim.update()
        self.viewer.redraw()

        for t in self.sim.targets:
            lim = self.viewer.radius - 20
            if abs(t.x) > lim:
                t.theta = math.pi - t.theta
            if abs(t.y) > lim:
                t.theta = -t.theta


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())