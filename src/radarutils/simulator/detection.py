import pyqtgraph as pg
from PySide6 import QtGui, QtCore

class DetectionPlot(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.setBackground('k')
        self.setLabel('bottom', 'Time', units='s')
        self.setLabel('left', 'Range', units='m')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setYRange(0, 1000) # Default range, PPI will likely update this
        
        self.legend = self.addLegend(offset=(100, 10))
        self.legend.setParentItem(self.plotItem)
        self.legend.setZValue(1000)
        self.legend.setBrush(pg.mkBrush(0, 0, 0, 160))
        self.legend.setPen(pg.mkPen(200, 200, 200, width=1))
        
        # We use ScatterPlotItem because we might have multiple detections at once
        self.plot_data = pg.ScatterPlotItem(
            size=8, 
            pen=pg.mkPen(None)
        )
        self.addItem(self.plot_data)
        
        # Available symbols in pyqtgraph
        self.available_symbols = ['o', 's', 't', 'd', '+', 'x', 'star', 'p', 'h']
        
        # Manual Jet-like colormap (Blue -> Cyan -> Green -> Yellow -> Red)
        self.colormap = pg.ColorMap(
            pos=[0.0, 0.25, 0.5, 0.75, 1.0],
            color=[
                (0, 0, 255, 255),   # Blue (Cold/Edge)
                (0, 255, 255, 255), # Cyan
                (0, 255, 0, 255),   # Green
                (255, 255, 0, 255), # Yellow
                (255, 0, 0, 255)    # Red (Hot/Center)
            ]
        )
        
        self.times = []
        self.ranges = []
        self.brushes = []
        self.symbols = []
        self.target_legend_added = set()
        
    def add_detections(self, t, detection_list):
        """
        Adds new detections to the plot history.
        detection_list: list of (range, error, target_idx) tuples.
        """
        if not detection_list:
            self.times.append(t)
            self.ranges.append(0)
            self.brushes.append(pg.mkBrush(100, 100, 100, 50))
            self.symbols.append('o')
        else:
            for r, err, idx in detection_list:
                self.times.append(t)
                self.ranges.append(r)
                
                # Symbol based on target index
                sym = self.available_symbols[idx % len(self.available_symbols)]
                self.symbols.append(sym)

                # Add to legend if first time seeing this target
                if idx not in self.target_legend_added:
                    # Create a dummy item for legend
                    dummy = pg.ScatterPlotItem(symbol=sym, pen=None, brush=pg.mkBrush(200, 200, 200))
                    self.legend.addItem(dummy, f"Target {idx}")
                    self.target_legend_added.add(idx)
                
                # Color based on beam error
                color = self.colormap.mapToQColor(1.0 - err)
                color.setAlpha(180)
                self.brushes.append(pg.mkBrush(color))
        
        # Update the plot with data and individualized brushes/symbols
        self.plot_data.setData(
            x=self.times, 
            y=self.ranges, 
            brush=self.brushes, 
            symbol=self.symbols
        )
        
        # Auto-scroll to keep the latest data visible (rolling window)
        if len(self.times) > 0:
            window_size = 15 # 15 seconds window
            current_time = self.times[-1]
            if current_time > window_size:
                self.setXRange(current_time - window_size, current_time)
            else:
                self.setXRange(0, window_size)
