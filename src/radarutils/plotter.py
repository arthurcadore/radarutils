import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots 
import os
from typing import Optional, List, Union, Tuple, Dict, Any

# General plot parameters
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["savefig.transparent"] = True

# Science plot style
plt.style.use("science")

# Colors and styles
mpl.rcParams["text.color"] = "black"
mpl.rcParams["axes.labelcolor"] = "black"
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
plt.rcParams["figure.figsize"] = (16, 9)

# Fonts
plt.rc("font", size=16)
plt.rc("axes", titlesize=22, labelsize=22)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
plt.rc("legend", fontsize=12, frameon=True)
plt.rc("figure", titlesize=22)

def create_figure(rows: int, cols: int, figsize: Tuple[int, int] = (16, 9)) -> Tuple[plt.Figure, gridspec.GridSpec]:
    r"""
    Creates a figure with `GridSpec`, returning the `fig` and `grid` objects for plotting.
    
    Args:
        rows (int): Number of rows in the GridSpec
        cols (int): Number of columns in the GridSpec
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        Tuple[plt.Figure, gridspec.GridSpec]: Tuple with the figure and GridSpec objects
    """
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(rows, cols, figure=fig)
    return fig, grid

def save_figure(fig: plt.Figure, filename: str, out_dir: str = "../../assets") -> None:
    r"""
    Saves the figure in `<out_dir>/<filename>` from the script root directory. 
    
    Args:
        fig (plt.Figure): Matplotlib `Figure` object
        filename (str): Output file name
        out_dir (str): Output directory
    
    Raises:
        ValueError: If the output directory is invalid
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(script_dir, out_dir))
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

class BasePlot:
    r"""
    Base class for plotting graphs, implementing common functionality for all plots.
    
    Args:
        ax (plt.Axes): Matplotlib `Axes` object. 
        title (str): Plot title. 
        labels (Optional[List[str]]): List of axis labels. 
        xlim (Optional[Tuple[float, float]]): Limits of the x-axis `x = [xlim[0], xlim[1]]`. 
        ylim (Optional[Tuple[float, float]]): Limits of the y-axis `y = [ylim[0], ylim[1]]`. 
        colors (Optional[Union[str, List[str]]]): Plot colors. 
        style (Optional[Dict[str, Any]]): Plot style.
    """
    def __init__(self,
                 ax: plt.Axes,
                 title: str = "",
                 labels: Optional[List[str]] = None,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 colors: Optional[Union[str, List[str]]] = None,
                 style: Optional[Dict[str, Any]] = None) -> None:
        self.ax = ax
        self.title = title
        self.labels = labels
        self.xlim = xlim
        self.ylim = ylim
        self.colors = colors
        self.style = style or {}

    # Apply general styles to the axis
    def apply_ax_style(self) -> None:
        grid_kwargs = self.style.get("grid", {"alpha": 0.6, "linestyle": "--", "linewidth": 0.5})
        self.ax.grid(True, **grid_kwargs)
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        if self.title:
            self.ax.set_title(self.title)
        self.apply_legend()

    # Apply legends
    def apply_legend(self) -> None:
        handles, labels = self.ax.get_legend_handles_labels()
        if not handles:
            return
        leg = self.ax.legend(
            loc="upper right",
            frameon=True,
            edgecolor="black",
            fancybox=True,
            fontsize=self.style.get("legend_fontsize", 12),
        )
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")
        frame.set_alpha(1)

    # Apply colors
    def apply_color(self, idx: int) -> Optional[str]:
        if self.colors is None:
            return None
        if isinstance(self.colors, str):
            return self.colors
        if isinstance(self.colors, (list, tuple)):
            return self.colors[idx % len(self.colors)]
        return None

class GainPatternPlot(BasePlot):
    r"""
    Plot class for antenna gain pattern in polar coordinates.

    Args:
        ax (plt.Axes): Polar axis
        gp: GainPattern object with theta_vec and gain_dBi_vec
        title (str): Plot title
        colors: Plot color(s)
        style: Additional style dict

    Examples:
        - 4 Different Antennas: ![pageplot](../../assets/plots/example_antenna_patterns.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 position: Tuple[int, int],
                 gp,
                 title: str = "Diagrama de Irradiação (dBi)",
                 colors: Optional[Union[str, List[str]]] = None,
                 style: Optional[Dict[str, Any]] = None):

        ax = fig.add_subplot(grid[position], projection='polar')

        super().__init__(
            ax=ax,
            title=title,
            labels=None,
            colors=colors,
            style=style
        )

        self.gp = gp
        self.plot()

    def plot(self) -> None:
        theta = self.gp.theta_vec
        gain = self.gp.gain_dBi_vec

        if len(theta) == 0 or len(gain) == 0:
            print("Erro: vetores vazios")
            return

        color = self.apply_color(0) or "blue"
        self.ax.plot(theta, gain, color=color)
        self.ax.set_rlabel_position(90)

        # Escala
        self.ax.set_ylim(np.min(gain), np.max(gain))

        self.apply_ax_style()