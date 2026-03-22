import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots 
import os
from typing import Optional, List, Union, Tuple, Dict, Any
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
    Examples:
        - 4 Different Antennas: ![pageplot](../../assets/plots/example_antenna_patterns.svg)
    
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
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 position: Tuple[int, int],
                 gp,
                 title: str = "Diagrama de Irradiação (dBi)",
                 colors: Optional[Union[str, List[str]]] = None,
                 style: Optional[Dict[str, Any]] = None,
                 r_min: Optional[float] = None,
                 r_max: Optional[float] = None):

        ax = fig.add_subplot(grid[position], projection='polar')

        super().__init__(
            ax=ax,
            title=title,
            labels=None,
            colors=colors,
            style=style
        )

        self.gp = gp
        self.r_min = r_min
        self.r_max = r_max

        self.plot()

    def plot(self) -> None:
        theta = self.gp.theta_vec
        phi   = self.gp.phi_vec
        Hgain = self.gp.Hgain_dBi_vec
        Vgain = self.gp.Vgain_dBi_vec

        if len(theta) == 0 or len(Hgain) == 0 or len(Vgain) == 0:
            print("Erro: vetores vazios")
            return

        # Escolhe cores, ou default
        color_H = self.apply_color(0) or "blue"
        color_V = self.apply_color(1) or "red"

        # Plot H-plane e V-plane
        self.ax.plot(theta, Hgain, color=color_H, label="H-plane", linestyle="--")
        self.ax.plot(phi, Vgain, color=color_V, label="V-plane", linestyle=":")

        # Ajusta posição dos rótulos radiais
        self.ax.set_rlabel_position(90)

        # Ajusta escala radial com base nos dois vetores ou valores fornecidos
        min_gain = self.r_min if self.r_min is not None else min(np.min(Hgain), np.min(Vgain))
        max_gain = self.r_max if self.r_max is not None else max(np.max(Hgain), np.max(Vgain))
        self.ax.set_ylim(min_gain, max_gain)

        self.apply_ax_style()


class GainPattern3DPlot(BasePlot):
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 position: Tuple[int, int],
                 gp,
                 title: str = "3D Radiation Pattern (Estimated)",
                 cmap: str = "jet",
                 phi_res: int = 90,
                 normalize: bool = True):

        ax = fig.add_subplot(grid[position], projection='3d')

        super().__init__(
            ax=ax,
            title=title,
            labels=None,
            colors=None,
            style=None
        )

        self.gp = gp
        self.cmap = cmap
        self.phi_res = phi_res
        self.normalize = normalize

        self.plot()
    def plot(self) -> None:

        theta = self.gp.theta_vec
        phi   = self.gp.phi_vec

        if self.gp.gain_dBi_matrix is None:
            print("Erro: gain_dBi_matrix não definido")
            return

        phi_mask = phi <= np.pi
        phi_plot = phi[phi_mask]
        gain_matrix = self.gp.gain_dBi_matrix[phi_mask, :]
        THETA, PHI = np.meshgrid(theta, phi_plot)

        R = 10 ** (gain_matrix / 10)
        if self.normalize:
            R = R / np.max(R)

        X = R * np.sin(PHI) * np.cos(THETA)
        Y = R * np.sin(PHI) * np.sin(THETA)
        Z = R * np.cos(PHI)

        gmin = np.min(gain_matrix)
        gmax = np.max(gain_matrix)
        vmin = gmin - 1
        vmax = gmax + 1

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.get_cmap(self.cmap)
        facecolors = cmap(norm(gain_matrix))

        surf = self.ax.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True
        )

        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.set_title(self.title)

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array(gain_matrix)

        cbar = plt.colorbar(mappable, ax=self.ax, shrink=0.6, pad=0.1)
        cbar.set_label("Gain (dBi)")
        ticks = np.linspace(vmin, vmax, 5)
        ticks[0] = vmin
        ticks[-1] = vmax

        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

        self.ax.grid(True)
        self.apply_ax_style()