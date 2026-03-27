import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots 
import os
import numpy as np

from typing import Optional, List, Union, Tuple, Dict, Any
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.signal import freqz
from ..core.power import mag_to_db
from ..core.env_vars import *


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

def save_figure(fig: plt.Figure, filename: str, out_dir: str = "../../../assets") -> None:
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
    Example:
        ![pageplot](../../assets/plots/example_antenna_patterns.svg)
    
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
        # Only set ylim if min and max are different to avoid warning
        if min_gain != max_gain:
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

class TimePlot(BasePlot):
    r"""
    Class for plotting signals in the time domain, receiving a time vector $t$, and a list of signals $s(t)$.

    Args:
        fig (plt.Figure): Figure object
        grid (gridspec.GridSpec): GridSpec object
        pos (int): Plot position
        t (np.ndarray): Time vector
        signals (Union[np.ndarray, List[np.ndarray]]): Signal or list of signals $s(t)$.
        time_unit (str): Time unit for plotting ("ms" by default, can be "s").
        amp_norm (bool): Signal normalization for maximum amplitude

    Examples:
        - Modulator Time Domain Example: ![pageplot](assets/example_modulator_time.svg)
        - AWGN addition Time Domain Example: ![pageplot](assets/example_noise_time_ebn0.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 t: np.ndarray,
                 signals: Union[np.ndarray, List[np.ndarray]],
                 time_unit: str = "ms",
                 amp_norm: bool = False,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)

        self.amp_norm = amp_norm

        # Copy the input signals to avoid modifying the original signal
        original_signals = signals if isinstance(signals, (list, tuple)) else [signals]
        self.signals = [sig.copy() for sig in original_signals]

        # Time unit
        self.time_unit = time_unit.lower()
        if self.time_unit == "ms":
            self.t = t * 1e3
        else:
            self.t = t

        # Signal or list of signals
        if self.labels is None:
            self.labels = [f"Signal {i+1}" for i in range(len(self.signals))]

    def plot(self) -> None:
        # Normalization
        if self.amp_norm:
            max_val = np.max(np.abs(np.concatenate(self.signals)))
            if max_val > 0:
                f = 1 / max_val
                for i, sig in enumerate(self.signals):
                    self.signals[i] *= f

        # Plot
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        for i, sig in enumerate(self.signals):
            color = self.apply_color(i)
            if color is not None:
                self.ax.plot(self.t, sig, label=self.labels[i], color=color, **line_kwargs)
            else:
                self.ax.plot(self.t, sig, label=self.labels[i], **line_kwargs)

        # Labels
        xlabel = r"Time (ms)" if self.time_unit == "ms" else r"Time ($s$)"
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(r"Amplitude")
        self.apply_ax_style()


class FrequencyPlot(BasePlot):
    r"""
    Class for plotting signals in the frequency domain, receiving a sampling frequency $f_s$ and a signal $s(t)$ and performing the Fourier transform of the signal, according to the expression below. 

    $$
    \begin{equation}
        S(f) = \mathcal{F}\{s(t)\}
    \end{equation}
    $$

    Where:
        - $S(f)$: Signal in the frequency domain.
        - $s(t)$: Signal in the time domain.
        - $\mathcal{F}$: Fourier transform.
    
    Args:
        fig (plt.Figure): Figure object
        grid (gridspec.GridSpec): GridSpec object
        pos (int): Plot position
        fs (float): Sampling frequency
        signal (np.ndarray): Signal to be plotted
        fc (float): Central frequency

    Examples:
        - Modulator Frequency Domain Example: ![pageplot](assets/example_modulator_freq.svg)
        - AWGN addition Frequency Domain Example: ![pageplot](assets/example_noise_freq_ebn0.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 fs: float,
                 signal: np.ndarray,
                 fc: float = 0.0,
                 bandwidth: float | None = None,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.fs = fs
        self.fc = fc
        self.signal = signal
        self.bandwidth = bandwidth

    def plot(self) -> None:
        # Fourier transform
        freqs = np.fft.fftshift(np.fft.fftfreq(len(self.signal), d=1 / self.fs))
        fft_signal = np.fft.fftshift(np.fft.fft(self.signal))
        y = mag_to_db(fft_signal)

        # Frequency scale
        freqs = freqs / 1000
        fc = self.fc / 1000
        bw = self.bandwidth / 1000 if self.bandwidth is not None else None
        self.ax.set_xlabel(r"Frequency (kHz)")
        scale_khz = True

        # Plot main curve
        line_kwargs = {"linewidth": 1, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0)
        label = self.labels[0] if self.labels else None

        if color is not None:
            self.ax.plot(freqs, y, label=label, color=color, **line_kwargs)
        else:
            self.ax.plot(freqs, y, label=label, **line_kwargs)

        # Plot bandwidth markers (if provided)
        if bw is not None and bw > 0:
            for f in [fc - bw, fc + bw]:
                self.ax.axvline(f, color=COLOR_AUX2, linestyle="--", linewidth=2, alpha=0.8)
            unit = "kHz" if scale_khz else "Hz"
            self.ax.plot([], [], color=COLOR_AUX2, linestyle="--", label=f"$W$ = {self.bandwidth/1000:.3f} {unit}")

        # Labels
        self.ax.set_ylabel(r"Magnitude (dB)")
        if self.ylim is None:
            self.ax.set_ylim(-60, 5)

        self.apply_ax_style()


class TxRxSignalPlot(BasePlot):
    """
    Plotador de sinais TX e RX em dB, recebendo vetores diretamente,
    seguindo o padrão do plotter.py (herda BasePlot).
    """

    def __init__(self,
                 fig: plt.Figure,
                 grid,
                 position,
                 t_tx: np.ndarray,
                 a_tx: np.ndarray,
                 t_rx: np.ndarray,
                 a_rx: np.ndarray,
                 title: str = "Tempo x Potência (dB)"):

        ax = fig.add_subplot(grid[position])

        super().__init__(
            ax=ax,
            title=title,
            labels=["Tempo (s)", "Potência (dB)"],
            colors=["blue", "red"],
            style={"grid": {"alpha": 0.5, "linestyle": "--", "linewidth": 0.7}},
        )

        self.t_tx = t_tx
        self.a_tx = a_tx
        self.t_rx = t_rx
        self.a_rx = a_rx

        self.plot()

    @staticmethod
    def _sort_vectors(t: np.ndarray, a: np.ndarray):
        if len(t) == 0:
            return t, a
        idx = np.argsort(t)
        return t[idx], a[idx]

    def plot(self):
        # Ordenar
        t_tx, a_tx = self._sort_vectors(self.t_tx, self.a_tx)
        t_rx, a_rx = self._sort_vectors(self.t_rx, self.a_rx)

        # Converter para dB
        a_tx_db = 10 * np.log10(np.maximum(a_tx, 1e-12))
        a_rx_db = 10 * np.log10(np.maximum(a_rx, 1e-12))

        # Cores
        color_tx = self.apply_color(0) or "blue"
        color_rx = self.apply_color(1) or "red"
        
        markerline, stemlines, baseline = self.ax.stem(
            t_tx, a_tx_db, basefmt=" ", label="TX"
        )
        plt.setp(markerline, color=color_tx, marker='o')
        plt.setp(stemlines, color=color_tx)

        markerline, stemlines, baseline = self.ax.stem(
            t_rx, a_rx_db, basefmt=" ", label="RX"
        )
        plt.setp(markerline, color=color_rx, marker='o')
        plt.setp(stemlines, color=color_rx)

        # Labels
        self.ax.set_xlabel(self.labels[0])
        self.ax.set_ylabel(self.labels[1])

        # Estilo geral herdado de BasePlot
        self.apply_ax_style()

if __name__ == "__main__":
    from ..core.data import ImportData
    
    tx = ImportData("tx_data_radar_0").load()
    rx = ImportData("rx_data_radar_0").load()

    t_tx, a_tx = tx
    t_rx, a_rx = rx

    fig, grid = create_figure(1, 1)

    TxRxSignalPlot(
        fig=fig,
        grid=grid,
        position=(0, 0),
        t_tx=t_tx,
        a_tx=a_tx,
        t_rx=t_rx,
        a_rx=a_rx
    )

    plt.show()
    
#TODO: add phase plot into response. 
class FrequencyResponsePlot(BasePlot):
    r"""
    Plot the frequency response of a filter from its coefficients (b, a). 
    Calculates the Discrete Fourier Transform of the impulse response using `scipy.signal.freqz`.

    $$
        H(f) = \sum_{n=0}^{N} b_n e^{-j 2 \pi f n} \Big/ \sum_{m=0}^{M} a_m e^{-j 2 \pi f m}
    $$

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int): Position in the GridSpec
        b (np.ndarray): Coefficients of the numerator of the filter
        a (np.ndarray): Coefficients of the denominator of the filter
        fs (float): Sampling frequency
        f_cut (Optional[float]): Cut-off frequency of the filter (Hz)
        xlim (Optional[Tuple[float, float]]): Limit of the x-axis (Hz)
        worN (int): Number of points for the Discrete Fourier Transform
        show_phase (bool): If `True`, plots the phase of the frequency response
        xlabel (str): Label of the x-axis
        ylabel (str): Label of the y-axis

    Examples:
        - Frequency Domain Plot Example: ![pageplot](assets/example_lpf_freq_response.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 b: np.ndarray,
                 a: np.ndarray,
                 fs: float,
                 f_cut: float = None,
                 xlim: tuple = None,
                 worN: int = 1024,
                 show_phase: bool = False,
                 xlabel: str = r"Frequency (Hz)",
                 ylabel: str = r"Magnitude (dB)",
                 **kwargs) -> None:

        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.b = b
        self.a = a
        self.fs = fs
        self.f_cut = f_cut
        self.xlim = xlim
        self.worN = worN
        self.show_phase = show_phase
        self.xlabel = xlabel
        self.ylabel = ylabel    

    def plot(self) -> None:
        # Calculate frequency response
        w, h = freqz(self.b, self.a, worN=self.worN, fs=self.fs)
        magnitude = mag_to_db(h)

        # Plot
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0) or COLOR_IMPULSE
        label = self.labels[0] if self.labels else "$H(f)$"
        self.ax.plot(w, magnitude, color=color, label=label, **line_kwargs)

        # Plot the phase
        if self.show_phase:
            ax2 = self.ax.twinx()
            phase = np.unwrap(np.angle(h))
            ax2.plot(w, phase, color=LPF_PHASE_COLOR, linestyle="--", linewidth=1.5, label="Phase ($rad$)")
            ax2.set_ylabel("Phase ($rad$)")

        # Add vertical bar at cut-off frequency
        if self.f_cut is not None:
            self.ax.axvline(self.f_cut, color=LPF_CUT_OFF_COLOR, linestyle="--", linewidth=2, label=f"$f_c$ = {self.f_cut} Hz")

        # Adjust axis
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(-60, 5)

        # Adjust labels
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.apply_ax_style()


class GaussianNoisePlot(BasePlot):
    r"""
    Class to plot the probability density $p(x)$ of a given variance $\sigma^2$, following the expression below. 

    $$
    p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right)
    $$

    Where: 
        - $p(x)$: Probability density of the noise.
        - $\sigma^2$: Variance of the noise.
        - $x$: Amplitude of the noise.

    Args:
        fig (plt.Figure): Figure of the plot
        grid (gridspec.GridSpec): GridSpec of the plot
        pos (int): Position of the plot in the GridSpec
        variance (float): Variance of the noise
        num_points (int): Number of points for the gaussian curve
        legend (str): Legend of the plot
        xlabel (str): Label of the x-axis
        ylabel (str): Label of the y-axis
        xlim (Optional[Tuple[float, float]]): Limit of the x-axis
        span (int): Span of the plot

    Examples:
        - Noise Density Plot Example: ![pageplot](assets/example_noise_gaussian_ebn0.svg)
    """
    def __init__(self,
                 fig: plt.Figure,
                 grid: gridspec.GridSpec,
                 pos,
                 variance: float,
                 num_points: int = 5000,
                 legend: str = "$p(x)$",
                 xlabel: str = "Amplitude ($x$)",
                 ylabel: str = "Probability Density $p(x)$",
                 ylim: Optional[Tuple[float, float]] = None,
                 span: int = 100,
                 **kwargs) -> None:
        ax = fig.add_subplot(grid[pos])
        super().__init__(ax, **kwargs)
        self.variance = variance
        self.num_points = num_points
        self.legend = legend
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.ylim = ylim
        self.span = span

    def plot(self) -> None:
        # Calculate the pdf
        sigma = np.sqrt(self.variance)
        x = np.linspace(-self.span*sigma, self.span*sigma, self.num_points)
        pdf = (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-x**2 / (2*self.variance))

        # Plot
        line_kwargs = {"linewidth": 2, "alpha": 1.0}
        line_kwargs.update(self.style.get("line", {}))
        color = self.apply_color(0) or "darkgreen"

        # plot the pdf
        label = r"$p(x)$" + "\n" + r"$\sigma^2 = " + f"{self.variance:.4f}" + "$"
        self.ax.plot(pdf, x, label=label, color=color, **line_kwargs)
 

        # Adjust axis
        self.ax.set_xlabel(self.ylabel)  
        self.ax.set_ylabel(self.xlabel) 
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        else:
            self.ax.set_ylim([-1, 1])
        self.apply_ax_style()