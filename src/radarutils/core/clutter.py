import abc
import numpy as np

VALID_CLUTTER_TYPES: tuple[str, ...] = ("none", "rayleigh", "rice", "weibull")

def clutter_from_str(
    name:      str,
    n_samples: int,
    amplitude: float = 1e-6,
) -> "Clutter | None":
    """
    Cria uma instância de Clutter a partir de um nome em string.

    Mapeamento (case-insensitive):
        ``'rayleigh'`` - RayleighClutter
        ``'rice'``     - RiceClutter (k_factor=1.0)
        ``'weibull'``  - WeibullClutter (shape=1.0)
        ``'none'``     - None (sem clutter)

    Args:
        name:      Nome do modelo de clutter.
        n_samples: Número de amostras por PRI.
        amplitude: Amplitude característica do clutter.

    Returns:
        Instância de Clutter correspondente, ou None se name='none'.

    Raises:
        ValueError: Se o nome não corresponder a nenhum modelo em VALID_CLUTTER_TYPES.
    """
    key = name.strip().lower()

    if key in ("none", ""):
        return None
    elif key == "rayleigh":
        return RayleighClutter(n_samples, amplitude)
    elif key == "rice":
        return RiceClutter(n_samples, amplitude)
    elif key == "weibull":
        return WeibullClutter(n_samples, amplitude)
    else:
        valid = ", ".join(f"'{v}'" for v in VALID_CLUTTER_TYPES)
        raise ValueError(
            f"Modelo de clutter desconhecido: '{name}'. "
            f"Opções válidas: {valid}."
        )


class Clutter(abc.ABC):
    r"""
    Base abstract class for radar clutter models.

    Concrete subclasses implement ``generate()`` to return a complex IQ signal
    representing the environmental clutter in a Pulse Repetition Interval (PRI).

    $$
    \begin{equation}
        c[n] = I[n] + j\,Q[n]
    \end{equation}
    $$

    Where:
        - $c[n]$ is the generated complex clutter sample.
        - $I[n]$ is the In-phase component.
        - $Q[n]$ is the Quadrature component.
        - $n$ is the sample index within the PRI.

    Attributes:
        n_samples (int):   Número de amostras por PRI.
        amplitude (float): Amplitude característica do clutter.        
    """

    def __init__(self, n_samples: int, amplitude: float = 1e-6):
        """
        Args:
            n_samples: Número de amostras complexas por PRI.
            amplitude: Amplitude característica (controla a intensidade do clutter).
        """
        self.n_samples = n_samples
        self.amplitude = amplitude

    @abc.abstractmethod
    def generate(self) -> np.ndarray:
        """
        Gera um vetor de amostras de clutter complexas (IQ).

        Returns:
            np.ndarray: Sinal complexo de clutter com ``n_samples`` amostras.
        """

class RayleighClutter(Clutter):
    r"""
    Generate clutter with Rayleigh envelope statistics.
    
    Widely adopted model for returns from multiple simultaneous scatterers (ground, rain, sea).
    The I and Q components are independent and identically distributed Gaussian processes.

    $$
    \begin{equation}
        c[n] = \frac{A}{\sqrt{2}} \bigl( \mathcal{N}(0,1) + j\,\mathcal{N}(0,1) \bigr)
    \end{equation}
    $$
    
    Where:
        - $c[n]$ is the complex clutter sample.
        - $A$ is the characteristic amplitude.
        - $\mathcal{N}(0,1)$ is a standard normal distribution.
        - The envelope follows a Rayleigh distribution with parameter $\sigma = A / \sqrt{2}$.

    Examples: 
        ![pageplot](../../assets/plots/rayleigh_clutter.svg)

    <div class="referencia">
        <b>Reference:</b>
        <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Cap. 7)</p>
    </div>
    """

    def generate(self) -> np.ndarray:
        """Gera amostras IQ com envelope Rayleigh."""
        return self.amplitude * (
            np.random.randn(self.n_samples) + 1j * np.random.randn(self.n_samples)
        ) / np.sqrt(2)

    def generate_pdf(self, r: np.ndarray) -> np.ndarray:
        """Calcula a PDF teórica da distribuição Rayleigh."""
        sigma = self.amplitude / np.sqrt(2)
        pdf = (r / sigma**2) * np.exp(-r**2 / (2 * sigma**2))
        return np.where(r >= 0, pdf, 0.0)

class RiceClutter(Clutter):
    r"""
    Generate clutter with Rician envelope statistics.
    
    The Rice model represents an environment with a dominant specular component (deterministic)
    added to multiple diffuse scatterers (Gaussian). It is suitable for terrain clutter with
    specular reflection or calm sea with a regular wave component.
    
    $$
    \begin{equation}
        p(r) = \frac{r}{\sigma^2} \exp\left(-\frac{r^2 + \nu^2}{2\sigma^2}\right) I_0\left(\frac{r \nu}{\sigma^2}\right) h(r)
    \end{equation}
    $$

    It also can be expressed in terms of the Rice factor $K$: 

    $$
    \begin{equation}
        K = \frac{\nu^2}{2\sigma^2}
    \end{equation}
    $$
    
    Where:
        - $K$ is the Rice factor (ratio of dominant to diffuse power).
        - $\nu$ is the amplitude of the dominant component.
        - $\sigma$ is the standard deviation of the diffuse components.
        - $p(r)$ is the probability density function of the envelope.
        - $I_0$ is the modified Bessel function of the first kind with order zero.
        - $h(r)$ is the Heaviside step function.
        - For $K \to 0$, it degenerates to a Rayleigh distribution.
    
    Examples: 
        ![pageplot](../../assets/plots/rice_clutter.svg)

    <div class="referencia">
        <b>Reference:</b>
        <p>Merill I. Skolnik - Introduction To Radar Systems Third Edition (Cap. 7)</p>
    </div>
    """

    def __init__(self, n_samples: int, amplitude: float = 1e-6, k_factor: float = 1.0):
        """
        Args:
            n_samples: Número de amostras complexas por PRI.
            amplitude: Amplitude característica total do clutter.
            k_factor:  Fator K de Rice (≥ 0).  k_factor=0 → Rayleigh puro.
        """
        super().__init__(n_samples, amplitude)
        self.k_factor = max(k_factor, 0.0)

    def generate(self) -> np.ndarray:
        """Gera amostras IQ com envelope Rice."""
        # Potência total: P = amplitude²
        # Potência difusa: sigma² = P / (2*(K+1))
        sigma = self.amplitude / np.sqrt(2.0 * (self.k_factor + 1.0))

        # Amplitude do componente dominante (determinístico)
        nu = self.amplitude * np.sqrt(self.k_factor / (self.k_factor + 1.0))

        # Componentes I e Q: dominante + difusos
        i_component = nu + sigma * np.random.randn(self.n_samples)
        q_component =      sigma * np.random.randn(self.n_samples)

        return (i_component + 1j * q_component)

    def generate_pdf(self, r: np.ndarray) -> np.ndarray:
        """Calcula a PDF teórica da distribuição Rice (fórmula estendida)."""
        from scipy.special import i0
        sigma = self.amplitude / np.sqrt(2.0 * (self.k_factor + 1.0))
        nu = self.amplitude * np.sqrt(self.k_factor / (self.k_factor + 1.0))
        
        pdf = (r / sigma**2) * np.exp(-(r**2 + nu**2) / (2 * sigma**2)) * i0(r * nu / sigma**2)
        return np.where(r >= 0, pdf, 0.0)

class WeibullClutter(Clutter):
    r"""
    Generate clutter with Weibull envelope statistics.
    
    The Weibull distribution is more versatile than Rayleigh and Rice for modeling
    impulsive ("spiky") clutter, such as heavy sea clutter, intense rain, and low-altitude targets.

    $$
    \begin{equation}
        p(r) = \frac{c}{\lambda} \left(\frac{r}{\lambda}\right)^{c-1} \exp\!\left[-\left(\frac{r}{\lambda}\right)^c\right], \quad r \ge 0
    \end{equation}
    $$
    
    Where:
        - $p(r)$ is the probability density function of the envelope.
        - $c$ is the shape parameter.
        - $\lambda$ is the scale parameter (amplitude).
        - For $c = 2$ and $\lambda = \sigma\sqrt{2}$, it degenerates to Rayleigh.
    
    Examples: 
        ![pageplot](../../assets/plots/weibull_clutter.svg)

    <div class="referencia">
        <b>Reference:</b>
        <p>Hermann Rohling - Radar CFAR Thresholding in Clutter and Multiple Target Situations, IEEE Trans. AES, 1983.</p>
    </div>
    """

    def __init__(self, n_samples: int, amplitude: float = 1e-6, shape: float = 1.0):
        """
        Args:
            n_samples: Número de amostras complexas por PRI.
            amplitude: Amplitude característica (parâmetro de escala λ = amplitude).
            shape:     Parâmetro de forma c da distribuição Weibull (> 0).
        """
        super().__init__(n_samples, amplitude)
        self.shape = max(shape, 1e-6)

    def generate(self) -> np.ndarray:
        """Gera amostras IQ com envelope Weibull e fase aleatória uniforme."""
        # Envelope: Weibull(shape=c, scale=amplitude)
        envelope = np.random.weibull(self.shape, self.n_samples) * self.amplitude

        # Fase aleatória uniforme [0, 2π)
        phase = np.random.uniform(0.0, 2.0 * np.pi, self.n_samples)

        return envelope * np.exp(1j * phase)

    def generate_pdf(self, r: np.ndarray) -> np.ndarray:
        """Calcula a PDF teórica da distribuição Weibull."""
        lam = self.amplitude
        c = self.shape
        # Add small epsilon to avoid division by zero at r=0 for c < 1
        r_safe = np.where(r == 0, 1e-10, r)
        pdf = (c / lam) * (r_safe / lam)**(c - 1) * np.exp(-(r_safe / lam)**c)
        return np.where(r >= 0, pdf, 0.0)
