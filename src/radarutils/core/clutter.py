"""
clutter.py — Geração de clutter para simulação de radar.

References:
    Merill I. Skolnik — Introduction To Radar Systems, 3rd Ed. (Cap. 7).
    Hermann Rohling — Radar CFAR Thresholding in Clutter and Multiple Target
        Situations, IEEE Trans. AES, 1983.
"""

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
    """
    Classe base abstrata para modelos de clutter radar.

    Subclasses concretas implementam ``generate()`` para retornar um sinal
    complexo IQ que representa o clutter ambiental em um PRI.

    Attributes:
        n_samples (int):   Número de amostras por PRI.
        amplitude (float): Amplitude característica do clutter (controla σ).
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
    Clutter com estatística de envelope Rayleigh.

    Modelo amplamente adotado para retornos de múltiplos espalhadores
    simultâneos (solo, chuva, mar).  As componentes I e Q são processos
    gaussianos independentes e identicamente distribuídos:

    .. math::

        c[n] = \frac{A}{\sqrt{2}} \bigl( \mathcal{N}(0,1) + j\,\mathcal{N}(0,1) \bigr)

    de modo que o envelope segue a distribuição Rayleigh com parâmetro
    :math:`\sigma = A / \sqrt{2}`.
    """

    def generate(self) -> np.ndarray:
        """Gera amostras IQ com envelope Rayleigh."""
        return self.amplitude * (
            np.random.randn(self.n_samples) + 1j * np.random.randn(self.n_samples)
        ) / np.sqrt(2)

class RiceClutter(Clutter):
    r"""
    Clutter com estatística de envelope Rice (Rician).

    O modelo Rice representa um ambiente com um componente especular
    dominante (determinístico) somado a múltiplos espalhadores difusos
    (gaussianos).  É adequado para clutter de terreno com reflexão
    especular ou mar calmo com componente de onda regular.

    O envelope de Rice é parametrizado pelo fator K (Rice factor):

    .. math::

        K = \frac{\nu^2}{2\sigma^2}

    onde :math:`\nu` é a amplitude do componente dominante e :math:`\sigma`
    é o desvio-padrão dos difusos.  Para K→0 degenera em Rayleigh.

    Attributes:
        k_factor (float): Fator K de Rice (razão potência dominante / difusa).
                          Valores típicos: 0 (Rayleigh) a 10 (muito especular).
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

class WeibullClutter(Clutter):
    r"""
    Clutter com estatística de envelope Weibull.

    A distribuição Weibull é mais versátil que Rayleigh e Rice para modelar
    clutter impulsivo ("spiky") como clutter de mar agitado, chuva intensa
    e alvos de baixa altitude.

    A PDF do envelope é:

    .. math::

        p(r) = \frac{c}{\lambda} \left(\frac{r}{\lambda}\right)^{c-1}
               \exp\!\left[-\left(\frac{r}{\lambda}\right)^c\right],
               \quad r \ge 0

    onde :math:`c` é o parâmetro de forma e :math:`\lambda` é o parâmetro
    de escala. Para c=2 e :math:`\lambda = \sigma\sqrt{2}` degenera em
    Rayleigh.

    O sinal complexo é gerado como:
        envelope ~ Weibull(c, λ),
        fase ~ Uniforme(0, 2π).

    Attributes:
        shape (float): Parâmetro de forma c (> 0).
                       c < 2 → cauda pesada (spiky); c = 2 → Rayleigh; c > 2 → compacto.
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
