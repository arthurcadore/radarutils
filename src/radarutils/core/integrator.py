r"""
integrator.py — Integrador de pulsos.

Define a classe base abstrata ``PulseIntegrator`` e duas implementações
concretas para integração de pulsos radar:

  - ``NonCoherentIntegrator`` : soma de potências (|y|²) dos últimos N PRIs.
  - ``CoherentIntegrator``    : soma vetorial IQ dos últimos N PRIs.

Hierarquia de classes::

    PulseIntegrator (ABC)
    ├── NonCoherentIntegrator
    └── CoherentIntegrator

A função ``integrator_from_str()`` cria a instância correta a partir de uma
string, seguindo o padrão de ``radarutils.core.clutter.clutter_from_str``.

References:
    Merill I. Skolnik — Introduction To Radar Systems, 3rd Ed. (Cap. 2).
"""

import abc
import numpy as np
from collections import deque

VALID_INTEGRATOR_TYPES: tuple[str, ...] = ("noncoherent", "coherent")


def integrator_from_str(
    name: str,
    n_int: int,
) -> "PulseIntegrator":
    """
    Cria uma instância de ``PulseIntegrator`` a partir de um nome em string.

    Mapeamento (case-insensitive):
        ``'noncoherent'``  → NonCoherentIntegrator
        ``'coherent'``     → CoherentIntegrator

    Args:
        name:  Nome do modo de integração.
        n_int: Número de PRIs a acumular no buffer.

    Returns:
        Instância de ``PulseIntegrator`` correspondente.

    Raises:
        ValueError: Se ``name`` não corresponder a nenhum tipo válido.
    """
    key = name.strip().lower().replace("-", "").replace("_", "")

    if key == "noncoherent":
        return NonCoherentIntegrator(n_int)
    elif key == "coherent":
        return CoherentIntegrator(n_int)
    else:
        valid = ", ".join(f"'{v}'" for v in VALID_INTEGRATOR_TYPES)
        raise ValueError(
            f"Tipo de integrador desconhecido: '{name}'. "
            f"Opções válidas: {valid}."
        )


class PulseIntegrator(abc.ABC):
    r"""
    Classe base abstrata para integradores de pulso radar.

    Mantém um buffer circular dos últimos ``n_int`` PRIs e acumula energia
    para melhorar a relação sinal-ruído (SNR).

    Subclasses concretas implementam ``process()`` com sua lógica específica
    de acumulação.

    Examples:
        ![integrator_chain_noncoherent](../../assets/plots/integrator_chain_noncoherent.svg)

    Attributes:
        n_int (int):     Número de PRIs a integrar.
        _buffer (deque): Buffer circular dos sinais recentes.
    """

    def __init__(self, n_int: int):
        """
        Args:
            n_int: Número de PRIs a integrar (tamanho do buffer).
        """
        self.n_int   = n_int
        self._buffer = deque(maxlen=n_int)

    @abc.abstractmethod
    def process(
        self,
        mti_real: np.ndarray,
        comp_complex: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Processa um PRI e retorna o sinal integrado acumulado.

        Args:
            mti_real (np.ndarray):        Sinal real pós-MTI.
            comp_complex (np.ndarray | None): Sinal IQ complexo do MF
                                              (necessário para CoherentIntegrator).

        Returns:
            np.ndarray: Sinal integrado (potência acumulada).
        """


class NonCoherentIntegrator(PulseIntegrator):
    r"""
    Integração não-coerente (detecção de envelope / soma de potências).

    Acumula a potência instantânea de cada PRI e soma os últimos ``n_int``
    quadros, sem preservar a fase do sinal:

    $$
    z[n] = \sum_{k=0}^{N_{\text{int}}-1} |y_k[n]|^2
    $$

    Ganho de SNR ≈ :math:`\sqrt{N_{\text{int}}}` (≈ :math:`10\log_{10}(N_{\text{int}})/2` dB).

    Vantagem: robusto a variações de fase entre PRIs (ideal quando não há
    coerência de fase garantida entre pulsos).

    Examples: 
        ![integrator_pulses_noncoherent](../../assets/plots/integrator_pulses_noncoherent.svg)

    <div class="referencia">
        <b>Reference:</b>
        <p>Merill I. Skolnik — Introduction To Radar Systems, 3rd Ed. (Cap. 2).</p>
    </div>
    """

    def process(
        self,
        mti_real: np.ndarray,
        comp_complex: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Soma as potências dos últimos ``n_int`` PRIs.

        Args:
            mti_real:    Sinal real pós-MTI do PRI atual.
            comp_complex: Ignorado neste modo.

        Returns:
            np.ndarray: Soma de potências acumulada (shape igual a ``mti_real``).
        """
        self._buffer.append(mti_real ** 2)
        return np.sum(list(self._buffer), axis=0)


class CoherentIntegrator(PulseIntegrator):
    r"""
    Integração coerente (preservação de fase IQ / soma vetorial).

    Acumula as amplitudes complexas IQ dos últimos ``n_int`` PRIs e extrai
    o envelope de potência da soma vetorial:

    $$
    z[n] = \left|\sum_{k=0}^{N_{\text{int}}-1} \tilde{y}_k[n]\right|^2
    $$

    Ganho de SNR ≈ :math:`N_{\text{int}}` (≈ :math:`20\log_{10}(N_{\text{int}})` dB),
    porém exige coerência de fase entre PRIs consecutivos.

    Examples: 
        ![integrator_pulses_coherent](../../assets/plots/integrator_pulses_coherent.svg)

    <div class="referencia">
        <b>Reference:</b>
        <p>Merill I. Skolnik — Introduction To Radar Systems, 3rd Ed. (Cap. 2).</p>
    </div>
    """

    def process(
        self,
        mti_real: np.ndarray,
        comp_complex: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Soma vetorial IQ dos últimos ``n_int`` PRIs e retorna o envelope de potência.

        Args:
            mti_real:    Ignorado neste modo (mantido para compatibilidade de assinatura).
            comp_complex: Sinal IQ complexo do MF do PRI atual.

        Returns:
            np.ndarray: Envelope de potência |soma IQ|² (shape igual a ``comp_complex``).

        Raises:
            ValueError: Se ``comp_complex`` for None.
        """
        if comp_complex is None:
            raise ValueError(
                "comp_complex deve ser fornecido para CoherentIntegrator."
            )
        self._buffer.append(comp_complex)
        coh_sum = np.sum(list(self._buffer), axis=0)   # soma vetorial complexa
        return np.abs(coh_sum) ** 2                    # envelope de potência
