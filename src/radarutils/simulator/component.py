"""
component.py — Componentes físicos do simulador de radar.

Define as classes de base e especializadas para todos os objetos que
compõem a cena de simulação: o radar e os alvos (targets).

Hierarquia de classes::

    Components (ABC)
    ├── Radar
    └── Target
        ├── OrbitalTarget
        └── NestedOrbitalTarget

O método ``update(dt)`` avança o estado cinemático de cada componente
por um passo de tempo ``dt`` (em segundos).
"""

import abc
import numpy as np

from ..core.basics import calc_max_prf


# ══════════════════════════════════════════════════════════════════════════
#  Classe Base Abstrata
# ══════════════════════════════════════════════════════════════════════════

class Components(abc.ABC):
    """
    Classe base abstrata para todos os componentes da cena.

    Mantém o estado cinemático (posição, velocidade, aceleração, ângulo)
    e fornece métodos de atualização para movimento linear uniforme.

    Subclasses concretas (Radar, Target, etc.) sobrescrevem ``update()``
    para implementar sua cinemática específica.

    Attributes:
        x (float):            Posição x em metros (sistema cartesiano centrado no radar).
        y (float):            Posição y em metros.
        theta (float):        Ângulo de orientação/heading em radianos.
        velocity (float):     Módulo da velocidade em m/s.
        acceleration (float): Módulo da aceleração em m/s².
        phi (float):          Ângulo de fase auxiliar (uso específico de subclasses).
    """

    def __init__(self, x: float, y: float, vel: float, acc: float, theta: float):
        self.x            = x
        self.y            = y
        self.theta        = theta
        self.velocity     = vel
        self.acceleration = acc
        self.phi          = 0

    def update_acceleration(self, acc: float) -> None:
        """Atualiza a aceleração do componente."""
        self.acceleration = acc

    def update_velocity(self, dt: float) -> None:
        """Atualiza a velocidade escalar por integração da aceleração: v += a·dt."""
        self.velocity += self.acceleration * dt

    def update_position(self, dt: float) -> None:
        """
        Atualiza a posição por integração da velocidade em direção a theta:
            x += v·cos(θ)·dt
            y += v·sin(θ)·dt
        """
        self.x += self.velocity * np.cos(self.theta) * dt
        self.y += self.velocity * np.sin(self.theta) * dt

    def update(self, dt: float) -> None:
        """
        Avança o estado cinemático por um passo de tempo dt.
        Primeiro atualiza a velocidade (v += a·dt) e depois a posição.
        """
        self.update_velocity(dt)
        self.update_position(dt)


# ══════════════════════════════════════════════════════════════════════════
#  Radar
# ══════════════════════════════════════════════════════════════════════════

class Radar(Components):
    """
    Radar monoestático com antena giratória.

    O radar fica fixo na origem (x=0, y=0) e rotaciona em torno do eixo Z
    a uma velocidade angular constante definida por ``rpm``.

    Attributes:
        rpm (float):       Rotações por minuto da antena.
        clockwise (bool):  Sentido de rotação (True = horário).
        prf (float):       Pulse Repetition Frequency máxima (Hz) calculada de r_max.
        pt (float):        Potência de transmissão (W).
        gt (float):        Ganho da antena em transmissão (dBi).
        s_min (float):     Sensibilidade mínima do receptor (W/m²).
        beamwidth (float): Largura do feixe principal (°).
        deg_step (float):  Resolução angular de quantização das detecções (°).
        irradPattern:      Callable opcional G(deg_error) → ganho linear.
                           Se None, usa modelo Gaussiano calibrado pela beamwidth.
    """

    def __init__(
        self,
        r_max: float,
        pt: float,
        gt: float,
        s_min: float,
        beamwidth: float,
        irradPattern,
        x: float = 0,
        y: float = 0,
        theta: float = 0,
        rpm: float = 1,
        clockwise: bool = False,
        deg_step: float = 0.1,
    ):
        """
        Args:
            r_max:       Alcance máximo do radar (m). Usado para calcular a PRF máxima.
            pt:          Potência de transmissão (W).
            gt:          Ganho da antena (dBi).
            s_min:       Sensibilidade mínima do receptor (W/m²).
            beamwidth:   Largura do feixe principal (graus).
            irradPattern: Padrão de irradiação customizado. Se None, usa Gaussiano.
            x, y:        Posição inicial (padrão: origem).
            theta:       Ângulo inicial da antena (graus).
            rpm:         Velocidade de rotação (rotações por minuto).
            clockwise:   Sentido horário se True, anti-horário se False.
            deg_step:    Resolução angular de detecção (graus).
        """
        super().__init__(x, y, 0, 0, theta)

        self.rpm         = rpm
        self.clockwise   = clockwise
        self.prf         = calc_max_prf(r_max)
        self.pt          = pt
        self.gt          = gt
        self.s_min       = s_min
        self.beamwidth   = beamwidth
        self.deg_step    = deg_step
        self.irradPattern = irradPattern

    def rotate(self, dt: float) -> None:
        """
        Avança o ângulo da antena por dt segundos.

        A velocidade angular é rpm × 360° / 60 s = rpm × 6°/s.
        O ângulo é mantido no intervalo [0°, 360°).

        Args:
            dt: Passo de tempo em segundos.
        """
        direction   = -1 if self.clockwise else 1
        self.theta += direction * self.rpm * dt
        self.theta  = self.theta % 360   # mantém em [0°, 360°)

    def update(self, dt: float) -> None:
        """Avança o estado do radar (rotação da antena) por dt segundos."""
        self.rotate(dt)


# ══════════════════════════════════════════════════════════════════════════
#  Targets
# ══════════════════════════════════════════════════════════════════════════

class Target(Components):
    """
    Alvo com movimento linear (ou estático).

    Herda a cinemática linear de ``Components``:
      x(t) = x₀ + v·cos(θ)·t + ½·a·cos(θ)·t²
      y(t) = y₀ + v·sin(θ)·t + ½·a·sin(θ)·t²

    Para um alvo estático, basta vel=0 e acc=0.
    """

    def __init__(self, x: float, y: float, vel: float, acc: float, theta: float):
        """
        Args:
            x, y:  Posição inicial (m).
            vel:   Velocidade inicial (m/s).
            acc:   Aceleração tangencial (m/s²).
            theta: Heading (rad). Direção do movimento.
        """
        super().__init__(x, y, vel, acc, theta)


class OrbitalTarget(Target):
    """
    Alvo em movimento orbital circular (com aceleração tangencial opcional).

    A trajetória é parametrizada em coordenadas polares:
      α(t) = α₀ + ω·t + ½·β·t²
      x(t) = r · cos(α(t))
      y(t) = r · sin(α(t))

    onde ω = v₀/r é a velocidade angular inicial e β = acc/r é a
    aceleração angular.  O heading (theta) é sempre tangente à órbita.

    Attributes:
        r (float):        Raio orbital (m).
        alpha (float):    Fase atual (rad).
        omega (float):    Velocidade angular atual (rad/s). Negativo se clockwise.
        beta (float):     Aceleração angular (rad/s²).
        clockwise (bool): Sentido de rotação.
    """

    def __init__(
        self,
        r: float,
        speed: float,
        acceleration: float,
        clockwise: bool = False,
        alpha_start: float = 0,
    ):
        """
        Args:
            r:            Raio da órbita (m).
            speed:        Velocidade tangencial inicial (m/s).
            acceleration: Aceleração tangencial (m/s²). Positivo = acelera.
            clockwise:    Se True, orbita no sentido horário.
            alpha_start:  Fase inicial da órbita (rad). Padrão: 0 (eixo +X).
        """
        # Posição inicial derivada da fase inicial
        x = r * np.cos(alpha_start)
        y = r * np.sin(alpha_start)

        # Heading tangente à órbita (perpendicular ao raio)
        direction_factor = -1 if clockwise else 1
        theta = alpha_start + (direction_factor * np.pi / 2)

        super().__init__(x, y, speed, acceleration, theta)

        self.r         = r
        self.alpha     = alpha_start
        self.clockwise = clockwise

        # Velocidade e aceleração angular (rad/s e rad/s²)
        self.omega = speed / r if r != 0 else 0
        self.beta  = acceleration / r if r != 0 else 0

        if clockwise:
            self.omega = -self.omega
            self.beta  = -self.beta

    def update(self, dt: float) -> None:
        """
        Avança o estado orbital por dt segundos.

        Integra aceleração → velocidade angular → fase, depois recalcula
        posição cartesiana e heading tangente à trajetória.
        """
        self.omega += self.beta  * dt
        self.alpha += self.omega * dt

        self.x = self.r * np.cos(self.alpha)
        self.y = self.r * np.sin(self.alpha)

        # Heading: tangente à órbita (sentido de ω)
        direction_factor = 1 if self.omega >= 0 else -1
        self.theta = self.alpha + (direction_factor * np.pi / 2)


class NestedOrbitalTarget(Target):
    """
    Alvo com movimento epicíclico — órbita dentro de órbita.

    O alvo descreve uma trajetória epicicloidal:
      P(t) = O₁(t) + O₂(t)

    onde O₁ é o centro orbital primário (orbitando o radar) e O₂ é a
    posição do alvo relativa a O₁ (órbita secundária).

    Isso produz trajetórias complexas como rosetas, hipocicloides e
    epicicloides, adequadas para testar algoritmos de rastreamento.

    Attributes:
        r1, r2 (float):         Raios das órbitas primária e secundária (m).
        alpha1, alpha2 (float): Fases atuais das órbitas (rad).
        omega1, omega2 (float): Velocidades angulares (rad/s). Negativas se CW.
        beta1, beta2 (float):   Acelerações angulares (rad/s²).
    """

    def __init__(
        self,
        r1: float,   speed1: float, acc1: float,
        r2: float,   speed2: float, acc2: float,
        clockwise1: bool = False,
        clockwise2: bool = False,
        alpha1_start: float = 0,
        alpha2_start: float = 0,
    ):
        """
        Args:
            r1, speed1, acc1:   Parâmetros da órbita primária (raio m, vel m/s, acc m/s²).
            r2, speed2, acc2:   Parâmetros da órbita secundária.
            clockwise1:         Direção da órbita primária (True = horário).
            clockwise2:         Direção da órbita secundária.
            alpha1_start:       Fase inicial da órbita primária (rad).
            alpha2_start:       Fase inicial da órbita secundária (rad).
        """
        # Posição inicial: soma das duas contribuições orbitais
        x1 = r1 * np.cos(alpha1_start)
        y1 = r1 * np.sin(alpha1_start)
        x2 = r2 * np.cos(alpha2_start)
        y2 = r2 * np.sin(alpha2_start)

        super().__init__(x1 + x2, y1 + y2, 0, 0, 0)

        # Órbita primária
        self.r1     = r1
        self.alpha1 = alpha1_start
        self.omega1 = speed1 / r1 if r1 != 0 else 0
        self.beta1  = acc1   / r1 if r1 != 0 else 0
        if clockwise1:
            self.omega1 = -self.omega1
            self.beta1  = -self.beta1

        # Órbita secundária
        self.r2     = r2
        self.alpha2 = alpha2_start
        self.omega2 = speed2 / r2 if r2 != 0 else 0
        self.beta2  = acc2   / r2 if r2 != 0 else 0
        if clockwise2:
            self.omega2 = -self.omega2
            self.beta2  = -self.beta2

    def update(self, dt: float) -> None:
        """
        Avança o estado epicíclico por dt segundos.

        Integra ambas as órbitas independentemente, soma as posições
        cartesianas e recalcula velocidade e heading resultantes.
        """
        # Integra velocidade angular de cada órbita
        self.omega1 += self.beta1 * dt
        self.alpha1 += self.omega1 * dt

        self.omega2 += self.beta2 * dt
        self.alpha2 += self.omega2 * dt

        # Posições das contribuições primária e secundária
        x1 = self.r1 * np.cos(self.alpha1)
        y1 = self.r1 * np.sin(self.alpha1)

        x2 = self.r2 * np.cos(self.alpha2)
        y2 = self.r2 * np.sin(self.alpha2)

        # Posição total (soma epicicloidal)
        self.x = x1 + x2
        self.y = y1 + y2

        # Velocidade total como soma das contribuições tangenciais
        vx = (-self.omega1 * self.r1 * np.sin(self.alpha1)
              - self.omega2 * self.r2 * np.sin(self.alpha2))
        vy = ( self.omega1 * self.r1 * np.cos(self.alpha1)
              + self.omega2 * self.r2 * np.cos(self.alpha2))

        self.theta    = np.arctan2(vy, vx)
        self.velocity = np.sqrt(vx**2 + vy**2)