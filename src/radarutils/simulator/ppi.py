import numpy as np
from .component import Radar, Target, OrbitalTarget, NestedOrbitalTarget
from .detection import DetectionLog, DetectionRecord

class PPI(): 
    def __init__(self, dimensions: tuple[int, int] = (1000, 1000), t=10, dt=0.0001):
        self.dimensions = dimensions
        self.t = t
        self.dt = dt
        self._step = 0
        self.elapsed_time = 0.0
        self.targets = []
        self.clutters = []
        self.radar = None
        self.r_max = self.dimensions[0]
        self.detection_log = DetectionLog()
    
    @property
    def theta_low(self):
        if self.radar:
            return self.radar.theta - self.radar.beamwidth / 2
        return 0

    @property
    def theta_high(self):
        if self.radar:
            return self.radar.theta + self.radar.beamwidth / 2
        return 0

    def add_target(self, x, y, vel=0, acc=0, theta=0):
        target = Target(x, y, vel, acc, theta)
        self.targets.append(target)

    def add_radar(self, pt=1000, gt=30, s_min=1e-10, beamwidth=10, irradPattern=None, theta=0, rpm=1, clockwise=False, deg_step=0.1):
        self.radar = Radar(self.r_max, pt, gt, s_min, beamwidth, irradPattern, theta=theta, rpm=rpm, clockwise=clockwise, deg_step=deg_step)

    def add_orbital_target(self, r, speed, acceleration=0, clockwise=False, alpha_start=0):
        target = OrbitalTarget(r, speed, acceleration, clockwise, alpha_start)
        self.targets.append(target)

    def add_nested_orbital_target(self, r1, speed1, acc1, r2, speed2, acc2, clockwise1=False, clockwise2=False, alpha1_start=0, alpha2_start=0):
        target = NestedOrbitalTarget(r1, speed1, acc1, r2, speed2, acc2, clockwise1, clockwise2, alpha1_start, alpha2_start)
        self.targets.append(target)

    def add_clutter():
        pass

    # 
    # Radar equation helpers
    # 

    _WAVELENGTH_M   = 0.03   # λ default: 10 GHz (banda X) em metros
    _RCS_DEFAULT_M2 = 1.0    # σ default: 1 m²  (alvo genérico)
    _4PI3           = (4 * np.pi) ** 3

    def _antenna_gain_linear(self, deg_error: float) -> float:
        """
        Retorna o ganho linear G(θ) da antena para um dado erro angular.

        Se o radar possui um ``irradPattern`` callable, ele é chamado com
        ``deg_error`` (em graus) e seu resultado é interpretado como ganho
        **linear** (não em dB).

        Caso contrário, aplica um modelo Gaussiano calibrado pela beamwidth:
            G(θ) = G_max · exp(-θ² / (2σ²))
        onde σ = beamwidth / (2 · √(2 · ln(2)))  ≈  beamwidth / 2.355
        (HPBW half-power convention).

        Args:
            deg_error: Erro angular em relação ao centro do feixe (graus).

        Returns:
            Ganho linear (adimensional, ≥ 0).
        """
        radar = self.radar
        G_max_linear = 10 ** (radar.gt / 10)   # dBi → linear

        if callable(radar.irradPattern):
            return float(radar.irradPattern(deg_error))

        # Modelo Gaussiano: σ tal que G(bw/2) = G_max/2  (−3 dB)
        bw_half = radar.beamwidth / 2.0
        if bw_half == 0:
            return G_max_linear
        sigma = bw_half / np.sqrt(2 * np.log(2))   # sigma em graus
        return G_max_linear * np.exp(-(deg_error ** 2) / (2 * sigma ** 2))

    def _calc_rx_power_dbm(self, range_m: float, deg_error: float) -> float:
        """
        Calcula a potência recebida (dBm) via equação do radar:

            P_rx = (Pt · G_tx(θ) · G_rx(θ) · λ² · σ) / ((4π)³ · R⁴)

        Para este modelo, G_tx = G_rx = G(θ) (radar monoestático com mesma
        antena para TX e RX, degradada pelo deg_error).

        Args:
            range_m:   Distância ao alvo (m).
            deg_error: Erro angular em relação ao centro do feixe (graus).

        Returns:
            Potência recebida em dBm.
        """
        if range_m <= 0:
            return -200.0

        G  = self._antenna_gain_linear(deg_error)
        Pt = self.radar.pt              # Watts
        lam = self._WAVELENGTH_M
        sigma = self._RCS_DEFAULT_M2

        P_rx_w = (Pt * G**2 * lam**2 * sigma) / (self._4PI3 * range_m**4)
        P_rx_w = max(P_rx_w, 1e-30)    # evita log(0)
        return 10 * np.log10(P_rx_w * 1e3)   # W → dBm

    # 

    def update(self) -> list[DetectionRecord]:
        """
        Avança um passo de simulação.

        Retorna:
            Lista de DetectionRecord para os targets dentro do feixe neste passo.
            Também acumula os registros em self.detection_log.
        """
        # Conta steps inteiros para evitar drift de ponto flutuante
        self._step += 1
        self.elapsed_time = round(self._step * self.dt, 10)
        detections: list[DetectionRecord] = []

        for i, target in enumerate(self.targets):
            target.update(self.dt)

            if self.radar:
                # Cálculo de range e azimute do target
                r     = np.sqrt(target.x**2 + target.y**2)
                alpha = np.degrees(np.arctan2(target.y, target.x)) % 360

                # Limites do feixe normalizados para [0, 360)
                l = self.theta_low  % 360
                h = self.theta_high % 360

                # Verifica se o target está dentro do feixe (com wrap-around em 360)
                in_beam = False
                if l <= h:
                    in_beam = (l <= alpha <= h)
                else:  # caso de wrap-around (ex.: low=355, high=5)
                    in_beam = (alpha >= l or alpha <= h)

                if in_beam:
                    # Erro angular em graus em relação ao centro do feixe,
                    # quantizado pelo passo angular do radar (deg_step).
                    # diff ∈ (-bw/2, +bw/2]; 0 = centro exato.
                    radar_az  = self.radar.theta % 360
                    diff      = ((alpha - radar_az + 180) % 360) - 180  # graus
                    deg_step  = self.radar.deg_step
                    deg_error = round(diff / deg_step) * deg_step

                    # Potência RX via equação do radar
                    rx_power_dbm = self._calc_rx_power_dbm(r, deg_error)

                    record = DetectionRecord(
                        time         = self.elapsed_time,
                        target_idx   = i,
                        range_m      = r,
                        azimuth_deg  = alpha,
                        deg_error    = deg_error,
                        rx_power_dbm = rx_power_dbm,
                    )
                    detections.append(record)
                    self.detection_log.add(record)

                # Eventos de entrada/saída do feixe
                was_in_beam = getattr(target, 'in_beam', False)
                if in_beam and not was_in_beam:
                    print(f"[{self.elapsed_time:.3f}s] Target ENTER: R={r:.2f}, Az={alpha:.2f}°")
                elif not in_beam and was_in_beam:
                    print(f"[{self.elapsed_time:.3f}s] Target EXIT:  R={r:.2f}, Az={alpha:.2f}°")
                target.in_beam = in_beam

        if self.radar:
            self.radar.update(self.dt)

        return detections