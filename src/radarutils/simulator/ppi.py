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

                    record = DetectionRecord(
                        time        = self.elapsed_time,
                        target_idx  = i,
                        range_m     = r,
                        azimuth_deg = alpha,
                        deg_error   = deg_error,
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