import sys
import argparse
import numpy as np

from radarutils.simulator.ppi import PPI

class Simulator:
    """
    Controla a simulação do radar PPI.

    Todos os parâmetros de configuração (radar, targets, dimensões, tempo)
    vivem aqui. A simulação pode rodar de forma headless (sem interface
    gráfica) ou com a janela Qt, produzindo os mesmos resultados.
    """

    def __init__(
        self,
        dimensions: tuple[int, int] = (2000, 2000),
        dt: float = 0.03,
        t: float = 10.0,
        r_max: float = 1000.0,
    ):
        """
        Args:
            dimensions: Dimensões do espaço de simulação em metros (largura, altura).
            dt:         Passo de tempo em segundos.
            t:          Duração total da simulação em segundos.
            r_max:      Alcance máximo do radar em metros.
        """
        self.ppi = PPI(dimensions=dimensions, dt=dt, t=t)
        self.ppi.r_max = r_max

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.ppi.dimensions

    @property
    def dt(self) -> float:
        return self.ppi.dt

    @property
    def t(self) -> float:
        return self.ppi.t

    @property
    def r_max(self) -> float:
        return self.ppi.r_max

    @property
    def elapsed_time(self) -> float:
        return self.ppi.elapsed_time

    def add_radar(
        self,
        theta: float = 0,
        rpm: float = 15,
        clockwise: bool = True,
        pt: float = 1000,
        gt: float = 30,
        s_min: float = 1e-10,
        beamwidth: float = 10,
        deg_step: float = 0.1,
        irradPattern=None,
    ):
        """Adiciona (e configura) o radar ao PPI."""
        self.ppi.add_radar(
            pt=pt,
            gt=gt,
            s_min=s_min,
            beamwidth=beamwidth,
            irradPattern=irradPattern,
            theta=theta,
            rpm=rpm,
            clockwise=clockwise,
            deg_step=deg_step,
        )

    def add_target(self, x: float, y: float, vel: float = 0, acc: float = 0, theta: float = 0):
        """Adiciona um target cartesiano estático ou com movimento linear."""
        self.ppi.add_target(x, y, vel, acc, theta)

    def add_orbital_target(
        self,
        r: float,
        speed: float,
        acceleration: float = 0,
        clockwise: bool = False,
        alpha_start: float = 0,
    ):
        """Adiciona um target com movimento orbital circular."""
        self.ppi.add_orbital_target(r, speed, acceleration, clockwise, alpha_start)

    def add_nested_orbital_target(
        self,
        r1: float,
        speed1: float,
        acc1: float,
        r2: float,
        speed2: float,
        acc2: float,
        clockwise1: bool = False,
        clockwise2: bool = False,
        alpha1_start: float = 0,
        alpha2_start: float = 0,
    ):
        """Adiciona um target com movimento epicíclico (órbita dentro de órbita)."""
        self.ppi.add_nested_orbital_target(
            r1, speed1, acc1, r2, speed2, acc2,
            clockwise1, clockwise2,
            alpha1_start, alpha2_start,
        )

    def run(self, gui: bool = True, show_vectors: bool = False, output_file: str = None):
        """
        Executa a simulação.

        Args:
            gui:     Se True, roda sem interface gráfica (apenas terminal).
                          Se False, abre a janela Qt com PPI + Detection Plot.
            show_vectors: (modo com tela) Exibe vetores de velocidade dos targets.
            output_file:  (modo com tela) Caminho para salvar o vídeo MP4. None = sem gravação.
        """
        if gui:
            self._run_headless()
        else:
            self._qt_exit_code = self._run_with_screen(
                show_vectors=show_vectors, output_file=output_file
            )

    @property
    def detection_log(self):
        """Acesso direto ao DetectionLog acumulado pelo PPI."""
        return self.ppi.detection_log

    def export(self, path: str = "detections.csv") -> str:
        """
        Exporta todas as detecções acumuladas para um arquivo CSV.

        Args:
            path: Caminho do arquivo de saída.

        Returns:
            Caminho resolvido do arquivo criado.
        """
        output = self.ppi.detection_log.export(path)
        print(f"Detections exported to: {output}  ({len(self.ppi.detection_log)} records)")
        return str(output)

    def _run_headless(self):
        """Loop de simulação sem interface gráfica."""
        print("=== Simulator running (headless) ===")
        while self.ppi.elapsed_time < self.ppi.t:
            self.ppi.update()
        print(f"=== Simulation finished at t={self.ppi.elapsed_time:.2f}s ===")

    def _run_with_screen(self, show_vectors: bool = False, output_file: str = None) -> int:
        """Abre a janela Qt e executa o loop de eventos. Retorna o exit code."""
        import pyqtgraph as pg
        from PySide6 import QtWidgets
        from radarutils.simulator.screen import MainWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(antialias=True)

        window = MainWindow(
            ppi=self.ppi,
            show_vectors=show_vectors,
            output_file=output_file,
        )
        window.show()
        return app.exec()


def _build_default_simulator() -> Simulator:
    """Cria o simulador com a configuração padrão de demonstração."""
    sim = Simulator(dimensions=(2000, 2000), dt=0.01, t=120.0, r_max=1000.0)

    sim.add_radar(theta=0, rpm=8, clockwise=True)

    # Targets lineares / estáticos
    sim.add_target(x=200,  y=200,  vel=0, acc=0, theta=0)
    sim.add_target(x=-200, y=-200, vel=0, acc=0, theta=0)

    # Targets orbitais simples
    sim.add_orbital_target(r=400,  speed=60,  clockwise=True,  alpha_start=0)
    sim.add_orbital_target(r=200,  speed=60,  clockwise=False, alpha_start=np.pi)

    # Targets epicíclicos (nested orbital)
    sim.add_nested_orbital_target(
        r1=700, speed1=80,  acc1=0,
        r2=200, speed2=100, acc2=0,
        clockwise1=True,  clockwise2=False,
        alpha1_start=0,     alpha2_start=0,
    )
    sim.add_nested_orbital_target(
        r1=200, speed1=20,  acc1=0,
        r2=440, speed2=120, acc2=0,
        clockwise1=False, clockwise2=True,
        alpha1_start=np.pi / 2, alpha2_start=0,
    )

    return sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Radar PPI Simulator')
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Roda a simulação sem interface gráfica (apenas terminal)',
    )
    parser.add_argument(
        '--no-vectors',
        action='store_true',
        help='Desativa vetores de velocidade na tela',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Caminho para salvar o vídeo MP4 (ex: simulation.mp4). Padrão: sem gravação.',
    )
    parser.add_argument(
        '--export',
        type=str,
        default=None,
        metavar='CSV',
        help='Exporta detecções para CSV após a simulação (ex: detections.csv).',
    )
    args = parser.parse_args()

    simulator = _build_default_simulator()

    if args.output and not args.gui:
        from radarutils.simulator.screen import prepare_output_file
        output_path = prepare_output_file(args.output)
    else:
        output_path = None

    simulator.run(
        gui=args.gui,
        show_vectors=not args.no_vectors,
        output_file=output_path,
    )

    if args.export:
        simulator.export(args.export)

    sys.exit(getattr(simulator, '_qt_exit_code', 0))
