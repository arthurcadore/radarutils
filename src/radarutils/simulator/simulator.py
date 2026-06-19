import sys
import os
import argparse
import numpy as np

from radarutils.simulator.ppi import PPI
from radarutils.core.clutter import VALID_CLUTTER_TYPES
from radarutils.core.integrator import VALID_INTEGRATOR_TYPES

class Simulator:
    r"""
    Controla a simulação do radar PPI.

    Coleta parâmetros de configuração e repassa ao RadarPipeline, que
    orquestra a geração de dados e instancia o PPI.
    """

    def __init__(
        self,
        dimensions: tuple[int, int] = (2000, 2000),
        dt: float = 0.03,
        t: float = 10.0,
        r_max: float = 1000.0,
    ):
        self.config = {
            'dimensions': dimensions,
            'dt': dt,
            't': t,
            'r_max': r_max,
            'radar': None,
            'targets': [],
            'orbital_targets': [],
            'nested_orbital_targets': [],
            'regional_clutter': [],
        }

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.config['dimensions']

    @property
    def dt(self) -> float:
        return self.config['dt']

    @property
    def t(self) -> float:
        return self.config['t']

    @property
    def r_max(self) -> float:
        return self.config['r_max']

    def add_radar(self, **kwargs):
        self.config['radar'] = kwargs

    def add_target(self, **kwargs):
        self.config['targets'].append(kwargs)

    def add_orbital_target(self, **kwargs):
        self.config['orbital_targets'].append(kwargs)

    def add_nested_orbital_target(self, **kwargs):
        self.config['nested_orbital_targets'].append(kwargs)

    def add_regional_clutter(self, **kwargs) -> None:
        self.config['regional_clutter'].append(kwargs)

    def run(self, gui: bool = True, show_vectors: bool = False, output_file: str = None,
            integrator_type: str = "noncoherent", clutter_type: str = "none", normalize_plots: bool = True,
            max_video_mb: float = None, video_quality: int = 8):
        self.config['integrator_type'] = integrator_type
        self.config['clutter_type'] = clutter_type
        self.config['normalize_plots'] = normalize_plots
        
        from radarutils.simulator.pipeline import RadarPipeline
        self.pipeline = RadarPipeline(self.config)

        if gui:
            self._run_headless()
        else:
            self._qt_exit_code = self._run_with_screen(
                show_vectors=show_vectors,
                output_file=output_file,
                max_video_mb=max_video_mb,
                video_quality=video_quality,
            )

    @property
    def detection_log(self):
        return self.pipeline.ppi.detection_log

    def export(self, path: str = "detections.csv") -> str:
        output = self.pipeline.ppi.detection_log.export(path)
        print(f"Detections exported to: {output}  ({len(self.pipeline.ppi.detection_log)} records)")
        return str(output)

    def _run_headless(self):
        print("=== Simulator running (headless) ===")
        while self.pipeline.ppi.elapsed_time < self.pipeline.ppi.t:
            self.pipeline.run_step()
        print(f"=== Simulation finished at t={self.pipeline.ppi.elapsed_time:.2f}s ===")

    def _run_with_screen(self, show_vectors: bool = False, output_file: str = None,
                         max_video_mb: float = None, video_quality: int = 8) -> int:
        r"""Abre a janela Qt e executa o loop de eventos. Retorna o exit code."""
        import pyqtgraph as pg
        from PySide6 import QtWidgets
        from radarutils.simulator.screen import MainWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(antialias=True)

        window = MainWindow(
            pipeline=self.pipeline,
            show_vectors=show_vectors,
            output_file=output_file,
            max_video_mb=max_video_mb,
            video_quality=video_quality,
        )
        window.show()
        return app.exec()


def _build_default_simulator() -> Simulator:
    """Cria o simulador com a configuração padrão de demonstração."""
    sim = Simulator(dimensions=(2400, 2400), dt=0.03, t=180.0, r_max=1200.0)

    sim.add_radar(theta=0, rpm=5, clockwise=True, beamwidth=10)

    # Targets lineares / estáticos
    sim.add_target(x=-600, y=-600, vel=0, acc=0, theta=0)

    # Targets orbitais simples
    sim.add_orbital_target(r=900,  speed=60,  clockwise=False,  alpha_start=np.pi)
    sim.add_orbital_target(r=600,  speed=60,  clockwise=False, alpha_start=np.pi/2)

    sim.add_nested_orbital_target(
        r1=220, speed1=95,  acc1=1,
        r2=520, speed2=130, acc2=-1,
        clockwise1=False, clockwise2=True,
        alpha1_start=np.pi, alpha2_start=0,
    )
    
    sim.add_nested_orbital_target(
        r1=250, speed1=140, acc1=-2,
        r2=600, speed2=85,  acc2=1,
        clockwise1=True, clockwise2=False,
        alpha1_start=np.pi, alpha2_start=np.pi/4,
    )
    
    sim.add_nested_orbital_target(
        r1=220, speed1=110, acc1=0,
        r2=500, speed2=90,  acc2=2,
        clockwise1=False, clockwise2=False,
        alpha1_start=np.pi/2, alpha2_start=3*np.pi/2,
    )
    
    sim.add_nested_orbital_target(
        r1=280, speed1=75,  acc1=3,
        r2=660, speed2=145, acc2=-2,
        clockwise1=True, clockwise2=True,
        alpha1_start=5*np.pi/6, alpha2_start=np.pi/2,
    )
    
    sim.add_nested_orbital_target(
        r1=250, speed1=125, acc1=-1,
        r2=500, speed2=115, acc2=1,
        clockwise1=False, clockwise2=True,
        alpha1_start=7*np.pi/6, alpha2_start=np.pi,
    )
    
    sim.add_nested_orbital_target(
        r1=200, speed1=160, acc1=2,
        r2=550, speed2=70, acc2=0,
        clockwise1=True, clockwise2=False,
        alpha1_start=np.pi/8, alpha2_start=11*np.pi/8,
    )

    sim.add_regional_clutter(
        x=800, y=-400, radius=150,  intensity=1e-3,
        distribution="weibull", shape=1.6
    )
    sim.add_regional_clutter(
        x=-400, y=-400, radius=100, intensity=1e-3,
        distribution="weibull", shape=1.8
    )

    sim.add_regional_clutter(
        x=-900, y=+500, radius=100, intensity=1e-3,
        distribution="weibull", shape=1.6
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
        '--integrator',
        type=lambda s: s.lower().replace('-', '').replace('_', ''),
        default='coherent',
        choices=VALID_INTEGRATOR_TYPES,
        metavar='{' + '|'.join(VALID_INTEGRATOR_TYPES) + '}',
        help='Tipo de integrador de pulso. Opções: ' + ', '.join(f"'{v}'" for v in VALID_INTEGRATOR_TYPES),
    )
    parser.add_argument(
        '--no-vectors',
        action='store_true',
        help='Desativa vetores de velocidade na tela',
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Desativa a normalização (escala 0 a 1) dos gráficos de processamento',
    )
    parser.add_argument(
        '--clutter',
        type=lambda s: s.lower(),
        default='none',
        choices=VALID_CLUTTER_TYPES,
        metavar='{' + '|'.join(VALID_CLUTTER_TYPES) + '}',
        help='Tipo de clutter ambiente. Opções: ' + ', '.join(f"'{v}'" for v in VALID_CLUTTER_TYPES),
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Caminho para salvar o vídeo MP4 (ex: simulation.mp4). Padrão: sem gravação.',
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=8,
        help='Qualidade do vídeo (0 a 10). Padrão: 8. Valores menores diminuem o tamanho do arquivo.',
    )
    parser.add_argument(
        '--max-mb',
        type=float,
        default=None,
        help='Tamanho máximo do vídeo exportado em MB. A simulação encerra quando atingir.',
    )
    parser.add_argument(
        '--export',
        type=str,
        default=None,
        metavar='CSV',
        help='Exporta detecções para CSV após a simulação (ex: detections.csv).',
    )
    args = parser.parse_args()

    # Se estiver rodando sem tela (ex: servidor Linux sem X11) para exportar vídeo
    if not os.environ.get('DISPLAY') and sys.platform.startswith('linux') and not args.gui:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'

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
        integrator_type=args.integrator,
        clutter_type=args.clutter,
        normalize_plots=not args.no_normalize,
        max_video_mb=args.max_mb,
        video_quality=args.quality,
    )

    if args.export:
        simulator.export(args.export)

    sys.exit(getattr(simulator, '_qt_exit_code', 0))
