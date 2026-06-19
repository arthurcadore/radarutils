import sys
import os
import argparse
import numpy as np

from radarutils.simulator.ppi import PPI
from radarutils.core.clutter import VALID_CLUTTER_TYPES

class Simulator:
    r"""
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
        r"""
        Constructor methodology that initializes the simulator instance.

        Args:
            dimensions (tuple[int, int]): Dimensões do espaço de simulação em metros (largura, altura).
            dt (float): Passo de tempo em segundos.
            t (float): Duração total da simulação em segundos.
            r_max (float): Alcance máximo do radar em metros.
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
        r"""Adiciona (e configura) o radar ao PPI."""
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
        r"""Adiciona um target cartesiano estático ou com movimento linear."""
        self.ppi.add_target(x, y, vel, acc, theta)

    def add_orbital_target(
        self,
        r: float,
        speed: float,
        acceleration: float = 0,
        clockwise: bool = False,
        alpha_start: float = 0,
    ):
        r"""Adiciona um target com movimento orbital circular."""
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
        r"""Adiciona um target com movimento epicíclico (órbita dentro de órbita)."""
        self.ppi.add_nested_orbital_target(
            r1, speed1, acc1, r2, speed2, acc2,
            clockwise1, clockwise2,
            alpha1_start, alpha2_start,
        )

    def add_regional_clutter(
        self,
        x: float,
        y: float,
        radius: float,
        intensity: float,
        distribution: str = "rayleigh",
        **kwargs,
    ) -> None:
        r"""
        Adiciona uma região circular de clutter ao PPI.

        Args:
            x, y:         Centro da região em metros.
            radius:       Raio da região em metros (> 0).
            intensity:    Amplitude característica do clutter (escala linear).
            distribution: Modelo de amplitude: 'rayleigh', 'rice' ou 'weibull'.
            **kwargs:     Parâmetros extras do modelo (ex.: k_factor, shape).
        """
        self.ppi.add_regional_clutter(x, y, radius, intensity, distribution, **kwargs)

    def run(self, gui: bool = True, show_vectors: bool = False, output_file: str = None,
            coherent_integration: bool = False, clutter_type: str = "None", normalize_plots: bool = True,
            max_video_mb: float = None, video_quality: int = 8):
        r"""
        Executa a simulação.

        Args:
            gui (bool): Se True, roda sem interface gráfica (apenas terminal).
            show_vectors (bool): (modo com tela) Exibe vetores de velocidade dos targets.
            output_file (str): (modo com tela) Caminho para salvar o vídeo MP4. None = sem gravação.
            coherent_integration (bool): Se True, usa integração coerente (soma de amplitudes IQ). Se False (padrão), usa integração não-coerente (soma de potências).
            clutter_type (str): Tipo de clutter a ser aplicado.
            normalize_plots (bool): Se True, normaliza os gráficos da terceira coluna (e Filtro Casado) de 0 a 1.
            max_video_mb (float): Tamanho máximo em MB para o vídeo gerado. A simulação para ao atingir.
            video_quality (int): Qualidade do vídeo (0-10). Padrão: 8.
        """
        if gui:
            self._run_headless()
        else:
            self._qt_exit_code = self._run_with_screen(
                show_vectors=show_vectors,
                output_file=output_file,
                coherent_integration=coherent_integration,
                clutter_type=clutter_type,
                normalize_plots=normalize_plots,
                max_video_mb=max_video_mb,
                video_quality=video_quality,
            )

    @property
    def detection_log(self):
        r"""Acesso direto ao DetectionLog acumulado pelo PPI."""
        return self.ppi.detection_log

    def export(self, path: str = "detections.csv") -> str:
        r"""
        Exporta todas as detecções acumuladas para um arquivo CSV.

        Args:
            path (str): Caminho do arquivo de saída.

        Returns:
            output (str): Caminho resolvido do arquivo criado.
        """
        output = self.ppi.detection_log.export(path)
        print(f"Detections exported to: {output}  ({len(self.ppi.detection_log)} records)")
        return str(output)

    def _run_headless(self):
        r"""Loop de simulação sem interface gráfica."""
        print("=== Simulator running (headless) ===")
        while self.ppi.elapsed_time < self.ppi.t:
            self.ppi.update()  # retorna (detections, active_regional) — ignorado no headless
        print(f"=== Simulation finished at t={self.ppi.elapsed_time:.2f}s ===")

    def _run_with_screen(self, show_vectors: bool = False, output_file: str = None,
                         coherent_integration: bool = False, clutter_type: str = "None", normalize_plots: bool = True,
                         max_video_mb: float = None, video_quality: int = 8) -> int:
        r"""Abre a janela Qt e executa o loop de eventos. Retorna o exit code."""
        import pyqtgraph as pg
        from PySide6 import QtWidgets
        from radarutils.simulator.screen import MainWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(antialias=True)

        window = MainWindow(
            ppi=self.ppi,
            show_vectors=show_vectors,
            output_file=output_file,
            coherent_integration=coherent_integration,
            clutter_type=clutter_type,
            normalize_plots=normalize_plots,
            max_video_mb=max_video_mb,
            video_quality=video_quality,
        )
        window.show()
        return app.exec()


def _build_default_simulator() -> Simulator:
    """Cria o simulador com a configuração padrão de demonstração."""
    sim = Simulator(dimensions=(2400, 2400), dt=0.06, t=180.0, r_max=1200.0)

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
        distribution="rice", k_factor=0.6
    )
    sim.add_regional_clutter(
        x=-400, y=-400, radius=100, intensity=1e-3,
        distribution="weibull", shape=1.8
    )

    sim.add_regional_clutter(
        x=-900, y=+500, radius=100, intensity=1e-3,
        distribution="rayleigh", k_factor=0.6
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
        '--coherent',
        action='store_true',
        help='Usa integração coerente (soma IQ). Padrão: não-coerente (soma de potências).',
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
        coherent_integration=False,
        clutter_type=args.clutter,
        normalize_plots=not args.no_normalize,
        max_video_mb=args.max_mb,
        video_quality=args.quality,
    )

    if args.export:
        simulator.export(args.export)

    sys.exit(getattr(simulator, '_qt_exit_code', 0))
