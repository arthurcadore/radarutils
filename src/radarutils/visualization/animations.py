import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from ..radar_components.wave import Wavefront
from ..radar_components.radar import Radar

def animate_wavefront(wavefront_template: Wavefront, R_values, pulse_interval=10):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_facecolor("black")

    max_R = np.max(R_values)
    ax.set_xlim(-max_R * 1.1, max_R * 1.1)
    ax.set_ylim(-max_R * 1.1, max_R * 1.1)

    # Representação do Radar no centro
    ax.plot(0, 0, 'wo', markersize=4)

    collection = None
    wavefronts = []  # Agora armazena diretamente os objetos Wavefront
    pulse_timer = 0
    
    # Cálculo do passo de distância (dR)
    dR = R_values[1] - R_values[0] if len(R_values) > 1 else 1.0

    def update(frame_idx):
        nonlocal collection, wavefronts, pulse_timer

        pulse_timer += 1

        # Lógica de disparo de novos pulsos
        if pulse_timer >= pulse_interval:
            pulse_timer = 0
            wf_new = Wavefront(
                wavefront_template.Pw,
                wavefront_template.gain_pattern,
                wavefront_template.threshold_db,
                wavefront_template.phi
            )
            # Inicializa o raio inicial (pode ser 0 ou dR conforme sua lógica física)
            wf_new.R = 1.0 
            wavefronts.append(wf_new)

        segments = []
        powers_db = []
        active_wavefronts = []

        # Variável para o título (R da onda mais distante)
        max_current_r = 0

        for wf in wavefronts:
            # O método update agora incrementa o R interno e filtra os arcos
            wf.update(step=dR)

            # Se ainda houver componentes acima do threshold
            if wf.arcs:
                active_wavefronts.append(wf)
                max_current_r = max(max_current_r, wf.R)

                for arc in wf.arcs:
                    theta = wf.gain_pattern.theta_vec[arc.idx]
                    dtheta = wf.gain_pattern.res_deg 

                    # Cria o arco para o LineCollection usando o R do objeto
                    theta_segment = np.array([
                        theta - dtheta/2,
                        theta + dtheta/2
                    ])

                    # Usa wf.R que foi atualizado no wf.update()
                    x = wf.R * np.cos(theta_segment)
                    y = wf.R * np.sin(theta_segment)

                    segments.append(np.column_stack((x, y)))
                    powers_db.append(arc.power_db)

        wavefronts = active_wavefronts

        if collection:
            collection.remove()

        if not segments:
            ax.set_title("Aguardando pulsos...", color="white")
            return []

        # Normalização das cores baseada no threshold
        powers_db = np.array(powers_db)
        norm = np.clip(
            (powers_db - wavefront_template.threshold_db) /
            (0 - wavefront_template.threshold_db),
            0, 1
        )

        colors = cm.turbo(norm)
        collection = LineCollection(segments, colors=colors, linewidths=2)
        ax.add_collection(collection)

        ax.set_title(f"Pulsos ativos: {len(wavefronts)} | R_max: {max_current_r:.1f}m", color="white")
        return [collection]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(R_values),
        interval=50,
        blit=True
    )

    plt.show()


def animate_radar(radar: Radar, R_values, pulse_interval=10):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_facecolor("black")

    max_R = np.max(R_values)
    ax.set_xlim(-max_R * 1.1, max_R * 1.1)
    ax.set_ylim(-max_R * 1.1, max_R * 1.1)

    ax.plot(radar.x, radar.y, 'wo', markersize=5)

    collection = None
    beam_line = None
    waves = [] 
    pulse_timer = 0
    dR = R_values[1] - R_values[0] if len(R_values) > 1 else 1.0

    # Configuração da barra de cores
    norm_cbar = plt.Normalize(vmin=radar.smin_db, vmax=0)
    sm = cm.ScalarMappable(cmap=cm.turbo, norm=norm_cbar)
    plt.colorbar(sm, ax=ax, label="Power (dBW)")

    def update(frame_idx):
        nonlocal collection, beam_line, waves, pulse_timer

        # Atualiza a rotação do radar
        radar.phi = (radar.phi + radar.ang_vel) % 360
        phi_rad = np.deg2rad(radar.phi)

        pulse_timer += 1
        if pulse_timer >= pulse_interval:
            pulse_timer = 0
            # Cria nova onda na direção atual do radar
            wf_new = Wavefront(radar.Pw, radar.gain, radar.smin_db, radar.phi)
            wf_new.R = 1.0
            waves.append(wf_new)

        segments = []
        powers_db = []
        active_waves = []

        for wf in waves:
            # Chama o update simplificado
            wf.update(step=dR)

            if wf.arcs:
                active_waves.append(wf)
                for arc in wf.arcs:
                    theta = wf.gain_pattern.theta_vec[arc.idx]
                    dtheta = wf.gain_pattern.res_deg 
                    
                    theta_segment = np.array([theta - dtheta/2, theta + dtheta/2])
                    
                    x = radar.x + wf.R * np.cos(theta_segment)
                    y = radar.y + wf.R * np.sin(theta_segment)
                    
                    segments.append(np.column_stack((x, y)))
                    powers_db.append(arc.power_db)

        waves = active_waves

        # Atualiza a linha do feixe principal (beam)
        if beam_line:
            beam_line.remove()
        
        beam_x = [radar.x, radar.x + max_R * np.cos(phi_rad)]
        beam_y = [radar.y, radar.y + max_R * np.sin(phi_rad)]
        beam_line, = ax.plot(beam_x, beam_y, 'w-', alpha=0.3, linewidth=1)

        # Atualiza a coleção de arcos
        if collection:
            collection.remove()

        if segments:
            powers_db = np.array(powers_db)
            norm = np.clip((powers_db - radar.smin_db) / (0 - radar.smin_db), 0, 1)
            colors = cm.turbo(norm)
            collection = LineCollection(segments, colors=colors, linewidths=2)
            ax.add_collection(collection)
        else:
            collection = None

        return [collection, beam_line] if collection else [beam_line]

    anim = FuncAnimation(fig, update, frames=len(R_values), interval=50, blit=True)
    plt.show()