import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from .wave import Wavefront
from .radar import Radar

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
    wavefronts = []
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
            wavefronts.append({"wf": wf_new, "R": 1.0})

        segments = []
        powers_db = []
        active_wavefronts = []

        for wf_data in wavefronts:
            wf = wf_data["wf"]
            R = wf_data["R"]

            # Atualiza a potência baseada na distância R
            wf.update(R)

            # Se ainda houver componentes acima do threshold, processa para desenho
            if wf.arcs:
                active_wavefronts.append(wf_data)

                for arc in wf.arcs:
                    # Usa res_deg (definido no __init__ da GainPattern)
                    theta = wf.gain_pattern.theta_vec[arc.idx]
                    dtheta = wf.gain_pattern.res_deg 

                    # Cria o arco infinitesimal para o LineCollection
                    theta_segment = np.array([
                        theta - dtheta/2,
                        theta + dtheta/2
                    ])

                    x = R * np.cos(theta_segment)
                    y = R * np.sin(theta_segment)

                    segments.append(np.column_stack((x, y)))
                    powers_db.append(arc.power_db)

                # Propaga o pulso para o próximo frame
                wf_data["R"] += dR

        wavefronts = active_wavefronts

        if collection:
            collection.remove()

        if not segments:
            return []

        # Normalização das cores baseada no threshold do usuário
        powers_db = np.array(powers_db)
        norm = np.clip(
            (powers_db - wavefront_template.threshold_db) /
            (0 - wavefront_template.threshold_db),
            0, 1
        )

        colors = cm.turbo(norm)
        collection = LineCollection(segments, colors=colors, linewidths=2)
        ax.add_collection(collection)

        ax.set_title(f"Pulsos ativos: {len(wavefronts)} | R_max: {R:.1f}m", color="white")
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
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_facecolor("black")

    max_R = np.max(R_values)

    ax.set_xlim(-max_R * 1.1, max_R * 1.1)
    ax.set_ylim(-max_R * 1.1, max_R * 1.1)

    ax.plot(0, 0, 'wo')

    collection = None
    beam_line = None

    waves = []
    pulse_timer = 0
    dR = R_values[1] - R_values[0]

    norm = plt.Normalize(vmin=radar.smin_db, vmax=0)
    sm = cm.ScalarMappable(cmap=cm.turbo, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Power (dBW)")

    def update(frame_idx):
        nonlocal collection, beam_line, waves, pulse_timer

        radar.phi = (radar.phi + radar.ang_vel) % 360
        phi_rad = np.deg2rad(radar.phi)

        pulse_timer += 1
        if pulse_timer >= pulse_interval:
            pulse_timer = 0

            wf_new = Wavefront(
                radar.Pw,
                radar.gain,
                radar.smin_db,
                radar.phi
            )

            waves.append({
                "wf": wf_new,
                "R": 1.0
            })

        segments = []
        powers_db = []
        active_waves = []

        for w in waves:
            wf = w["wf"]
            R = w["R"]

            wf.update(R)

            if wf.arcs:
                active_waves.append(w)

            for arc in wf.arcs:
                # Garante que usamos o ângulo correto do vetor original do padrão de ganho
                theta = wf.gain_pattern.theta_vec[arc.idx]
                
                # CORREÇÃO: Use o nome exato definido no __init__ da GainPattern (res_deg)
                dtheta = wf.gain_pattern.res_deg 
                theta_segment = np.array([
                    theta - dtheta/2,
                    theta + dtheta/2
                ])
                # O cálculo de x e y permanece o mesmo
                x = R * np.cos(theta_segment)
                y = R * np.sin(theta_segment)
                segments.append(np.column_stack((x, y)))
                # O power_db já foi calculado no passo anterior wf.update(R)
                powers_db.append(arc.power_db)
                # 🔹 propaga onda
                w["R"] += dR

        waves = active_waves


        if collection:
            collection.remove()

        if segments:
            powers_db = np.array(powers_db)

            norm = np.clip(
                (powers_db - radar.smin_db) / (0 - radar.smin_db),
                0, 1
            )

            colors = cm.turbo(norm)

            collection = LineCollection(segments, colors=colors, linewidths=2)
            ax.add_collection(collection)


        if beam_line:
            beam_line.remove()

        beam_x = [radar.x, radar.x + max_R * np.cos(phi_rad)]
        beam_y = [radar.y, radar.y + max_R * np.sin(phi_rad)]

        beam_line, = ax.plot(beam_x, beam_y, 'w-', linewidth=1)

        artists = []
        if collection:
            artists.append(collection)
        if beam_line:
            artists.append(beam_line)

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(R_values),
        interval=50,
        blit=True
    )

    plt.show()