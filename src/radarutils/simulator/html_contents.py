"""
html_contents.py — Templates HTML para os widgets de visualização do simulador.

Centraliza todo o conteúdo de UI (textos, tabelas, overlays) escrito em HTML
utilizado pelo PyQtGraph no Radar Simulator, mantendo os módulos lógicos
livres de strings de markup.

Funções e constantes exportadas:

    get_pulse_header_html(...)              → cabeçalho de parâmetros do PulseWidget
    PPI_REAL_LABEL_HTML                     → rótulo do PPI Real
    get_ppi_real_angle_html(angle_deg)      → rótulos de ângulo no grid do PPI Real
    PPI_ESTIMATED_LABEL_HTML                → rótulo do PPI Estimado
    PPI_ESTIMATED_INITIAL_LEGEND_HTML       → legenda inicial do PPI Estimado
    get_ppi_est_angle_html(angle_deg)       → rótulos de ângulo no grid do PPI Estimado
    get_ppi_est_legend_html(fa, det, vrs)   → caixa de status FAR/DET + velocidades
"""

from typing import List


# ══════════════════════════════════════════════════════════════════════════
#  PulseWidget — Painel de parâmetros do radar (cabeçalho central)
# ══════════════════════════════════════════════════════════════════════════

def get_pulse_header_html(
    PRI_us:       float,
    T_us:         float,
    F_C_GHz:      float,
    B_MHz:        float,
    snr_db:       float,
    c_str:        str,
    r_min:        float,
    r_max:        float,
    bw:           float,
    int_mode_str: str,
    c_time:       float,
    t_total:      float,
) -> str:
    """
    Retorna o HTML da tabela do painel superior central (PulseWidget).

    Args:
        PRI_us:       Período de repetição de pulso em µs.
        T_us:         Duração do pulso em µs.
        F_C_GHz:      Frequência de portadora em GHz.
        B_MHz:        Largura de banda em MHz.
        snr_db:       SNR estimado em dB.
        c_str:        Tipo de clutter (string descritiva).
        r_min:        Alcance mínimo do radar (m).
        r_max:        Alcance máximo do radar (m).
        bw:           Largura de feixe em graus.
        int_mode_str: Modo de integração (string descritiva).
        c_time:       Tempo atual da simulação (s).
        t_total:      Duração total da simulação (s).

    Returns:
        String HTML pronta para uso em ``QLabel`` ou ``pg.TextItem``.
    """
    return (
        f'<div align="center">'
        f'<table align="center" cellpadding="3" cellspacing="0" '
        f'style="font-family: Consolas; font-size:10pt; color: #DDDDDD;">'
        # ── Título ──────────────────────────────────────────────────────
        f'<tr><td colspan="8" align="center" '
        f'style="font-size:14pt; color: #00C8FF; font-weight: bold; padding-bottom: 8px;">'
        f'RADAR SIMULATION PARAMETERS</td></tr>'
        # ── Linha 1: PRI | Tp | Fc | BW ─────────────────────────────────
        f'<tr>'
        f'<td align="right" style="color:#88CCFF; padding-right:4px;">PRI:</td>'
        f'<td align="left" style="font-weight:bold; padding-right:15px;">{PRI_us:.2f} µs</td>'
        f'<td align="right" style="color:#88CCFF; padding-right:4px;">Tp:</td>'
        f'<td align="left" style="font-weight:bold; padding-right:15px;">{T_us:.2f} µs</td>'
        f'<td align="right" style="color:#88CCFF; padding-right:4px;">Fc:</td>'
        f'<td align="left" style="font-weight:bold; padding-right:15px;">{F_C_GHz:.0f} GHz</td>'
        f'<td align="right" style="color:#88CCFF; padding-right:4px;">BW:</td>'
        f'<td align="left" style="font-weight:bold;">{B_MHz:.0f} MHz</td>'
        f'</tr>'
        # ── Linha 2: SNR | Clutter | R_min | R_max ───────────────────────
        f'<tr>'
        f'<td align="right" style="color:#88CCFF;">SNR:</td>'
        f'<td align="left" style="font-weight:bold;">{snr_db:.0f} dB</td>'
        f'<td align="right" style="color:#88CCFF;">Clutter:</td>'
        f'<td align="left" style="font-weight:bold;">{c_str}</td>'
        f'<td align="right" style="color:#88CCFF;">R_min:</td>'
        f'<td align="left" style="font-weight:bold;">{r_min:.1f} m</td>'
        f'<td align="right" style="color:#88CCFF;">R_max:</td>'
        f'<td align="left" style="font-weight:bold;">{r_max:.1f} m</td>'
        f'</tr>'
        # ── Linha 3: Beam | Int. Mode | T_now | T_max ────────────────────
        f'<tr>'
        f'<td align="right" style="color:#88CCFF;">Beam:</td>'
        f'<td align="left" style="font-weight:bold;">{bw:.1f}&deg;</td>'
        f'<td align="right" style="color:#88CCFF;">Int. Mode:</td>'
        f'<td align="left" style="font-weight:bold;">{int_mode_str}</td>'
        f'<td align="right" style="color:#88CCFF;">T_now:</td>'
        f'<td align="left" style="font-weight:bold;">{c_time:.2f} s</td>'
        f'<td align="right" style="color:#88CCFF; padding-right:4px;">T_max:</td>'
        f'<td align="left" style="font-weight:bold;">{t_total:.1f} s</td>'
        f'</tr>'
        f'</table>'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════════════
#  PPIViewer — PPI Real
# ══════════════════════════════════════════════════════════════════════════

PPI_REAL_LABEL_HTML = """
<div style="
    font-family: Consolas;
    font-size: 12pt;
    color: #00FF00;
    font-weight: bold;
    background-color: rgba(0,0,0,160);
    padding: 6px;
">
PPI REAL
</div>
"""


def get_ppi_real_angle_html(angle_deg: int) -> str:
    """
    Retorna o HTML para um rótulo de graus no grid do PPI Real.

    Args:
        angle_deg: Ângulo em graus inteiros (0–359).

    Returns:
        String HTML do rótulo.
    """
    return (
        f'<div style="'
        f'color: rgb(0,220,0);'
        f'font-weight: bold;'
        f'font-size: 10pt;'
        f'font-family: Consolas;'
        f'">'
        f'{angle_deg}&deg;'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════════════
#  PPIEstimatedViewer — PPI Estimado (Tracker)
# ══════════════════════════════════════════════════════════════════════════

PPI_ESTIMATED_LABEL_HTML = """
<div style="
    font-family: Consolas;
    font-size: 12pt;
    color: #00AAFF;
    font-weight: bold;
    background-color: rgba(0,0,0,160);
    padding: 6px;
">
PPI ESTIMADO
</div>
"""

PPI_ESTIMATED_INITIAL_LEGEND_HTML = """
<div style="
    font-family: Consolas;
    font-size: 9pt;
    background-color: rgba(0,0,0,160);
    padding: 6px;
">
<span style="color:#FFDD00;">&#11044; FAR Count: 0</span><br/>
<span style="color:#FF3333;">&#11044; DET Count: 0</span><br/>
<span style="color:#FF3333;">&#11044;</span>
<span style="color:#DDDDDD;"> V_r: <b>+0.0 m/s</b></span>
</div>
"""


def get_ppi_est_angle_html(angle_deg: int) -> str:
    """
    Retorna o HTML para um rótulo de graus no grid do PPI Estimado.

    Args:
        angle_deg: Ângulo em graus inteiros (0–359).

    Returns:
        String HTML do rótulo.
    """
    return (
        f'<div style="'
        f'color: rgb(0,140,200);'
        f'font-weight: bold;'
        f'font-size: 10pt;'
        f'font-family: Consolas;'
        f'">'
        f'{angle_deg}&deg;'
        f'</div>'
    )


def get_ppi_est_legend_html(
    total_fa:   int,
    total_true: int,
    vrs:        List[float],
) -> str:
    """
    Retorna o HTML da caixa de status do PPI Estimado.

    Exibe contadores de FAR/DET e a lista de velocidades radiais estimadas
    das últimas detecções verdadeiras.

    Args:
        total_fa:   Contagem total de Falsos Alarmes.
        total_true: Contagem total de Detecções Verdadeiras (True Positives).
        vrs:        Lista de velocidades radiais estimadas (m/s).

    Returns:
        String HTML da caixa de legenda.
    """
    lines = [
        f'<span style="color:#FFDD00;">&#11044; FAR Count: {total_fa}</span>',
        f'<span style="color:#FF3333;">&#11044; DET Count: {total_true}</span>',
    ]
    for vr in vrs:
        lines.append(
            f'<span style="color:#FF3333;">&#11044;</span> '
            f'<span style="color:#DDDDDD;">V_r: <b>{vr:+.1f} m/s</b></span>'
        )

    return (
        '<div style="font-family:Consolas; font-size:10pt;'
        ' background-color:rgba(0,0,0,170); padding:6px;">'
        + '<br/>'.join(lines)
        + '</div>'
    )
