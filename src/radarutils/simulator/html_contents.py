"""
html_contents.py - Templates HTML para os widgets de visualização da simulação.

Este arquivo concentra o conteúdo de UI (textos, tabelas, popups na tela)
escrito em HTML utilizado pelo PyQtGraph no Radar Simulator, mantendo os
arquivos lógicos limpos.
"""

from typing import List


# ──────────────────────────────────────────────────────────────────────────
#  Pulse Modulation Widget (Header Panel)
# ──────────────────────────────────────────────────────────────────────────

def get_pulse_header_html(
    PRI_us: float,
    T_us: float,
    F_C_GHz: float,
    B_MHz: float,
    snr_db: float,
    c_str: str,
    r_min: float,
    r_max: float,
    bw: float,
    int_mode_str: str,
    c_time: float,
    t_total: float,
) -> str:
    """Retorna o HTML da tabela do painel superior central (PulseWidget)."""
    return (
        f'<div align="center">'
        f'<table align="center" cellpadding="3" cellspacing="0" '
        f'style="font-family: Consolas; font-size:10pt; color: #DDDDDD;">'
        f'<tr><td colspan="8" align="center" '
        f'style="font-size:14pt; color: #00C8FF; font-weight: bold; padding-bottom: 8px;">'
        f'RADAR SIMULATION PARAMETERS</td></tr>'
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


# ──────────────────────────────────────────────────────────────────────────
#  PPI Viewer (Real Viewer)
# ──────────────────────────────────────────────────────────────────────────

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
    """HTML para os rótulos de graus no grid do PPI Real."""
    return f"""
    <div style="
        color: rgb(0,220,0);
        font-weight: bold;
        font-size: 10pt;
        font-family: Consolas;
    ">
    {angle_deg}&deg;
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────
#  PPI Estimated Viewer (Tracker Viewer)
# ──────────────────────────────────────────────────────────────────────────

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
    """HTML para os rótulos de graus no grid do PPI Estimado."""
    return f"""
    <div style="
        color: rgb(0,140,200);
        font-weight: bold;
        font-size: 10pt;
        font-family: Consolas;
    ">
    {angle_deg}&deg;
    </div>
    """

def get_ppi_est_legend_html(total_fa: int, total_true: int, vrs: List[float]) -> str:
    """HTML da caixa de status (FAR/DET Count e velocidades radiais) do PPI Estimado."""
    legend_lines = [
        f'<span style="color:#FFDD00;">&#11044; FAR Count: {total_fa}</span>',
        f'<span style="color:#FF3333;">&#11044; DET Count: {total_true}</span>',
    ]
    for vr in vrs:
        legend_lines.append(
            f'<span style="color:#FF3333;">&#11044;</span> '
            f'<span style="color:#DDDDDD;">V_r: <b>{vr:+.1f} m/s</b></span>'
        )
    return (
        '<div style="font-family:Consolas; font-size:10pt;'
        ' background-color:rgba(0,0,0,170); padding:6px;">'
        + '<br/>'.join(legend_lines)
        + '</div>'
    )
