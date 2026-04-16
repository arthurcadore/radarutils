
from radarutils.core.signal import coherent_radar
from radarutils.visualization.plotter import TimePlot, FrequencyPlot, FrequencyResponsePlot, create_figure, save_figure
from radarutils.core.signal import *
from radarutils.core.probability import NoiseAWGN

import numpy as np

if __name__ == "__main__":
    
    fs = 128000
    t_final = 0.1 
    t = np.linspace(0, t_final, int(fs * t_final))
    a1 = 1
    a2 = 2

    y = coherent_radar(a1, a2, t, fs=fs)

    print(y.Om)

    noise = NoiseAWGN(n=len(t), sigma=0.5)
    noise.generate()
    print(noise.samples)

    y.Om = y.Om + noise.samples


    fig, grid = create_figure(3,1, figsize=(16, 9))
    TimePlot(
        fig, grid, (0,0), 
        t,
        signals =[y.Os],
        labels=[r"$O_s(t)$"]

    ).plot()
    TimePlot(
        fig, grid, (1,0), 
        t,
        signals =[y.Oc],
        labels=[r"$O_c(t)$"]

    ).plot()
    TimePlot(
        fig, grid, (2,0), 
        t,
        signals =[y.Om],
        labels=[r"$O_m(t)$"]

    ).plot()

    fig.tight_layout()
    save_figure(fig, "coherent_radar_time.pdf")


    fig, grid = create_figure(3,1, figsize=(16, 9))
    FrequencyPlot(
        fig, grid, (0,0), 
        fs=fs,
        signal=y.Os,
        fc=100,
        labels=[r"$O_s(F)$"],
        xlim=[-2,2]
    ).plot()
    FrequencyPlot(
        fig, grid, (1,0), 
        fs=fs,
        signal=y.Oc,
        fc=100,
        labels=[r"$O_c(F)$"],
        xlim=[-2,2]
    ).plot()
    FrequencyPlot(
        fig, grid, (2,0), 
        fs=fs,
        signal=y.Om,
        fc=100,
        labels=[r"$O_m(F)$"],
        xlim=[-2,2]
    ).plot()
    fig.tight_layout()
    save_figure(fig, "coherent_radar_freq.pdf")

    HPA = HighPassFilter(6, 900, fs, filter_type="butter")
    y_hp = HPA.apply(y.Om)
    print(y_hp)

    freq_response, grid_freq_response = create_figure(1, 1, figsize=(16,6))
    FrequencyResponsePlot(
            freq_response, grid_freq_response, (0,0), 
            HPA.b, HPA.a, 
            fs=HPA.fs, 
            f_cut=HPA.cutoff_freq, 
            xlim=(0, 3*HPA.cutoff_freq),
            title="High-Pass-filter Frequency Response"    
        ).plot()
    save_figure(freq_response, "hpa_freq_response.pdf")

    fig, grid = create_figure(2,1, figsize=(16, 9))
    FrequencyPlot(
        fig, grid, (0,0), 
        fs=fs,
        signal=y.Om,
        fc=100,
        labels=[r"$O_m(F)$"],
        xlim=[-2,2]
    ).plot()
    FrequencyPlot(
        fig, grid, (1,0), 
        fs=fs,
        signal=y_hp,
        fc=100,
        labels=[r"$O_m * HPA$"],
        xlim=[-2,2]
    ).plot()
    fig.tight_layout()
    save_figure(fig, "coherent_radar_freq_filtered.pdf")
