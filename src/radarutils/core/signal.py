# """
# This module contains signal related functions for radar calculations;
#
# Author: Arthur Cadore
# """

# TODO: implement
# function to modulate/demodulate IQ

import numpy as np
import matplotlib.pyplot as plt
from .probability import randomVariable, notSoRandom, NoiseAWGN

def rect(t, tp):
    r"""
    if |t| < tp, return 1
    if |t| = tp, return 0.5
    if |t| > tp, return 0
    """

    if np.abs(t) < tp:
        return 1
    elif np.abs(t) == tp:
        return 0.5
    else:
        return 0


#TODO: implementar cumsum para gerar o erro de fase do oscilador com memória (err(t) = err(t-1) + va(t))
class oscilator():
    def __init__(self, A, t, fc=1000, va:randomVariable=notSoRandom): 
        self.A = A
        self.fc = fc
        self.t = t
        self.err = []
        self.va = va


    def generate(self):
        
        #(err(t) = err(t-1) + va(t))
        self.err = np.cumsum(self.va.generate(len(self.t)))
        print(self.err)

        return self.A * np.cos(2 * np.pi * self.fc * self.t + self.err)

    
class AmplitudeModulator():
    pass

class FrequencyModulator():
    pass

if __name__ == "__main__":

    va1 = NoiseAWGN(sigma=0.05)
    va2 = notSoRandom(value=1)
    fs = 128000
    ts = 1/fs
    t = np.arange(0, 0.02, ts)

    osc = oscilator(A=1, t=t, va=va2)
    signal = osc.generate()

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title("Sinal Oscilador")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
