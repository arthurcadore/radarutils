# """
# This module contains probability related functions for radar calculations;
#
# Author: Arthur Cadore
# """

#TODO: Implement
# function to calculate pd and pfa
# function to calculate equivalent noise bandwidth
# function to calculate likelihood ratio
# function to calculate system performance
# function to calculate bayes
# function to calculate cross-section radar

import numpy as np 
from abc import ABC, abstractmethod

class randomVariable(ABC):
    @abstractmethod
    def __init__(self):
        pass
    def generate(self, n):
        raise NotImplementedError

    def pdf(self, n, span=5):
        raise NotImplementedError

class notSoRandom(randomVariable):
    def __init__(self, value=0):
        super().__init__()
        self.value = value

    def generate(self, n):
        return np.full(n, self.value)

    def pdf(self, n, span=5):
        raise NotImplementedError

class Rayleigh(randomVariable):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def generate(self, n):
        return np.random.rayleigh(self.sigma, n)
    
    def pdf(self, n, span=5):
        x = np.linspace(0, span * self.sigma, len(n))
        return (x / self.sigma**2) * np.exp(-x**2 / (2 * self.sigma**2))

class NoiseAWGN(randomVariable):
    r"""
        Class for generating AWGN (Additive White Gaussian Noise).

        $$
        \begin{equation}
            p(n) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{n^2}{2\sigma^2}}
        \end{equation}
        $$

        Where: 
            - $p(n)$ is the probability density function of the noise.
            - $\sigma^2_n$ is the variance of the noise ($\sigma^2_n = \mathbb{E}[n^2]$)

    """
    def __init__(self, sigma=1, u=0, seed=None):
        r"""
        Initialize the NoiseAWGN class.
        
        Args:
            sigma (float): Standard deviation of the noise.
            u (float): Mean of the noise.
            seed (int): Seed for the random number generator.
        """
        super().__init__()
        self.u = u
        self.sigma = sigma
        self.seed = seed

    def generate(self, n):
        r"""
        Generate AWGN samples using `np.random.normal` function.
        
        Returns:
            samples (np.ndarray): Generated AWGN samples.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        self.samples = np.random.normal(loc=self.u, scale=self.sigma, size=n)
        return self.samples

    def pdf(self, n, span=5):
        r"""
        Compute the Gaussian PDF exactly as in the formula below: 

        $$
        \begin{equation}
            p(n) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{n^2}{2\sigma^2}}
        \end{equation}
        $$

        Where: 
            - $p(n)$ is the probability density function of the noise.
            - $\sigma^2_n$ is the variance of the noise ($\sigma^2_n = \mathbb{E}[n^2]$)
        
        Returns:
            pdf (np.ndarray): Gaussian PDF values.
        """

        x = np.linspace(-span * self.sigma, span * self.sigma, len(n))

        pdf = (1.0 / np.sqrt(2 * np.pi * self.sigma**2)) * \
               np.exp(-(x**2) / (2 * self.sigma**2))
        return pdf
