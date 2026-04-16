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
        if isinstance(n, int):
            x = np.linspace(0, span * self.sigma, n)
        else:
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
    def __init__(self, sigma=1, u=0, seed=None, n=None):
        r"""
        Initialize the NoiseAWGN class.
        
        Args:
            sigma (float): Standard deviation of the noise.
            u (float): Mean of the noise.
            seed (int): Seed for the random number generator.
            n (int): Number of samples to generate on initialization.
        """
        super().__init__()
        self.u = u
        self.sigma = sigma
        self.seed = seed
        self.n = n
        
        # If n is provided, generate everything immediately (expected by some tests)
        if n is not None:
            self.generate(n)
            self.x = np.linspace(-5 * self.sigma, 5 * self.sigma, n)
            self.pdf_values = self.pdf(self.x)
            self.variance_value = np.var(self.samples)
        else:
            self.samples = None
            self.pdf_values = None
            self.variance_value = None
            self.x = None

    def generate(self, n):
        r"""
        Generate AWGN samples using `np.random.normal` function.
        
        Returns:
            samples (np.ndarray): Generated AWGN samples.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.samples = np.random.normal(loc=self.u, scale=self.sigma, size=n)
        self.n = n # Update current n
        self.variance_value = np.var(self.samples)
        
        return self.samples

    def pdf(self, n=None, span=5):
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
        
        Args:
            n (int, array, or None): Number of points or points to calculate PDF for.
            span (float): Span in multiples of sigma.

        Returns:
            pdf (np.ndarray): Gaussian PDF values.
        """
        # Determine the x axis points
        if n is None:
            if self.x is not None:
                x = self.x
            elif self.n is not None:
                x = np.linspace(-span * self.sigma, span * self.sigma, self.n)
            else:
                raise ValueError("Number of points (n) must be provided if not initialized with n.")
        elif isinstance(n, int):
            x = np.linspace(-span * self.sigma, span * self.sigma, n)
        else:
            # Assume n is an array of points
            x = n

        pdf = (1.0 / np.sqrt(2 * np.pi * self.sigma**2)) * \
               np.exp(-(x**2) / (2 * self.sigma**2))
        
        # Update state if calculated for initial size
        if n is None or (isinstance(n, int) and n == self.n):
            self.pdf_values = pdf
            self.x = x

        return pdf
