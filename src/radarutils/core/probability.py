
import numpy as np 
# função pra pd e pfa

# função pra gerar ruido AWGN

class NoiseAWGN:
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
    def __init__(self, n, sigma, seed=None, span=5):
        r"""
        Initialize the NoiseAWGN class.
        
        Args:
            n (int): Number of samples.
            sigma (float): Standard deviation of the noise.
            seed (int): Seed for the random number generator.
            span (int): Span of the plot (number of standard deviations from the mean).
        """
        self.n = n
        self.span = span
        self.sigma = sigma
        self.seed = seed
        self.samples = self.generate()
        self.x = np.linspace(-self.span*self.sigma, self.span*self.sigma, self.n)
        self.variance_value = self.variance()
        self.pdf_values = self.pdf()

    def generate(self):
        r"""
        Generate AWGN samples using `np.random.normal` function.
        
        Returns:
            samples (np.ndarray): Generated AWGN samples.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        self.samples = np.random.normal(loc=0.0, scale=self.sigma, size=self.n)
        return self.samples

    def pdf(self):
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
        pdf = (1.0 / np.sqrt(2 * np.pi * self.sigma**2)) * \
               np.exp(-(self.x**2) / (2 * self.sigma**2))
        return pdf

    def variance(self):
        r"""
        Compute the variance of the generated samples using the formula based on the mean value of the samples vector:

        $$
        \begin{equation}
            \sigma^2_n = \mathbb{E}[n^2]
        \end{equation}
        $$
        
        Where: 
            - $\sigma^2_n$ is the variance of the noise.
            - $\mathbb{E}[n^2]$ is the expected value of the square of the samples.
        
        Returns:
            sigma_n (float): Variance of the generated samples.
        """
        sigma_n = np.mean(self.samples**2)
        return sigma_n


# banda equivalente de ruido 

# razão verossimilhança

# desempenho do sistema 

# Bayes

# cross-section radar