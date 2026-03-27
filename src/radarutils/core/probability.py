
import numpy as np 
# função pra pd e pfa

# função pra gerar ruido AWGN

class noise_awgn:
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

        Args:
            n (int): Number of samples.
            sigma (float): Standard deviation of the noise.
            seed (int): Seed for the random number generator.

    """
    def __init__(self, n, sigma, seed=None):
        self.n = n
        self.sigma = sigma
        self.samples = self.generate(seed)
        self.variance_value = self.variance()
        self.pdf_values = self.pdf(self.samples)

    def generate(self, seed=None):
        r"""Generate AWGN samples."""
        if seed is not None:
            np.random.seed(seed)
        self.samples = np.random.normal(loc=0.0, scale=self.sigma, size=self.n)
        return self.samples

    def pdf(self, x):
        r"""Compute the Gaussian PDF exactly as in the formula."""

        pdf = (1.0 / np.sqrt(2 * np.pi * self.sigma**2)) * \
               np.exp(-(x**2) / (2 * self.sigma**2))
        return pdf

    def variance(self):
        r"""Compute $\sigma^2_n = \mathbb{E}[n^2]$ from the generated samples."""
        sigma_n = np.mean(self.samples**2)
        return sigma_n


# banda equivalente de ruido 

# razão verossimilhança

# desempenho do sistema 

# Bayes

# cross-section radar 

if __name__ == "__main__":
    awgn = noise_awgn(n=1000000, sigma=0.5, seed=10)

    var = awgn.variance_value          
    p = awgn.pdf_values[:5]   

    print(f"Variance: {var}")
    print(f"PDF at samples[:5]: {p}")

    import matplotlib.pyplot as plt
    import numpy as np
    
    # histograma normalizado
    plt.hist(awgn.samples, bins=500, density=True, alpha=0.5, label="Samples")
    
    # PDF teórica
    x = np.linspace(-3*awgn.sigma, 3*awgn.sigma, 200)
    plt.plot(x, awgn.pdf(x), label="PDF", linewidth=2)
    
    plt.legend()
    plt.show()
