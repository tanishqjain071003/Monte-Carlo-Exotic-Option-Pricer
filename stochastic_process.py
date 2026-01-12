import numpy as np

class GBMProcess:
    def __init__(self, s0, r, sigma, T, steps, paths):
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.paths = paths

    def generate_paths(self, seed=None):
        # Dynamically calculate dT based on current T
        # This ensures Theta calculations are accurate
        dt = self.T / self.steps
        
        if seed is not None:
            np.random.seed(seed)

        # Using standard Monte Carlo (Antithetic can be toggled)
        size = (self.paths, self.steps)
        Z = np.random.standard_normal(size)

        # GBM Formula: S_t = S_{t-1} * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z

        log_returns = drift + diffusion
        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        
        paths = self.s0 * np.exp(cumulative_log_returns)
        # Add the starting price s0 to the beginning of the paths
        return np.insert(paths, 0, self.s0, axis=1)