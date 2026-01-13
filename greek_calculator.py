import numpy as np
import copy

class GreekCalculator:
    """
    Monte Carlo Greeks using Common Random Numbers (CRN)
    """

    def __init__(self, process, instrument, r, T):
        self.process = process
        self.instrument = instrument
        self.r = r
        self.T = T

        self.ds = 0.01 * process.s0     # 1% spot
        self.dsigma = 0.01              # 1 vol point
        self.dr = 0.0001                # 1 bp
        self.dt = 1 / 365               # 1 day

        self.Z = np.random.standard_normal((process.paths, process.steps))

    def _price_with_process(self, proc, T=None, r=None):
        if T is None:
            T = self.T
        if r is None:
            r = self.r

        dt = T / proc.steps
        drift = (r - 0.5 * proc.sigma**2) * dt
        diffusion = proc.sigma * np.sqrt(dt) * self.Z

        log_returns = drift + diffusion
        paths = proc.s0 * np.exp(np.cumsum(log_returns, axis=1))
        paths = np.insert(paths, 0, proc.s0, axis=1)

        payoff = self.instrument.payoff(paths)
        return np.mean(np.exp(-r * T) * payoff)

    def price(self):
        price = self._price_with_process(self.process)
        return price, np.nan

    def delta(self):
        base = self._price_with_process(self.process)

        up = copy.deepcopy(self.process)
        up.s0 += self.ds

        price_up = self._price_with_process(up)

        type = self.instrument.type

        if type == 1:
            return max(0,min(1, (price_up - base) / self.ds))
        else:
            return min(0,max(-1, (price_up - base) / self.ds))
        

    def gamma(self):
        base = self._price_with_process(self.process)

        up = copy.deepcopy(self.process)
        down = copy.deepcopy(self.process)

        up.s0 += self.ds
        down.s0 -= self.ds

        price_up = self._price_with_process(up)
        price_down = self._price_with_process(down)

        return max(0,(price_up - 2 * base + price_down) / (self.ds ** 2))

    def vega(self):
        base = self._price_with_process(self.process)

        bumped = copy.deepcopy(self.process)
        bumped.sigma += self.dsigma

        price_bumped = self._price_with_process(bumped)
        return (price_bumped - base) / self.dsigma*0.01

    def rho(self):
        base = self._price_with_process(self.process)

        bumped = copy.deepcopy(self.process)
        bumped_price = self._price_with_process(
            bumped,
            r=self.r + self.dr
        )

        # scale to 1% move (market convention)
        return (bumped_price - base) / self.dr / 100

    def theta(self):
        if self.T <= self.dt:
            return np.nan

        base = self._price_with_process(self.process)

        bumped = copy.deepcopy(self.process)
        bumped_price = self._price_with_process(
            bumped,
            T=self.T - self.dt
        )

        return (bumped_price - base) 

    def main(self):
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "rho": self.rho(),
            "theta": self.theta()
        }
