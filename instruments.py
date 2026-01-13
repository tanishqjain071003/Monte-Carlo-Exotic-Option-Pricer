from abc import ABC, abstractmethod
import numpy as np

class BaseOption(ABC):
    @abstractmethod
    def payoff(self, paths):
        pass

class VanillaOption(BaseOption):
    def __init__(self, strike, option_type = "call"):
        self.K = strike
        self.type = 1 if option_type == "call" else -1

    def payoff(self, paths):   
        return np.maximum(self.type * (paths[:, -1] - self.K), 0)

class AsianOption(BaseOption):
    def __init__(self, strike, option_type = "call", avg_type = "simple"):
        self.K = strike
        self.type = 1 if option_type == "call" else -1
        self.avg_type = avg_type

    def payoff(self, paths):
        if self.avg_type == "arithmetic":
            averages = np.mean(paths[:, 1:], axis=1)
        elif self.avg_type == "geometric":
            averages = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        
        return np.maximum(self.type * (averages - self.K), 0)
    

class BarrierOption(BaseOption):
    def __init__(self, strike, barrier, barrier_type = "down-and-out", option_type = "call"):
        self.K = strike
        self.B = barrier
        self.type = 1 if option_type == "call" else -1
        self.barrier_type = barrier_type

    def payoff(self, paths):
        ST = paths[:, -1]
        vanilla = np.maximum(self.type * (ST - self.K), 0)
        
        if "down" in self.barrier_type:
            hit = np.any(paths <= self.B, axis = 1)
        else:
            hit = np.any(paths >= self.B, axis = 1)

        if "out" in self.barrier_type:
            return np.where(hit, 0, vanilla)
        else:
            return np.where(hit, vanilla, 0)

class LookbackOption(BaseOption):
    def __init__(self, lookback, option_type = "call"):
        self.lookback = lookback
        self.type = 1 if option_type == "call" else -1

    def payoff(self, paths):
        min = np.min(paths, axis = 1)
        max = np.max(paths, axis = 1)

        ST = paths[:, -1]

        if self.type == 1:
            return np.maximum(ST - min, 0)
        else:
            return np.maximum(max - ST, 0)

class DigitalOption(BaseOption):

    def __init__(self, strike, payout, option_type = "call"):
        self.K = strike
        self.payout = payout
        self.type = 1 if option_type == "call" else -1
    
    def payoff(self, paths):
        ST = paths[:, -1]
        if self.type == 1:
            return np.where(ST > self.K, self.payout, 0)
        else:
            return np.where(ST < self.K, self.payout, 0)
