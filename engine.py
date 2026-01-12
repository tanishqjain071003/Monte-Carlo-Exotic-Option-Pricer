import numpy as np


class Engine:
    def __init__(self, process, instruments, r, T):
        self.process = process
        self.instruments = instruments
        self.r = r
        self.T = T
