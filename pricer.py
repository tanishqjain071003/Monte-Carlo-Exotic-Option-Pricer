import numpy as np
from stochastic_process import GBMProcess
from instruments import VanillaOption, AsianOption, BarrierOption, LookbackOption, DigitalOption
from greek_calculator import GreekCalculator


class OptionPricer:
    """
    Main class that connects stochastic processes, instruments, and Greek calculations
    """
    
    def __init__(self, s0, r, sigma, T, steps, paths, instrument, instrument_params):
        """
        Initialize the pricer with market parameters and instrument
        
        Parameters:
        -----------
        s0 : float
            Initial stock price
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        T : float
            Time to expiration (in years)
        steps : int
            Number of time steps for Monte Carlo simulation
        paths : int
            Number of Monte Carlo paths
        instrument : str
            Type of instrument: 'vanilla', 'asian', 'barrier', 'lookback'
        instrument_params : dict
            Parameters specific to the instrument type
        """
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.paths = paths
        
        # Initialize stochastic process
        self.process = GBMProcess(s0, r, sigma, T, steps, paths)
        
        # Initialize instrument based on type
        self.instrument = self._create_instrument(instrument, instrument_params)
        
        # Initialize Greek calculator
        self.calculator = GreekCalculator(self.process, self.instrument, r, T)
    
    def _create_instrument(self, instrument_type, params):
        """Create the appropriate instrument based on type"""
        if instrument_type == 'vanilla':
            return VanillaOption(
                strike=params.get('strike', 100),
                option_type=params.get('option_type', 'call')
            )
        elif instrument_type == 'asian':
            return AsianOption(
                strike=params.get('strike', 100),
                option_type=params.get('option_type', 'call'),
                avg_type=params.get('avg_type', 'arithmetic')
            )
        elif instrument_type == 'barrier':
            return BarrierOption(
                strike=params.get('strike', 100),
                barrier=params.get('barrier', 80),
                barrier_type=params.get('barrier_type', 'down-and-out'),
                option_type=params.get('option_type', 'call')
            )
        elif instrument_type == 'lookback':
            return LookbackOption(
                lookback=params.get('lookback', 'min'),
                option_type=params.get('option_type', 'call')
            )
        elif instrument_type == 'digital':
            return DigitalOption(
                strike=params.get('strike', 100),
                payout=params.get('payout', 10),
                option_type=params.get('option_type', 'call')
            )
        else:
            raise ValueError(f"Unknown instrument type: {instrument_type}")
    
    def price(self):
        """
        Calculate the option price using Monte Carlo simulation
        
        Returns:
        --------
        price : float
            Option price
        std_err : float
            Standard error of the estimate
        """
        return self.calculator.price()
    
    def calculate_greeks(self):
        """
        Calculate all Greeks (Delta, Gamma, Vega, Rho, Theta)
        
        Returns:
        --------
        dict : Dictionary containing all Greek values
        """
        return self.calculator.main()
    
    def get_price_and_greeks(self):
        """
        Get both price and Greeks in a single call
        
        Returns:
        --------
        dict : Dictionary containing price, standard error, and all Greeks
        """
        price, std_err = self.price()
        greeks = self.calculate_greeks()
        
        return {
            'price': price,
            'standard_error': std_err,
            'greeks': greeks
        }
    
    def generate_sample_paths(self, n_paths=10):
        """
        Generate sample paths for visualization
        
        Parameters:
        -----------
        n_paths : int
            Number of sample paths to generate
            
        Returns:
        --------
        numpy.ndarray : Array of shape (n_paths, steps+1) containing sample paths
        """
        original_paths = self.process.paths
        self.process.paths = n_paths
        paths = self.process.generate_paths()  
        self.process.paths = original_paths
        return paths

