"""
Black-Scholes-Merton (BSM) Option Pricing Model
Used for benchmarking Monte Carlo results for vanilla options
"""
import numpy as np
from scipy.stats import norm


def bsm_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes-Merton option price
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility (annualized)
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    float : Option price
    """
    if T <= 0:
        # Option expired
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    if sigma <= 0:
        # No volatility
        if option_type == 'call':
            return max(S * np.exp(-r * T) - K * np.exp(-r * T), 0)
        else:
            return max(K * np.exp(-r * T) - S * np.exp(-r * T), 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def bsm_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes-Merton Greeks analytically
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility (annualized)
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    dict : Dictionary containing delta, gamma, vega, rho, theta
    """
    if T <= 0 or sigma <= 0:
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'theta': 0.0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Common terms
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)  # PDF of standard normal
    
    # Delta
    if option_type == 'call':
        delta = N_d1
    else:  # put
        delta = N_d1 - 1
    
    # Gamma (same for call and put)
    gamma = n_d1 / (S * sigma * np.sqrt(T))
    
    # Vega (same for call and put)
    vega = S * n_d1 * np.sqrt(T) / 100  # Per 1% change in volatility
    
    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * N_d2 / 100  # Per 1% change in rate
    else:  # put
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    # Theta (per day, negative for time decay)
    if option_type == 'call':
        theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365
    else:  # put
        theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'rho': rho,
        'theta': theta
    }
