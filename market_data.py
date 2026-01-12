"""
Market Data Utility - Fetch real-time market data using yfinance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np


def fetch_market_data(ticker: str):
    """
    Fetch real-time market data for a given ticker
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')
    
    Returns:
    --------
    dict : Dictionary containing market data
        - 'spot': Current spot price
        - 'info': Additional stock information
        - 'history': Recent price history
        - 'volatility': Historical volatility estimate
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get latest price data
        hist = stock.history(period="1d", interval="1m")
        
        if hist.empty:
            # Fallback to daily data
            hist = stock.history(period="5d")
        
        if hist.empty:
            raise ValueError(f"No data available for ticker {ticker}")
        
        # Get the most recent price
        latest_price = hist['Close'].iloc[-1]
        
        # Calculate historical volatility (30-day)
        hist_30d = stock.history(period="30d")
        if not hist_30d.empty and len(hist_30d) > 1:
            returns = hist_30d['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        else:
            volatility = None
        
        return {
            'spot': float(latest_price),
            'info': info,
            'history': hist,
            'volatility': float(volatility) if volatility is not None else None,
            'ticker': ticker.upper(),
            'name': info.get('longName', ticker.upper()),
            'currency': info.get('currency', 'USD')
        }
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")


def fetch_options_chain(ticker: str, expiration_date=None):
    """
    Fetch options chain data for implied volatility calculation
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    expiration_date : str or None
        Expiration date in 'YYYY-MM-DD' format. If None, fetches nearest expiration
    
    Returns:
    --------
    dict : Options chain data with calls and puts
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        expirations = stock.options
        
        if not expirations:
            raise ValueError(f"No options data available for {ticker}")
        
        # Use specified expiration or nearest one
        if expiration_date and expiration_date in expirations:
            exp_date = expiration_date
        else:
            exp_date = expirations[0]  # Nearest expiration
        
        # Fetch options chain
        opt_chain = stock.option_chain(exp_date)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        return {
            'expiration': exp_date,
            'calls': calls,
            'puts': puts,
            'available_expirations': expirations
        }
    except Exception as e:
        raise ValueError(f"Error fetching options chain for {ticker}: {str(e)}")


def calculate_implied_volatility_surface(ticker: str, spot_price: float, risk_free_rate: float, 
                                        num_strikes: int = 20, num_expirations: int = 5):
    """
    Calculate implied volatility surface for 3D visualization
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    spot_price : float
        Current spot price
    risk_free_rate : float
        Risk-free interest rate
    num_strikes : int
        Number of strike prices to include
    num_expirations : int
        Number of expiration dates to include
    
    Returns:
    --------
    dict : Implied volatility surface data
        - 'strikes': Array of strike prices
        - 'expirations': Array of expiration dates (in days)
        - 'implied_vols': 2D array of implied volatilities
        - 'moneyness': Moneyness values (strike/spot)
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options[:num_expirations]  # Get first N expirations
        
        if not expirations:
            raise ValueError(f"No options data available for {ticker}")
        
        # Calculate time to expiration for each expiration date
        today = datetime.now().date()
        times_to_exp = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            days_to_exp = (exp_date - today).days
            times_to_exp.append(days_to_exp / 365.25)  # Convert to years
        
        # Collect all strikes from all expirations
        all_strikes = set()
        for exp in expirations:
            opt_chain = stock.option_chain(exp)
            all_strikes.update(opt_chain.calls['strike'].tolist())
            all_strikes.update(opt_chain.puts['strike'].tolist())
        
        # Filter strikes around ATM (within reasonable range)
        all_strikes = sorted([s for s in all_strikes if 0.5 * spot_price <= s <= 2.0 * spot_price])
        
        # Select evenly spaced strikes
        if len(all_strikes) > num_strikes:
            indices = np.linspace(0, len(all_strikes) - 1, num_strikes, dtype=int)
            selected_strikes = [all_strikes[i] for i in indices]
        else:
            selected_strikes = all_strikes
        
        # Build implied volatility surface
        implied_vols = []
        strikes_array = []
        expirations_array = []
        moneyness_array = []
        
        for i, exp in enumerate(expirations):
            opt_chain = stock.option_chain(exp)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            for strike in selected_strikes:
                # Find closest strike in options chain
                call_row = calls.iloc[(calls['strike'] - strike).abs().argsort()[:1]]
                put_row = puts.iloc[(puts['strike'] - strike).abs().argsort()[:1]]
                
                # Use call IV if available, otherwise put IV
                if not call_row.empty and pd.notna(call_row['impliedVolatility'].iloc[0]):
                    iv = call_row['impliedVolatility'].iloc[0]
                elif not put_row.empty and pd.notna(put_row['impliedVolatility'].iloc[0]):
                    iv = put_row['impliedVolatility'].iloc[0]
                else:
                    iv = None
                
                if iv is not None and iv > 0 and iv < 5:  # Filter reasonable IV values
                    implied_vols.append(iv)
                    strikes_array.append(strike)
                    expirations_array.append(times_to_exp[i])
                    moneyness_array.append(strike / spot_price)
        
        # Reshape into 2D surface if we have enough data
        if len(implied_vols) > 0:
            # Create a grid for visualization
            unique_strikes = sorted(set(strikes_array))
            unique_exps = sorted(set(expirations_array))
            
            # Create 2D grid
            iv_surface = np.full((len(unique_exps), len(unique_strikes)), np.nan)
            
            for i, exp in enumerate(unique_exps):
                for j, strike in enumerate(unique_strikes):
                    # Find matching data point
                    for k in range(len(implied_vols)):
                        if abs(expirations_array[k] - exp) < 0.001 and abs(strikes_array[k] - strike) < 0.01:
                            iv_surface[i, j] = implied_vols[k]
                            break
            
            return {
                'strikes': np.array(unique_strikes),
                'expirations': np.array(unique_exps),
                'implied_vols': iv_surface,
                'moneyness': np.array(unique_strikes) / spot_price,
                'raw_data': {
                    'strikes': strikes_array,
                    'expirations': expirations_array,
                    'implied_vols': implied_vols,
                    'moneyness': moneyness_array
                }
            }
        else:
            raise ValueError("No valid implied volatility data found")
            
    except Exception as e:
        raise ValueError(f"Error calculating implied volatility surface: {str(e)}")
