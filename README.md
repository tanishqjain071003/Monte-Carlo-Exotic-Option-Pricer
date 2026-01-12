# Options & Exotic Options Pricer

A Monte Carlo simulation-based options pricing application with interactive Streamlit interface, real-time market data integration, and advanced implied volatility visualization.

## Features

### Core Pricing Features
- **Multiple Option Types**: Vanilla, Asian, Barrier, and Lookback options
- **Greeks Calculation**: Delta, Gamma, Vega, Rho, and Theta
- **Interactive UI**: Modern Streamlit interface with real-time visualization
- **Monte Carlo Simulation**: Uses Geometric Brownian Motion with antithetic variates

### Real-Time Market Data (NEW!)
- **Live Market Data**: Fetch real-time stock prices using yfinance
- **Ticker Input**: Enter any stock ticker (e.g., AAPL, MSFT, TSLA) to get live data
- **Auto-Populated Spot Price**: Spot price automatically updates from market data
- **Historical Volatility**: 30-day historical volatility estimation

### Implied Volatility Surface (NEW!)
- **3D Visualization**: Interactive 3D surface plot of implied volatility
- **Volatility Smile/Skew**: Visualize how IV changes with moneyness
- **Term Structure**: See how IV varies across different expiration dates
- **Real-Time Options Data**: Fetches live options chain data from Yahoo Finance
- **Multiple Visualizations**: 3D surface, volatility smile, and term structure charts
- **Data Export**: Download IV surface data as CSV

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser with multiple pages:
- **Main Page**: Options pricing with real-time market data integration
- **Implied Volatility Surface**: 3D visualization of options market implied volatility

### Using Real-Time Market Data

1. On the main page, enter a stock ticker in the sidebar (e.g., "AAPL")
2. The spot price will automatically populate with the latest market price
3. Historical volatility will be displayed if available
4. Configure other parameters and calculate option prices

### Using Implied Volatility Surface

1. Navigate to the "Implied Volatility Surface" page from the sidebar
2. Enter a stock ticker with active options (e.g., "AAPL", "SPY", "TSLA")
3. Adjust visualization settings (number of strikes, expirations)
4. Click "Generate IV Surface" to create the 3D visualization
5. Interact with the 3D chart: rotate, zoom, and explore the volatility surface
6. View additional charts: volatility smile and term structure
7. Download the data as CSV if needed

## Project Structure

- `pricer.py` - Main pricer module that connects all components
- `app.py` - Streamlit application interface (main pricing page)
- `market_data.py` - Real-time market data fetching using yfinance (NEW!)
- `pages/Implied_Volatility_Surface.py` - IV surface visualization page (NEW!)
- `stochastic_process.py` - GBM process for Monte Carlo simulation
- `instruments.py` - Option instrument definitions
- `greek_calculator.py` - Greeks calculation logic
- `engine.py` - Base engine class
- `bsm_pricer.py` - Black-Scholes-Merton analytical pricing

## How It Works

### Main Pricing Page
1. Enter a stock ticker to fetch real-time market data (optional)
2. Spot price auto-populates from market data
3. Configure market parameters (volatility, interest rate, etc.)
4. Select option type and parameters
5. Set Monte Carlo simulation parameters
6. Click "Calculate Price & Greeks" to run the simulation
7. View results including price, Greeks, and visualizations

### Implied Volatility Surface
1. Enter a stock ticker with active options
2. System fetches options chain data from Yahoo Finance
3. Calculates implied volatility for different strikes and expirations
4. Creates 3D surface visualization showing IV patterns
5. Displays volatility smile and term structure charts
6. Provides downloadable data table

## Dependencies

- `streamlit` - Web application framework
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `plotly` - Interactive visualizations
- `scipy` - Scientific computing (for BSM model)
- `yfinance` - Yahoo Finance market data API

## Notes

- Real-time data requires an active internet connection
- Some tickers may not have options data available
- Market data is fetched from Yahoo Finance (free, but rate-limited)
- Implied volatility calculations use market data from options chain
