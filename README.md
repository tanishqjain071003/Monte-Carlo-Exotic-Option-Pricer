# Options & Exotic Options Pricer

A Monte Carlo simulation-based options pricing application with interactive Streamlit interface.

## Features

- **Multiple Option Types**: Vanilla, Asian, Barrier, and Lookback options
- **Greeks Calculation**: Delta, Gamma, Vega, Rho, and Theta
- **Interactive UI**: Modern Streamlit interface with real-time visualization
- **Monte Carlo Simulation**: Uses Geometric Brownian Motion with antithetic variates

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

The application will open in your default web browser.

## Project Structure

- `pricer.py` - Main pricer module that connects all components
- `app.py` - Streamlit application interface
- `stochastic_process.py` - GBM process for Monte Carlo simulation
- `instruments.py` - Option instrument definitions
- `greek_calculator.py` - Greeks calculation logic
- `engine.py` - Base engine class

## How It Works

1. Configure market parameters (stock price, volatility, interest rate, etc.)
2. Select option type and parameters
3. Set Monte Carlo simulation parameters
4. Click "Calculate Price & Greeks" to run the simulation
5. View results including price, Greeks, and visualizations
