# Options & Exotic Options Pricer - Project Description

This project is a comprehensive **Monte Carlo simulation-based options pricing platform** built with Streamlit that enables traders and analysts to price various option types and analyze their risk characteristics. The application supports **vanilla, Asian, barrier, and lookback options** with both call and put variants.

The core engine uses **Geometric Brownian Motion (GBM)** to simulate thousands of stock price paths, calculating option prices through risk-neutral valuation. For vanilla options, the system provides **Black-Scholes-Merton (BSM) analytical pricing** as a benchmark to validate Monte Carlo accuracy.

A key feature is the **interactive Greeks stress test**, allowing users to simulate market scenarios (price crashes, volatility spikes, rate changes) and instantly see how option values and sensitivities change. The platform calculates all major Greeks (Delta, Gamma, Vega, Rho, Theta) using **finite difference methods with Common Random Numbers** for enhanced accuracy.

The application includes date-based inputs for realistic time-to-expiration calculations, automatic daily pricing mode setup, and comprehensive visualizations of price paths and Greek sensitivities. Designed for both educational purposes and practical risk management, it demonstrates industry-standard quantitative finance techniques in an accessible, interactive interface.
