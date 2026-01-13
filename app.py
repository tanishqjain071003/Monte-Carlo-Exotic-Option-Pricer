import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pricer import OptionPricer
from bsm_pricer import bsm_price, bsm_greeks
from market_data import fetch_market_data, fetch_options_chain
import json
from datetime import datetime, date

# #region agent log
LOG_PATH = "/Users/tanishqjain/Desktop/Pricer Project/.cursor/debug.log"
def debug_log(location, message, data, hypothesis_id=None):
    try:
        with open(LOG_PATH, "a") as f:
            log_entry = {
                "location": location,
                "message": message,
                "data": data,
                "timestamp": pd.Timestamp.now().isoformat(),
                "sessionId": "debug-session",
                "hypothesisId": hypothesis_id
            }
            f.write(json.dumps(log_entry) + "\n")
    except: pass
# #endregion

# Page configuration
st.set_page_config(
    page_title="Options & Exotic Options Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 Options & Exotic Options Pricer</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for inputs
with st.sidebar:
    st.header("📊 Market Data")
    
    # Ticker input for real-time data
    ticker = st.text_input(
        "Stock Ticker",
        value="",
        placeholder="e.g., AAPL, MSFT, TSLA",
        help="Enter a stock ticker to fetch real-time market data"
    )
    
    # Fetch market data if ticker is provided
    market_data = None
    if ticker:
        try:
            with st.spinner(f"Fetching data for {ticker}..."):
                market_data = fetch_market_data(ticker.upper())
                st.success(f"✅ {market_data['name']} ({market_data['ticker']})")
                if market_data.get('currency'):
                    st.caption(f"Currency: {market_data['currency']}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            market_data = None
    
    st.markdown("---")
    st.header("⚙️ Market Parameters")
    
    # Auto-populate spot price from market data if available
    default_s0 = market_data['spot'] if market_data else 100.0
    s0 = st.number_input(
        "Spot Price (S₀)",
        min_value=1.0,
        max_value=10000.0,
        value=default_s0,
        step=0.01,
        help="Current price of the underlying asset (auto-filled from ticker if provided)"
    )
    
    # Show spot price indicator if fetched from market
    if market_data:
        st.caption(f"💡 Latest spot price: ${market_data['spot']:.2f}")
        if market_data.get('volatility'):
            st.caption(f"📈 30-day historical volatility: {market_data['volatility']*100:.2f}%")
    
    r = st.number_input(
        "Risk-Free Rate (r)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.001,
        format="%.3f",
        help="Annual risk-free interest rate"
    )
    
    sigma = st.number_input(
        "Volatility (σ)",
        min_value=0.01,
        max_value=2.0,
        value=0.2,
        step=0.01,
        format="%.2f",
        help="Annual volatility of the underlying asset"
    )
    
    st.markdown("---")
    st.header("📅 Option Dates")
    
    current_date = st.date_input(
        "Current Date",
        value=date.today(),
        help="Today's date or valuation date"
    )
    
    expiration_date = st.date_input(
        "Expiration Date",
        value=date.today().replace(year=date.today().year + 1),
        min_value=current_date,
        help="Option expiration date"
    )
    
    # Calculate time to expiration in years
    days_to_expiration = (expiration_date - current_date).days
    if days_to_expiration <= 0:
        st.error("⚠️ Expiration date must be after current date!")
        T = 0.01  # Default to avoid division by zero
        trading_days = 1
    else:
        # Calculate trading days (approximately 252 trading days per year)
        # Simple approximation: exclude weekends (about 2/7 of days)
        trading_days = int(days_to_expiration * 5 / 7)
        T = days_to_expiration / 365.25  # Time in years (accounting for leap years)
    
    # Display calculated values with explanation
    st.info(f"**Time to Expiration:** {days_to_expiration} calendar days ({trading_days} trading days) = {T:.2f} years")
    
    st.markdown("---")
    st.header("🎲 Monte Carlo Parameters")
    
    # For daily pricing, automatically set steps equal to trading days
    # Check if dates have changed (by comparing with stored trading_days)
    if "last_trading_days" not in st.session_state:
        st.session_state.last_trading_days = trading_days
        st.session_state.time_steps = min(trading_days, 500)  # Cap at 500 for performance
    elif st.session_state.last_trading_days != trading_days:
        # Dates changed - reset to new trading_days for daily pricing
        st.session_state.time_steps = min(trading_days, 500)
        st.session_state.last_trading_days = trading_days
    
    # Calculate valid range
    min_steps = max(1, trading_days // 10)
    max_steps = min(500, trading_days * 2)
    
    # Ensure current value is within valid range
    if st.session_state.time_steps < min_steps:
        st.session_state.time_steps = min_steps
    elif st.session_state.time_steps > max_steps:
        st.session_state.time_steps = min(max_steps, trading_days)
    
    steps = st.slider(
        "Time Steps (Trading Days)",
        min_value=min_steps,
        max_value=max_steps,
        value=st.session_state.time_steps,
        step=1,
        key="time_steps_slider",
        help=f"Number of time steps for path simulation. Set to {trading_days} for daily pricing (one step per trading day)."
    )
    
    # Update session state with slider value
    st.session_state.time_steps = steps
    
    # Show relationship between steps and time
    if steps > 0:
        time_per_step = T / steps
        if steps == trading_days:
            st.success(f"✅ **Daily Pricing Mode:** {steps} steps = {trading_days} trading days (1 step per trading day)")
        else:
            st.caption(f"📊 With {steps} steps: each step = {time_per_step*365.25:.2f} calendar days ({time_per_step*252:.3f} trading days) = {time_per_step:.3f} years")
            st.info(f"💡 **Tip:** For daily pricing, set steps to {trading_days} (currently {steps})")
    
    paths = st.slider(
        "Number of Paths",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="Number of Monte Carlo simulation paths"
    )
    
    st.markdown("---")
    st.header("📊 Instrument Selection")
    
    instrument_type = st.selectbox(
        "Option Type",
        ["Vanilla", "Asian", "Barrier", "Lookback"],
        help="Select the type of option to price"
    )
    
    option_type = st.radio(
        "Option Style",
        ["Call", "Put"],
        help="Call or Put option"
    )
    
    # Instrument-specific parameters
    if instrument_type == "Vanilla":
        strike = st.number_input(
            "Strike Price (K)",
            min_value=1.0,
            max_value=100000.0,
            value=100.0,
            step=1.0
        )
        instrument_params = {
            'strike': strike,
            'option_type': option_type.lower()
        }
    
    elif instrument_type == "Asian":
        strike = st.number_input(
            "Strike Price (K)",
            min_value=1.0,
            max_value=100000.0,
            value=100.0,
            step=1.0
        )
        avg_type = st.radio(
            "Average Type",
            ["Arithmetic", "Geometric"],
            help="Average Type"
        )
        instrument_params = {
            'strike': strike,
            'avg_type': avg_type.lower(),
            'option_type': option_type.lower()
        }

        
    elif instrument_type == "Barrier":
        strike = st.number_input(
            "Strike Price (K)",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=1.0
        )
        barrier = st.number_input(
            "Barrier Level (B)",
            min_value=1.0,
            max_value=10000.0,
            value=80.0,
            step=1.0
        )
        barrier_type = st.selectbox(
            "Barrier Type",
            ["down-and-out", "down-and-in", "up-and-out", "up-and-in"]
        )
        instrument_params = {
            'strike': strike,
            'barrier': barrier,
            'barrier_type': barrier_type,
            'option_type': option_type.lower()
        }
    
    elif instrument_type == "Lookback":
        lookback = st.selectbox(
            "Lookback Type",
            ["min", "max"]
        )
        instrument_params = {
            'lookback': lookback,
            'option_type': option_type.lower()
        }
    
    st.markdown("---")
    
    # Calculate button
    calculate_button = st.button("🚀 Calculate Price & Greeks", type="primary")
    
    # #region agent log
    debug_log("app.py:185", "Button state check", {
        "calculate_button": calculate_button,
        "has_results": "results" in st.session_state
    }, "A")
    # #endregion

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None
    # #region agent log
    debug_log("app.py:192", "Session state initialized", {"results": None}, "B")
    # #endregion

# Main content area
if calculate_button:
    # #region agent log
    debug_log("app.py:198", "Calculate button clicked", {"button_clicked": True}, "A")
    # #endregion
    try:
        with st.spinner("Running Monte Carlo simulation... This may take a moment."):
            # Create pricer instance
            pricer = OptionPricer(
                s0=s0,
                r=r,
                sigma=sigma,
                T=T,
                steps=steps,
                paths=paths,
                instrument=instrument_type.lower(),
                instrument_params=instrument_params
            )
            
            # Calculate price and Greeks
            results = pricer.get_price_and_greeks()
            
            # Store results in session state
            st.session_state.results = results
            new_params = {
                's0': s0, 'r': r, 'sigma': sigma, 'T': T, 'steps': steps,
                'paths': paths, 'instrument': instrument_type.lower(),
                'instrument_params': instrument_params
            }
            
            # Clear cached paths if parameters changed
            if "pricer_params" in st.session_state and st.session_state.pricer_params != new_params:
                if "cached_paths" in st.session_state:
                    del st.session_state.cached_paths
                    del st.session_state.cached_n_paths
            
            st.session_state.pricer_params = new_params
            
            # #region agent log
            debug_log("app.py:211", "Results stored in session state", {
                "price": float(results['price']),
                "has_greeks": "greeks" in results,
                "cleared_cache": "cached_paths" not in st.session_state
            }, "B")
            # #endregion
            
    except Exception as e:
        st.error(f"❌ Error during calculation: {str(e)}")
        st.exception(e)
        # #region agent log
        debug_log("app.py:390", "Calculation error", {"error": str(e)}, "D")
        # #endregion

# Display results from session state if they exist (even if button wasn't just clicked)
if st.session_state.results is not None:
    # #region agent log
    debug_log("app.py:395", "Displaying results from session state", {
        "has_results": True,
        "calculate_button": calculate_button
    }, "B")
    # #endregion
    
    results = st.session_state.results
    pricer_params = st.session_state.pricer_params
    
    # Check if results have the expected structure
    if results is None or 'greeks' not in results or results['greeks'] is None:
        st.error("❌ Error: Calculation results are incomplete. Please recalculate.")
        # Clear invalid results
        st.session_state.results = None
    else:
        # Recreate pricer for path generation (lightweight operation)
        pricer = OptionPricer(**pricer_params)
        
        # Display results
        if calculate_button:  # Only show success message on new calculation
            st.success("✅ Calculation completed successfully!")
        
        # Create two columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Option Price (MC)",
                f"${results['price']:.3f}",
                help="Monte Carlo estimated option price"
            )
            
            
            # BSM benchmark for vanilla options
            if pricer_params['instrument'] == 'vanilla':
                strike = pricer_params['instrument_params'].get('strike', 100)
                option_type = pricer_params['instrument_params'].get('option_type', 'call')
                bsm_price_val = bsm_price(
                    pricer_params['s0'],
                    strike,
                    pricer_params['T'],
                    pricer_params['r'],
                    pricer_params['sigma'],
                    option_type
                )
                diff = results['price'] - bsm_price_val
                diff_pct = (diff / bsm_price_val * 100) if bsm_price_val > 0 else 0
                
                st.metric(
                    "Option Price (BSM)",
                    f"${bsm_price_val:.3f}",
                    delta=f"{diff:+.3f} ({diff_pct:+.2f}%)",
                    help="Black-Scholes-Merton analytical price (benchmark)"
                )
                st.caption(f"MC vs BSM: {abs(diff):.3f} difference")
        
        with col2:
            st.metric(
                "Delta (Δ)",
                f"{results['greeks']['delta']:.3f}",
                help="Sensitivity to underlying price changes"
            )
            st.metric(
                "Gamma (Γ)",
                f"{results['greeks']['gamma']:.3f}",
                help="Rate of change of Delta"
            )
        
        with col3:
            st.metric(
                "Vega (ν)",
                f"{results['greeks']['vega']:.3f}",
                help="Sensitivity to volatility changes"
            )
            st.metric(
                "Rho (ρ)",
                f"{results['greeks']['rho']:.3f}",
                help="Sensitivity to interest rate changes"
            )
        
        # Theta in a separate row
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric(
                "Theta (Θ)",
                f"{results['greeks']['theta']:.3f}",
                help="Time decay of option value"
            )
    
    st.markdown("---")
    
    # Greeks Stress Test / Scenario Analysis
    st.subheader("🎯 Greeks Stress Test - What-If Analysis")
    
    enable_stress_test = st.checkbox(
        "Enable Stress Test",
        value=False,
        help="Enable interactive scenario analysis to see how Greeks change under different market conditions"
    )
    
    if enable_stress_test:
        st.caption("Adjust market parameters to see how Greeks change in different scenarios")
        
        stress_col1, stress_col2 = st.columns(2)
        
        with stress_col1:
            st.write("**Market Scenario Adjustments**")
            stress_s0 = st.slider(
                "Stock Price Change (%)",
                min_value=-50.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                key="stress_s0",
                help="Simulate stock price movements (e.g., -5% for market crash, +10% for rally)"
            )
            stress_sigma = st.slider(
                "Volatility Change (absolute)",
                min_value=-0.3,
                max_value=0.3,
                value=0.0,
                step=0.01,
                key="stress_sigma",
                help="Simulate volatility changes (e.g., +0.1 = volatility jumps from 20% to 30%)"
            )
        
        with stress_col2:
            st.write("**Rate & Time Adjustments**")
            stress_r = st.slider(
                "Interest Rate Change (absolute)",
                min_value=-0.05,
                max_value=0.05,
                value=0.0,
                step=0.001,
                key="stress_r",
                help="Simulate interest rate changes"
            )
            stress_T = st.slider(
                "Time Decay (days)",
                min_value=-30,
                max_value=0,
                value=0,
                step=1,
                key="stress_T",
                help="Simulate time passing (negative = days passed)"
            )
        
        # Calculate stressed parameters
        stressed_s0 = pricer_params['s0'] * (1 + stress_s0 / 100)
        stressed_sigma = max(0.01, pricer_params['sigma'] + stress_sigma)
        stressed_r = max(0.0, pricer_params['r'] + stress_r)
        stressed_T = max(0.01, pricer_params['T'] + stress_T / 365.25)
        
        # Calculate stressed Greeks
        if stress_s0 != 0 or stress_sigma != 0 or stress_r != 0 or stress_T != 0:
            try:
                stressed_pricer = OptionPricer(
                    s0=stressed_s0,
                    r=stressed_r,
                    sigma=stressed_sigma,
                    T=stressed_T,
                    steps=pricer_params['steps'],
                    paths=min(5000, pricer_params['paths']),  # Use fewer paths for faster stress test
                    instrument=pricer_params['instrument'],
                    instrument_params=pricer_params['instrument_params']
                )
                stressed_greeks = stressed_pricer.calculate_greeks()
                stressed_price = stressed_pricer.price()[0]
                
                # Display comparison
                st.markdown("#### 📊 Stressed vs Base Scenario Comparison")
                
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.metric(
                        "Price Change",
                        f"${stressed_price:.3f}",
                        delta=f"${stressed_price - results['price']:+.3f}",
                        help="Option price under stressed scenario"
                    )
                
                with comp_col2:
                    delta_change = stressed_greeks['delta'] - results['greeks']['delta']
                    st.metric(
                        "Delta Change",
                        f"{stressed_greeks['delta']:.3f}",
                        delta=f"{delta_change:+.3f}",
                        help="Delta under stressed scenario"
                    )
                
                with comp_col3:
                    gamma_change = stressed_greeks['gamma'] - results['greeks']['gamma']
                    st.metric(
                        "Gamma Change",
                        f"{stressed_greeks['gamma']:.3f}",
                        delta=f"{gamma_change:+.3f}",
                        help="Gamma under stressed scenario"
                    )
                
                # Detailed comparison table
                comparison_df = pd.DataFrame({
                    'Greek': ['Delta', 'Gamma', 'Vega', 'Rho', 'Theta'],
                    'Base': [
                        results['greeks']['delta'],
                        results['greeks']['gamma'],
                        results['greeks']['vega'],
                        results['greeks']['rho'],
                        results['greeks']['theta']
                    ],
                    'Stressed': [
                        stressed_greeks['delta'],
                        stressed_greeks['gamma'],
                        stressed_greeks['vega'],
                        stressed_greeks['rho'],
                        stressed_greeks['theta']
                    ],
                    'Change': [
                        stressed_greeks['delta'] - results['greeks']['delta'],
                        stressed_greeks['gamma'] - results['greeks']['gamma'],
                        stressed_greeks['vega'] - results['greeks']['vega'],
                        stressed_greeks['rho'] - results['greeks']['rho'],
                        stressed_greeks['theta'] - results['greeks']['theta']
                    ],
                    'Change %': [
                        ((stressed_greeks['delta'] - results['greeks']['delta']) / abs(results['greeks']['delta']) * 100) if results['greeks']['delta'] != 0 else 0,
                        ((stressed_greeks['gamma'] - results['greeks']['gamma']) / abs(results['greeks']['gamma']) * 100) if results['greeks']['gamma'] != 0 else 0,
                        ((stressed_greeks['vega'] - results['greeks']['vega']) / abs(results['greeks']['vega']) * 100) if results['greeks']['vega'] != 0 else 0,
                        ((stressed_greeks['rho'] - results['greeks']['rho']) / abs(results['greeks']['rho']) * 100) if results['greeks']['rho'] != 0 else 0,
                        ((stressed_greeks['theta'] - results['greeks']['theta']) / abs(results['greeks']['theta']) * 100) if results['greeks']['theta'] != 0 else 0,
                    ]
                })
                
                st.dataframe(
                    comparison_df.style.format({
                        'Base': '{:.3f}',
                        'Stressed': '{:.3f}',
                        'Change': '{:+.3f}',
                        'Change %': '{:+.3f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualization of changes
                fig_stress = go.Figure()
                fig_stress.add_trace(go.Bar(
                    name='Base',
                    x=comparison_df['Greek'],
                    y=comparison_df['Base'],
                    marker_color='#1f77b4'
                ))
                fig_stress.add_trace(go.Bar(
                    name='Stressed',
                    x=comparison_df['Greek'],
                    y=comparison_df['Stressed'],
                    marker_color='#ff7f0e'
                ))
                fig_stress.update_layout(
                    title="Greeks: Base vs Stressed Scenario",
                    xaxis_title="Greek",
                    yaxis_title="Value",
                    barmode='group',
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig_stress, use_container_width=True)
                
            except Exception as e:
                st.warning(f"⚠️ Could not calculate stressed scenario: {str(e)}")
        else:
            st.info("👆 Adjust the sliders above to see how Greeks change under different market scenarios")
    
    st.markdown("---")
    
    # Greeks visualization
    st.subheader("📊 Greeks Visualization")
    
    greeks_data = {
        'Greek': ['Delta', 'Gamma', 'Vega', 'Rho', 'Theta'],
        'Value': [
            results['greeks']['delta'],
            results['greeks']['gamma'],
            results['greeks']['vega'],
            results['greeks']['rho'],
            results['greeks']['theta']
        ]
    }
    
    fig_greeks = go.Figure(data=[
        go.Bar(
            x=greeks_data['Greek'],
            y=greeks_data['Value'],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            text=[f"{v:.3f}" for v in greeks_data['Value']],
            textposition='outside'
        )
    ])
    fig_greeks.update_layout(
        title="Option Greeks",
        xaxis_title="Greek",
        yaxis_title="Value",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig_greeks, use_container_width=True)
    
    # Sample paths visualization
    st.subheader("📈 Sample Monte Carlo Paths")
    
    # Use a container to prevent full page rerun on slider change
    paths_container = st.container()
    
    with paths_container:
        n_sample_paths = st.slider(
            "Number of Sample Paths to Display",
            min_value=5,
            max_value=100,
            value=10,
            key="sample_paths"
        )
        
        # #region agent log
        debug_log("app.py:485", "Slider value retrieved", {
            "n_sample_paths": n_sample_paths,
            "has_results": True
        }, "C")
        # #endregion
        
        # Only regenerate paths if slider value changed or not cached
        if "cached_paths" not in st.session_state or st.session_state.get("cached_n_paths") != n_sample_paths:
            # #region agent log
            debug_log("app.py:492", "Generating new sample paths", {
                "n_sample_paths": n_sample_paths,
                "was_cached": "cached_paths" in st.session_state
            }, "C")
            # #endregion
            sample_paths = pricer.generate_sample_paths(n_paths=n_sample_paths)
            st.session_state.cached_paths = sample_paths
            st.session_state.cached_n_paths = n_sample_paths
        else:
            # #region agent log
            debug_log("app.py:500", "Using cached paths", {
                "n_sample_paths": n_sample_paths
            }, "C")
            # #endregion
            sample_paths = st.session_state.cached_paths
        
        time_points = np.linspace(0, pricer_params['T'], pricer_params['steps'] + 1)
        
        fig_paths = go.Figure()
        
        for i in range(n_sample_paths):
            fig_paths.add_trace(go.Scatter(
                x=time_points,
                y=sample_paths[i],
                mode='lines',
                name=f'Path {i+1}',
                line=dict(width=1),
                opacity=0.6,
                showlegend=False
            ))
        
        # Add strike line if applicable
        if pricer_params['instrument'] in ["vanilla", "asian", "barrier"]:
            strike = pricer_params['instrument_params'].get('strike', 100)
            fig_paths.add_hline(
                y=strike,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Strike: ${strike:.2f}",
                annotation_position="right"
            )
        
        # Add barrier line if applicable
        if pricer_params['instrument'] == "barrier":
            barrier = pricer_params['instrument_params'].get('barrier', 80)
            fig_paths.add_hline(
                y=barrier,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Barrier: ${barrier:.2f}",
                annotation_position="right"
            )
        
        fig_paths.update_layout(
            title="Sample Monte Carlo Simulation Paths",
            xaxis_title="Time (Years)",
            yaxis_title="Stock Price ($)",
            height=500,
            template="plotly_white",
            hovermode='x unified'
        )
        st.plotly_chart(fig_paths, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    results_df = pd.DataFrame([
        {"Parameter": "Option Price", "Value": f"${results['price']:.6f}"},
        {"Parameter": "Delta (Δ)", "Value": f"{results['greeks']['delta']:.6f}"},
        {"Parameter": "Gamma (Γ)", "Value": f"{results['greeks']['gamma']:.6f}"},
        {"Parameter": "Vega (ν)", "Value": f"{results['greeks']['vega']:.6f}"},
        {"Parameter": "Rho (ρ)", "Value": f"{results['greeks']['rho']:.6f}"},
        {"Parameter": "Theta (Θ)", "Value": f"{results['greeks']['theta']:.6f}"},
    ])
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)

else:
    # Welcome message
    st.info("👈 Please configure the parameters in the sidebar and click 'Calculate Price & Greeks' to begin.")
    
    # Information section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        ### Options & Exotic Options Pricer
        
        This application uses **Monte Carlo Simulation** to price various types of options and calculate their Greeks.
        
        #### Supported Option Types:
        - **Vanilla Options**: Standard European call/put options
        - **Asian Options**: Options with payoff based on average price
        - **Barrier Options**: Options that activate or deactivate based on barrier levels
        - **Lookback Options**: Options with payoff based on minimum/maximum price during life
        
        #### Greeks Calculated:
        - **Delta (Δ)**: Sensitivity to underlying price changes
        - **Gamma (Γ)**: Rate of change of Delta
        - **Vega (ν)**: Sensitivity to volatility changes
        - **Rho (ρ)**: Sensitivity to interest rate changes
        - **Theta (Θ)**: Time decay of option value
        
        #### Monte Carlo Method:
        The simulation uses Geometric Brownian Motion (GBM) with antithetic variates for variance reduction.
        """)
