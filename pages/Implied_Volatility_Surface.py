"""
Implied Volatility Surface 3D Visualization Page
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from market_data import fetch_market_data, calculate_implied_volatility_surface
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Options & Exotic Options Pricer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Implied Volatility Surface</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #718096; margin-bottom: 2rem;">Interactive 3D Visualization of Options Market Implied Volatility</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for inputs
with st.sidebar:
    st.header("📈 Market Data Input")
    
    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, TSLA, SPY",
        help="Enter a stock ticker to fetch real-time options data"
    )
    
    # Fetch market data
    market_data = None
    spot_price = None
    if ticker:
        try:
            with st.spinner(f"Fetching market data for {ticker}..."):
                market_data = fetch_market_data(ticker.upper())
                spot_price = market_data['spot']
                st.success(f"✅ {market_data['name']}")
                st.metric("Spot Price", f"${spot_price:.2f}")
                if market_data.get('volatility'):
                    st.metric("30-Day Volatility", f"{market_data['volatility']*100:.2f}%")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            market_data = None
            spot_price = None
    
    st.markdown("---")
    st.header("⚙️ Visualization Settings")
    
    risk_free_rate = st.number_input(
        "Risk-Free Rate (r)",
        min_value=0.0,
        max_value=0.2,
        value=0.05,
        step=0.001,
        format="%.3f",
        help="Risk-free interest rate for IV calculation"
    )
    
    num_strikes = st.slider(
        "Number of Strikes",
        min_value=10,
        max_value=50,
        value=25,
        step=5,
        help="Number of strike prices to include in the surface"
    )
    
    num_expirations = st.slider(
        "Number of Expirations",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="Number of expiration dates to include"
    )
    
    st.markdown("---")
    
    # Calculate button
    calculate_iv = st.button("🚀 Generate IV Surface", type="primary", use_container_width=True)
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# Main content
if not ticker:
    st.info("👈 Please enter a stock ticker in the sidebar to begin.")
    st.markdown("""
    ### About Implied Volatility Surface
    
    The Implied Volatility (IV) Surface is a 3D visualization showing how implied volatility 
    varies across different:
    - **Strike Prices** (X-axis): Option strike prices
    - **Time to Expiration** (Y-axis): Days until option expiration
    - **Implied Volatility** (Z-axis): Market-implied volatility levels
    
    #### Key Features:
    - **Real-time Data**: Fetches live options data from Yahoo Finance
    - **Interactive 3D Chart**: Rotate, zoom, and explore the volatility surface
    - **Volatility Smile/Skew**: Visualize how IV changes with moneyness
    - **Term Structure**: See how IV varies with time to expiration
    
    #### Common Patterns:
    - **Volatility Smile**: Higher IV for out-of-the-money options
    - **Volatility Skew**: Asymmetric IV distribution (common in equity markets)
    - **Term Structure**: How IV changes with expiration dates
    """)
else:
    if calculate_iv or 'iv_surface_data' not in st.session_state:
        if spot_price is None:
            st.error("❌ Please ensure market data is loaded. Try refreshing the ticker.")
        else:
            try:
                with st.spinner("🔄 Calculating implied volatility surface... This may take a moment."):
                    iv_surface_data = calculate_implied_volatility_surface(
                        ticker.upper(),
                        spot_price,
                        risk_free_rate,
                        num_strikes,
                        num_expirations
                    )
                    st.session_state.iv_surface_data = iv_surface_data
                    st.session_state.spot_price = spot_price
                    st.session_state.ticker = ticker.upper()
                    st.success("✅ Implied volatility surface calculated successfully!")
            except Exception as e:
                st.error(f"❌ Error calculating IV surface: {str(e)}")
                st.info("💡 Tip: Make sure the ticker has active options contracts available.")
                st.session_state.iv_surface_data = None
    
    # Display visualization if data is available
    if 'iv_surface_data' in st.session_state and st.session_state.iv_surface_data is not None:
        iv_data = st.session_state.iv_surface_data
        spot = st.session_state.spot_price
        ticker_display = st.session_state.ticker
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Ticker", ticker_display)
        with col2:
            st.metric("Spot Price", f"${spot:.2f}")
        with col3:
            min_iv = np.nanmin(iv_data['implied_vols'])
            max_iv = np.nanmax(iv_data['implied_vols'])
            st.metric("IV Range", f"{min_iv*100:.1f}% - {max_iv*100:.1f}%")
        with col4:
            avg_iv = np.nanmean(iv_data['implied_vols'])
            st.metric("Average IV", f"{avg_iv*100:.2f}%")
        
        st.markdown("---")
        
        # Create 3D surface plot
        st.subheader("🎨 3D Implied Volatility Surface")
        
        # Prepare data for 3D surface
        strikes = iv_data['strikes']
        expirations = iv_data['expirations']
        iv_surface = iv_data['implied_vols']
        moneyness = iv_data['moneyness']
        
        # Create meshgrid for surface plot
        X, Y = np.meshgrid(strikes, expirations * 365.25)  # Convert years to days for display
        
        # Create 3D surface plot with modern styling
        # Use minimal colorbar config to avoid compatibility issues
        surface_trace = go.Surface(
            x=X,
            y=Y,
            z=iv_surface * 100,  # Convert to percentage
            colorscale='Viridis',
            showscale=True,
            hovertemplate='<b>Strike:</b> $%{x:.2f}<br>' +
                         '<b>Days to Exp:</b> %{y:.0f}<br>' +
                         '<b>IV:</b> %{z:.2f}%<extra></extra>',
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.2,
                roughness=0.4,
                fresnel=0.1
            ),
            lightposition=dict(x=100, y=100, z=1000)
        )
        
        fig_3d = go.Figure(data=[surface_trace])
        
        # Update colorbar after creation using data access
        if len(fig_3d.data) > 0:
            fig_3d.data[0].colorbar = dict(
                title="Implied Volatility (%)",
                len=0.75,
                y=0.5,
                yanchor='middle'
            )
        
        # Update layout with modern styling
        fig_3d.update_layout(
            title=dict(
                text=f'Implied Volatility Surface - {ticker_display}',
                font=dict(size=24, color='#2d3748', family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text='Strike Price ($)', font=dict(size=14, color='#4a5568')),
                    backgroundcolor='rgba(255, 255, 255, 0.9)',
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showbackground=True,
                    zerolinecolor='rgba(100, 100, 100, 0.5)'
                ),
                yaxis=dict(
                    title=dict(text='Days to Expiration', font=dict(size=14, color='#4a5568')),
                    backgroundcolor='rgba(255, 255, 255, 0.9)',
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showbackground=True,
                    zerolinecolor='rgba(100, 100, 100, 0.5)'
                ),
                zaxis=dict(
                    title=dict(text='Implied Volatility (%)', font=dict(size=14, color='#4a5568')),
                    backgroundcolor='rgba(255, 255, 255, 0.9)',
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showbackground=True,
                    zerolinecolor='rgba(100, 100, 100, 0.5)'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=0.8, z=0.6),
                bgcolor='rgba(248, 250, 252, 1)'
            ),
            height=700,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Arial', size=12, color='#2d3748')
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Additional visualizations
        st.markdown("---")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("📈 Volatility Smile (ATM Strike)")
            # Find ATM strike
            atm_idx = np.argmin(np.abs(strikes - spot))
            atm_strikes = strikes
            atm_ivs = iv_surface[0, :]  # Use first expiration
            
            fig_smile = go.Figure()
            fig_smile.add_trace(go.Scatter(
                x=atm_strikes,
                y=atm_ivs * 100,
                mode='lines+markers',
                name='IV',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#764ba2'),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            fig_smile.add_vline(
                x=spot,
                line_dash="dash",
                line_color="red",
                annotation_text="ATM",
                annotation_position="top"
            )
            fig_smile.update_layout(
                title="Implied Volatility vs Strike Price",
                xaxis_title="Strike Price ($)",
                yaxis_title="Implied Volatility (%)",
                height=400,
                template="plotly_white",
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig_smile, use_container_width=True)
        
        with col_viz2:
            st.subheader("⏰ Term Structure of Volatility")
            # Average IV across strikes for each expiration
            avg_iv_by_exp = np.nanmean(iv_surface, axis=1)
            exp_days = expirations * 365.25
            
            fig_term = go.Figure()
            fig_term.add_trace(go.Scatter(
                x=exp_days,
                y=avg_iv_by_exp * 100,
                mode='lines+markers',
                name='Avg IV',
                line=dict(color='#f5576c', width=3),
                marker=dict(size=10, color='#f093fb'),
                fill='tozeroy',
                fillcolor='rgba(245, 87, 108, 0.1)'
            ))
            fig_term.update_layout(
                title="Average Implied Volatility vs Time to Expiration",
                xaxis_title="Days to Expiration",
                yaxis_title="Average Implied Volatility (%)",
                height=400,
                template="plotly_white",
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig_term, use_container_width=True)
        
        # Data table
        st.markdown("---")
        st.subheader("📋 Implied Volatility Data Table")
        
        # Create DataFrame from raw data
        raw_data = iv_data['raw_data']
        df_iv = pd.DataFrame({
            'Strike ($)': raw_data['strikes'],
            'Days to Expiration': [int(exp * 365.25) for exp in raw_data['expirations']],
            'Moneyness': [f"{m:.2f}" for m in raw_data['moneyness']],
            'Implied Volatility (%)': [f"{iv*100:.2f}" for iv in raw_data['implied_vols']]
        })
        
        st.dataframe(
            df_iv.sort_values(['Days to Expiration', 'Strike ($)']),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df_iv.to_csv(index=False)
        st.download_button(
            label="📥 Download IV Data as CSV",
            data=csv,
            file_name=f"{ticker_display}_iv_surface_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
