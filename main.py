import streamlit as st
import pandas as pd
import numpy as np
from crypto_utils import get_crypto_data, calculate_statistics, get_correlation_data, get_crypto_info
from visualizations import (
    create_price_chart, create_correlation_heatmap,
    create_technical_indicators_chart, create_prediction_analysis,
    create_portfolio_risk_dashboard
)
from technical_indicators import (
    calculate_rsi, calculate_macd, calculate_moving_averages,
    calculate_bollinger_bands
)
from ml_prediction import (
    prepare_features, train_model, get_feature_importance,
    predict_trend
)
from portfolio_risk import (
    calculate_portfolio_metrics, optimize_portfolio,
    calculate_risk_contribution
)

# Page configuration
st.set_page_config(
    page_title="Crypto Analysis Tool",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📊 Cryptocurrency Analysis Tool")
st.markdown("""
This tool provides historical data analysis and correlation insights for cryptocurrencies.
Enter a cryptocurrency symbol (e.g., BTC, ETH, XRP) to get started.

**Available Major Cryptocurrencies:**
- BTC (Bitcoin)
- ETH (Ethereum)
- XRP (Ripple)
- BNB (Binance Coin)
- SOL (Solana)
- ADA (Cardano)
- DOT (Polkadot)
- DOGE (Dogecoin)
- AVAX (Avalanche)
- MATIC (Polygon)
""")

# Input section
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    crypto_symbol = st.text_input("Enter Cryptocurrency Symbol:", value="BTC").upper()
with col2:
    time_period = st.selectbox(
        "Select Time Period:",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3
    )
with col3:
    show_indicators = st.checkbox("Show Technical Indicators", value=True)

# Fetch and display data
if crypto_symbol:
    df = get_crypto_data(crypto_symbol, time_period)

    if df is not None:
        # Display crypto name if available
        crypto_name = get_crypto_info(crypto_symbol)
        if crypto_name != crypto_symbol:
            st.subheader(f"{crypto_name} ({crypto_symbol}) Analysis")

        # Price chart tab
        price_chart = create_price_chart(df, crypto_symbol)
        st.plotly_chart(price_chart, use_container_width=True)

        if show_indicators:
            # Calculate technical indicators
            df['RSI'] = calculate_rsi(df)
            macd_data = calculate_macd(df)
            df = pd.concat([df, macd_data], axis=1)
            bollinger_bands = calculate_bollinger_bands(df)
            df = pd.concat([df, bollinger_bands], axis=1)

            # Display technical indicators
            st.subheader("Technical Indicators")
            indicators_chart = create_technical_indicators_chart(df, crypto_symbol)
            st.plotly_chart(indicators_chart, use_container_width=True)

            # Add explanation of indicators
            with st.expander("Understanding Technical Indicators"):
                st.markdown("""
                **RSI (Relative Strength Index)**
                - Measures momentum and identifies overbought (>70) or oversold (<30) conditions
                - Range: 0 to 100

                **MACD (Moving Average Convergence Divergence)**
                - Shows trend direction and momentum
                - Components: MACD line, Signal line, and Histogram
                - Bullish signal: MACD crosses above signal line
                - Bearish signal: MACD crosses below signal line

                **Bollinger Bands**
                - Shows volatility and potential price levels
                - Upper and lower bands represent 2 standard deviations from the middle band
                - Price touching bands might indicate potential reversal
                """)

            # Add ML predictions
            st.subheader("Machine Learning Predictions")

            # Prepare features and train model
            features = prepare_features(df)
            model, scaler, accuracy, (X_test, y_test, y_pred) = train_model(features)

            # Get feature importance
            importance = get_feature_importance(model, features)

            # Get latest prediction
            latest_features = features.drop('target', axis=1).iloc[-1:]
            prediction = predict_trend(model, scaler, latest_features)

            # Display prediction analysis
            prediction_chart = create_prediction_analysis(
                df, accuracy, importance, prediction
            )
            st.plotly_chart(prediction_chart, use_container_width=True)

            # Add explanation
            with st.expander("Understanding ML Predictions"):
                st.markdown("""
                **Machine Learning Model Details:**
                - Uses Random Forest Classifier for trend prediction
                - Features include technical indicators and price/volume changes
                - Model accuracy shown above is based on test data
                - Probability gauge shows likelihood of price increase
                - Feature importance shows which indicators most influence predictions

                **Note:** These predictions are for educational purposes only and should not be used as financial advice.
                """)


        # Statistics table
        st.subheader("Statistical Summary")
        stats = calculate_statistics(df)
        stats_df = pd.DataFrame.from_dict(stats.items())
        stats_df.columns = ['Metric', 'Value']
        st.table(stats_df.set_index('Metric'))

        # Correlation analysis
        st.subheader("Correlation Analysis")
        st.markdown("""
        This heatmap shows the correlation between the selected cryptocurrency and other major cryptocurrencies.
        - Values closer to 1 (dark red) indicate stronger positive correlation
        - Values closer to -1 (dark blue) indicate stronger negative correlation
        - Values closer to 0 indicate little to no correlation
        """)

        correlation_matrix = get_correlation_data(crypto_symbol)
        if correlation_matrix is not None:
            correlation_chart = create_correlation_heatmap(correlation_matrix)
            st.plotly_chart(correlation_chart, use_container_width=True)
        else:
            st.warning("Could not generate correlation analysis. Some data might be unavailable.")

    else:
        st.error(f"Error: Could not fetch data for {crypto_symbol}. Please check the symbol and try again.")

# Add Portfolio Risk Assessment section
    if crypto_symbol:
        st.header("Portfolio Risk Assessment")
        st.markdown("""
        Enter your current cryptocurrency holdings to analyze your portfolio and get optimization suggestions.
        The tool will calculate your current allocation and suggest an optimal portfolio based on historical performance.
        """)

        # Get available cryptocurrencies
        available_cryptos = ['BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'ADA', 'DOT', 'DOGE', 'AVAX', 'MATIC']

        # Portfolio selection
        selected_cryptos = st.multiselect(
            "Select cryptocurrencies in your portfolio",
            available_cryptos,
            default=['BTC', 'ETH']
        )

        if selected_cryptos:
            # Fetch historical data and current prices for selected cryptocurrencies
            prices_df = pd.DataFrame()
            current_prices = {}
            for crypto in selected_cryptos:
                df = get_crypto_data(crypto)
                if df is not None:
                    prices_df[crypto] = df['Close']
                    current_prices[crypto] = df['Close'].iloc[-1]

            if not prices_df.empty:
                # Input holdings
                st.subheader("Your Current Holdings")
                holdings = {}
                total_value = 0

                for crypto in selected_cryptos:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        amount = st.number_input(
                            f"Amount of {crypto}",
                            min_value=0.0,
                            value=0.0,
                            step=0.000001,
                            format="%.6f",
                            key=f"holding_{crypto}"
                        )
                    with col2:
                        value = amount * current_prices[crypto]
                        st.metric(
                            "Value (USD)", 
                            f"${value:,.2f}"
                        )
                        holdings[crypto] = amount
                        total_value += value

                if total_value > 0:
                    # Calculate current weights
                    current_weights = np.array([
                        holdings[crypto] * current_prices[crypto] / total_value 
                        for crypto in selected_cryptos
                    ])

                    # Calculate optimal weights
                    optimal_weights = optimize_portfolio(prices_df)

                    # Calculate metrics for both portfolios
                    current_metrics = calculate_portfolio_metrics(prices_df, current_weights)
                    optimal_metrics = calculate_portfolio_metrics(prices_df, optimal_weights)

                    # Display comparison
                    st.subheader("Portfolio Analysis")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Current Portfolio")
                        current_risk = calculate_risk_contribution(prices_df, current_weights)
                        current_dashboard = create_portfolio_risk_dashboard(
                            prices_df, current_weights, current_metrics, current_risk
                        )
                        st.plotly_chart(current_dashboard, use_container_width=True)

                    with col2:
                        st.markdown("### Optimal Portfolio")
                        optimal_risk = calculate_risk_contribution(prices_df, optimal_weights)
                        optimal_dashboard = create_portfolio_risk_dashboard(
                            prices_df, optimal_weights, optimal_metrics, optimal_risk
                        )
                        st.plotly_chart(optimal_dashboard, use_container_width=True)

                    # Show suggested changes
                    st.subheader("Suggested Portfolio Changes")
                    changes_df = pd.DataFrame({
                        'Asset': selected_cryptos,
                        'Current Allocation': current_weights * 100,
                        'Optimal Allocation': optimal_weights * 100,
                        'Suggested Change (%)': (optimal_weights - current_weights) * 100
                    })
                    changes_df = changes_df.round(2)
                    st.dataframe(changes_df, use_container_width=True)

                    # Performance comparison
                    st.subheader("Performance Comparison")
                    metrics_comparison = pd.DataFrame({
                        'Metric': ['Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                        'Current Portfolio': [
                            f"{current_metrics['return']:.2%}",
                            f"{current_metrics['volatility']:.2%}",
                            f"{current_metrics['sharpe_ratio']:.2f}",
                            f"{current_metrics['max_drawdown']:.2%}"
                        ],
                        'Optimal Portfolio': [
                            f"{optimal_metrics['return']:.2%}",
                            f"{optimal_metrics['volatility']:.2%}",
                            f"{optimal_metrics['sharpe_ratio']:.2f}",
                            f"{optimal_metrics['max_drawdown']:.2%}"
                        ]
                    })
                    st.table(metrics_comparison.set_index('Metric'))

                    # Add explanation
                    with st.expander("Understanding Portfolio Analysis"):
                        st.markdown("""
                        **Portfolio Metrics Explained:**
                        - **Return:** Annualized portfolio return
                        - **Volatility:** Annualized portfolio volatility (risk)
                        - **Sharpe Ratio:** Risk-adjusted return (higher is better)
                        - **Max Drawdown:** Largest peak-to-trough decline

                        **Optimization Details:**
                        - The optimal portfolio maximizes the Sharpe ratio
                        - Considers historical returns, volatility, and correlations
                        - Suggested changes show how to move towards the optimal allocation

                        **Important Note:** 
                        This analysis is based on historical data and should not be considered as financial advice. 
                        Past performance does not guarantee future results.
                        """)
                else:
                    st.info("Enter your holdings to see portfolio analysis")
            else:
                st.error("Could not fetch data for the selected cryptocurrencies.")
        else:
            st.info("Please select at least one cryptocurrency for your portfolio.")

# Footer
st.markdown("""
---
Data provided by Yahoo Finance | Created with Streamlit
""")