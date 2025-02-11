import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_crypto_data(symbol, period='1y'):
    """
    Fetch cryptocurrency historical data using yfinance
    """
    try:
        crypto = yf.Ticker(f"{symbol}-USD")
        df = crypto.history(period=period)
        return df
    except Exception as e:
        return None

def calculate_statistics(df):
    """
    Calculate basic statistics for the cryptocurrency
    """
    stats = {
        'Current Price': df['Close'].iloc[-1],
        'Daily Return (%)': ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100,
        'Weekly Return (%)': ((df['Close'].iloc[-1] / df['Close'].iloc[-7]) - 1) * 100,
        'Monthly Return (%)': ((df['Close'].iloc[-1] / df['Close'].iloc[-30]) - 1) * 100,
        'Volatility (30D)': df['Close'].pct_change().std() * np.sqrt(252) * 100,
        'Highest Price (1Y)': df['High'].max(),
        'Lowest Price (1Y)': df['Low'].min(),
        'Trading Volume': df['Volume'].iloc[-1]
    }
    return stats

def get_correlation_data(current_symbol):
    """
    Get correlation data for major cryptocurrencies and the currently selected one
    """
    # Extended list of major cryptocurrencies
    symbols = [
        'BTC',  # Bitcoin
        'ETH',  # Ethereum
        'XRP',  # Ripple
        'BNB',  # Binance Coin
        'SOL',  # Solana
        'ADA',  # Cardano
        'DOT',  # Polkadot
        'DOGE', # Dogecoin
        'AVAX', # Avalanche
        'MATIC' # Polygon
    ]

    # Add current symbol if it's not already in the list
    if current_symbol not in symbols:
        symbols.append(current_symbol)

    dfs = {}

    # Get data for each symbol
    for symbol in symbols:
        df = get_crypto_data(symbol)
        if df is not None:
            dfs[symbol] = df['Close']

    if dfs:
        correlation_df = pd.DataFrame(dfs)
        correlation_matrix = correlation_df.corr()
        # Round correlation values to 3 decimal places
        correlation_matrix = correlation_matrix.round(3)
        return correlation_matrix
    return None

def get_crypto_info(symbol):
    """
    Get basic information about a cryptocurrency
    """
    crypto_names = {
        'BTC': 'Bitcoin',
        'ETH': 'Ethereum',
        'XRP': 'Ripple',
        'BNB': 'Binance Coin',
        'SOL': 'Solana',
        'ADA': 'Cardano',
        'DOT': 'Polkadot',
        'DOGE': 'Dogecoin',
        'AVAX': 'Avalanche',
        'MATIC': 'Polygon'
    }
    return crypto_names.get(symbol, symbol)