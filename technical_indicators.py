import pandas as pd
import numpy as np

def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)"""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'MACD': macd,
        'Signal': signal_line,
        'Histogram': histogram
    })

def calculate_moving_averages(data):
    """Calculate various moving averages"""
    mas = pd.DataFrame(index=data.index)
    periods = [20, 50, 200]  # Common moving average periods
    
    for period in periods:
        mas[f'MA{period}'] = data['Close'].rolling(window=period).mean()
    
    return mas

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    ma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    
    upper_band = ma + (std * std_dev)
    lower_band = ma - (std * std_dev)
    
    return pd.DataFrame({
        'Middle Band': ma,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    })
