import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calculate_portfolio_metrics(prices_df, weights=None):
    """
    Calculate portfolio risk metrics
    """
    if weights is None:
        weights = np.ones(len(prices_df.columns)) / len(prices_df.columns)
    
    # Calculate returns
    returns = prices_df.pct_change()
    
    # Portfolio return
    portfolio_return = np.sum(returns.mean() * weights) * 252
    
    # Portfolio volatility
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * 252, weights))
    )
    
    # Sharpe ratio (assuming risk-free rate of 0.01)
    risk_free_rate = 0.01
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min().min()
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def optimize_portfolio(prices_df, risk_free_rate=0.01):
    """
    Optimize portfolio weights for maximum Sharpe ratio
    """
    returns = prices_df.pct_change()
    n_assets = len(returns.columns)
    
    def objective(weights):
        portfolio_metrics = calculate_portfolio_metrics(prices_df, weights)
        return -portfolio_metrics['sharpe_ratio']  # Minimize negative Sharpe ratio
    
    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
    )
    bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1
    
    # Initial weights (equal weight)
    initial_weights = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x if result.success else initial_weights

def calculate_risk_contribution(prices_df, weights):
    """
    Calculate risk contribution of each asset
    """
    returns = prices_df.pct_change()
    cov_matrix = returns.cov() * 252
    
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )
    
    marginal_risk = np.dot(cov_matrix, weights)
    risk_contribution = np.multiply(weights, marginal_risk) / portfolio_volatility
    
    return pd.Series(risk_contribution, index=prices_df.columns)
