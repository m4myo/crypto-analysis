import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots

def create_price_chart(df, symbol, show_mas=False, show_bb=False):
    """
    Create an interactive price chart using Plotly with optional technical indicators
    """
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            opacity=0.3
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'{symbol} Price History',
        yaxis_title='Price (USD)',
        yaxis2_title='Volume',
        xaxis_title='Date',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    return fig

def create_correlation_heatmap(correlation_matrix):
    """
    Create a correlation heatmap using Plotly
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        text=correlation_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='%{x} vs %{y}<br>Correlation: %{text}<extra></extra>',
    ))

    fig.update_layout(
        title='Cryptocurrency Correlation Matrix',
        height=700,  # Increased height for better visibility
        width=800,   # Fixed width for better presentation
    )

    return fig

def create_technical_indicators_chart(df, symbol):
    """
    Create technical indicators charts
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('RSI', 'MACD', 'Bollinger Bands'),
                        row_heights=[0.33, 0.33, 0.34])

    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
        row=1, col=1
    )
    # Add RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=df.index, y=df['Histogram'], name='Histogram'),
        row=2, col=1
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Upper Band'], name='Upper Band',
                  line=dict(color='gray', dash='dash')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Middle Band'], name='Middle Band',
                  line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Lower Band'], name='Lower Band',
                  line=dict(color='gray', dash='dash')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close Price',
                  line=dict(color='black')),
        row=3, col=1
    )

    fig.update_layout(
        height=900,
        title_text=f"Technical Indicators for {symbol}",
        showlegend=True
    )

    return fig

def create_prediction_analysis(df, model_accuracy, feature_importance, latest_prediction):
    """
    Create visualization for prediction analysis
    """
    # Create figure with proper specs for indicator
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "xy"}],  # First row for bar chart
               [{"type": "indicator"}]],  # Second row for gauge
        subplot_titles=('Feature Importance', 'Price Trend Prediction'),
        row_heights=[0.6, 0.4]
    )

    # Feature importance bar chart
    fig.add_trace(
        go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h',
            name='Feature Importance'
        ),
        row=1, col=1
    )

    # Prediction probability gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=latest_prediction[1] * 100,  # Probability of upward trend
            title={'text': f"Probability of Price Increase"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        title_text=f"ML Model Analysis (Accuracy: {model_accuracy:.2%})",
        showlegend=False
    )

    return fig

def create_portfolio_risk_dashboard(prices_df, weights, metrics, risk_contribution):
    """
    Create portfolio risk analysis dashboard
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Portfolio Allocation',
            'Risk Metrics',
            'Risk Contribution',
            'Portfolio Value Over Time'
        ),
        specs=[
            [{"type": "pie"}, {"type": "table"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )

    # Portfolio allocation pie chart
    fig.add_trace(
        go.Pie(
            labels=prices_df.columns,
            values=weights,
            name="Portfolio Weights"
        ),
        row=1, col=1
    )

    # Risk metrics table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                align='left'
            ),
            cells=dict(
                values=[
                    ['Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                    [
                        f"{metrics['return']:.2%}",
                        f"{metrics['volatility']:.2%}",
                        f"{metrics['sharpe_ratio']:.2f}",
                        f"{metrics['max_drawdown']:.2%}"
                    ]
                ],
                align='left'
            )
        ),
        row=1, col=2
    )

    # Risk contribution bar chart
    fig.add_trace(
        go.Bar(
            x=risk_contribution.index,
            y=risk_contribution.values,
            name='Risk Contribution'
        ),
        row=2, col=1
    )

    # Portfolio value over time
    portfolio_value = (1 + prices_df.pct_change()).cumprod()
    weighted_portfolio = (portfolio_value * weights).sum(axis=1)

    fig.add_trace(
        go.Scatter(
            x=portfolio_value.index,
            y=weighted_portfolio,
            name='Portfolio Value'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        title_text="Portfolio Risk Analysis Dashboard",
        showlegend=True
    )

    return fig