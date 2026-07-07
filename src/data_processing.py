import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ["TSLA", "BND", "SPY"]

def download_data(start="2015-01-01", end="2026-06-30"):
    data = yf.download(TICKERS, start=start, end=end, auto_adjust=False)
    prices = data["Adj Close"].dropna()
    return prices

def calculate_returns(prices):
    return prices.pct_change().dropna()

def calculate_risk_metrics(returns, risk_free_rate=0.02):
    trading_days = 252
    annual_return = returns.mean() * trading_days
    annual_volatility = returns.std() * np.sqrt(trading_days)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    var_95 = returns.quantile(0.05)

    return pd.DataFrame({
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "VaR 95%": var_95
    })