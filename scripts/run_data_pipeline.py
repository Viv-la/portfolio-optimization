import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import download_data, calculate_returns
from src.portfolio_optimization import optimize_portfolio, minimum_volatility_portfolio, portfolio_performance

os.makedirs("data/processed", exist_ok=True)

prices = download_data()
returns = calculate_returns(prices)

expected_returns = returns.mean() * 252
covariance_matrix = returns.cov() * 252

max_sharpe_weights = optimize_portfolio(expected_returns, covariance_matrix)
min_vol_weights = minimum_volatility_portfolio(covariance_matrix)

assets = expected_returns.index

portfolio_results = pd.DataFrame({
    "Asset": assets,
    "Max Sharpe Weight": max_sharpe_weights,
    "Minimum Volatility Weight": min_vol_weights
})

max_return, max_vol, max_sharpe = portfolio_performance(
    max_sharpe_weights, expected_returns, covariance_matrix
)

min_return, min_vol, min_sharpe = portfolio_performance(
    min_vol_weights, expected_returns, covariance_matrix
)

summary = pd.DataFrame({
    "Portfolio": ["Maximum Sharpe", "Minimum Volatility"],
    "Expected Annual Return": [max_return, min_return],
    "Expected Annual Volatility": [max_vol, min_vol],
    "Sharpe Ratio": [max_sharpe, min_sharpe]
})

portfolio_results.to_csv("data/processed/portfolio_weights.csv", index=False)
summary.to_csv("data/processed/portfolio_summary.csv", index=False)
covariance_matrix.to_csv("data/processed/covariance_matrix.csv")

print("Portfolio optimization completed successfully.")
print(portfolio_results)
print(summary)