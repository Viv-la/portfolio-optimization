import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import download_data, calculate_returns
from src.backtesting import backtest_strategy

os.makedirs("data/processed", exist_ok=True)

prices = download_data()
returns = calculate_returns(prices)

# Backtesting period
backtest_returns = returns.loc["2025-01-01":]

# Load optimized portfolio weights
weights_df = pd.read_csv("data/processed/portfolio_weights.csv")
strategy_weights = weights_df["Max Sharpe Weight"].values

# Benchmark: 60% SPY / 40% BND / 0% TSLA
benchmark_weights = [0.0, 0.4, 0.6]  # order: TSLA, BND, SPY

cumulative, metrics = backtest_strategy(
    backtest_returns,
    strategy_weights,
    benchmark_weights
)

cumulative.to_csv("data/processed/backtest_cumulative_returns.csv")
metrics.to_csv("data/processed/backtest_metrics.csv", index=False)

print("Backtesting completed successfully.")
print(metrics)