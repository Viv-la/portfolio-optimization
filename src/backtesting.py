import numpy as np
import pandas as pd


def calculate_portfolio_returns(returns, weights):
    return returns.dot(weights)


def cumulative_returns(portfolio_returns):
    return (1 + portfolio_returns).cumprod()


def maximum_drawdown(cumulative_return_series):
    running_max = cumulative_return_series.cummax()
    drawdown = (cumulative_return_series - running_max) / running_max
    return drawdown.min()


def backtest_strategy(returns, strategy_weights, benchmark_weights):
    strategy_returns = calculate_portfolio_returns(returns, strategy_weights)
    benchmark_returns = calculate_portfolio_returns(returns, benchmark_weights)

    strategy_cumulative = cumulative_returns(strategy_returns)
    benchmark_cumulative = cumulative_returns(benchmark_returns)

    trading_days = 252

    metrics = pd.DataFrame({
        "Portfolio": ["Optimized Strategy", "Benchmark 60% SPY / 40% BND"],
        "Total Return": [
            strategy_cumulative.iloc[-1] - 1,
            benchmark_cumulative.iloc[-1] - 1
        ],
        "Annualized Return": [
            strategy_returns.mean() * trading_days,
            benchmark_returns.mean() * trading_days
        ],
        "Annualized Volatility": [
            strategy_returns.std() * np.sqrt(trading_days),
            benchmark_returns.std() * np.sqrt(trading_days)
        ],
        "Sharpe Ratio": [
            (strategy_returns.mean() * trading_days) / (strategy_returns.std() * np.sqrt(trading_days)),
            (benchmark_returns.mean() * trading_days) / (benchmark_returns.std() * np.sqrt(trading_days))
        ],
        "Maximum Drawdown": [
            maximum_drawdown(strategy_cumulative),
            maximum_drawdown(benchmark_cumulative)
        ]
    })

    cumulative = pd.DataFrame({
        "Optimized Strategy": strategy_cumulative,
        "Benchmark 60% SPY / 40% BND": benchmark_cumulative
    })

    return cumulative, metrics