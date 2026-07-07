import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_performance(weights, expected_returns, covariance_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio


def negative_sharpe_ratio(weights, expected_returns, covariance_matrix):
    return -portfolio_performance(weights, expected_returns, covariance_matrix)[2]


def optimize_portfolio(expected_returns, covariance_matrix):
    num_assets = len(expected_returns)
    initial_weights = np.array([1 / num_assets] * num_assets)

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(expected_returns, covariance_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x


def minimum_volatility_portfolio(covariance_matrix):
    num_assets = covariance_matrix.shape[0]
    initial_weights = np.array([1 / num_assets] * num_assets)

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))

    result = minimize(
        lambda weights: np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights))),
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x


def simulate_random_portfolios(expected_returns, covariance_matrix, num_portfolios=5000):
    results = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(expected_returns))
        weights = weights / np.sum(weights)

        p_return, p_volatility, p_sharpe = portfolio_performance(
            weights, expected_returns, covariance_matrix
        )

        row = {
            "Return": p_return,
            "Volatility": p_volatility,
            "Sharpe Ratio": p_sharpe
        }

        for asset, weight in zip(expected_returns.index, weights):
            row[f"{asset} Weight"] = weight

        results.append(row)

    return pd.DataFrame(results)