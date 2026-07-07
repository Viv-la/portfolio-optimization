# Interim Report: Time Series Forecasting for Portfolio Management Optimization

## 1. Introduction

This interim report presents the initial progress for the GMF Investments portfolio forecasting project. The objective is to analyze historical financial data for Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY), and prepare the data for forecasting and portfolio optimization.

The interim submission focuses on Task 1 and initial progress on Task 2, as required by the challenge brief.

## 2. Data Extraction and Cleaning

Historical market data was extracted using the YFinance Python library for the period January 1, 2015 to June 30, 2026. The assets analyzed were TSLA, BND, and SPY.

Adjusted closing prices were used because they account for stock splits and dividends, making them more appropriate for historical return analysis.

The data was inspected for missing values, data types, and overall completeness. Missing values were removed after confirming that the cleaned dataset retained sufficient observations for analysis.

## 3. Exploratory Data Analysis

The exploratory analysis focused on identifying trends, volatility patterns, and risk differences across the three assets.

Tesla showed the strongest growth potential but also the highest volatility. BND displayed more stable price behavior, consistent with its role as a bond ETF. SPY showed moderate growth and moderate volatility, reflecting its broad market exposure.

Daily returns were calculated to analyze short-term fluctuations. The results showed that TSLA experienced larger daily return swings compared to BND and SPY.

## 4. Volatility Analysis

A 30-day rolling volatility measure was calculated for all three assets. TSLA consistently showed higher volatility than BND and SPY, confirming its high-risk profile. BND showed the lowest volatility, supporting its role as a stabilizing asset in a diversified portfolio.

## 5. Stationarity Testing

The Augmented Dickey-Fuller test was applied to both adjusted closing prices and daily returns.

The price series were generally non-stationary, meaning they contain trends and are not directly suitable for ARIMA modelling without differencing. The daily return series were more stationary, making them more appropriate for statistical modelling and risk analysis.

This supports the use of differencing in ARIMA modelling.

## 6. Risk Metrics

The following risk metrics were calculated:

- Annualized return
- Annualized volatility
- Sharpe Ratio
- 95% Value at Risk

The results confirmed that TSLA has the highest risk and return potential. BND has the lowest risk profile, while SPY provides balanced exposure between growth and diversification.

## 7. Initial Forecasting Model

Initial progress was made on Task 2 by implementing an ARIMA model for Tesla. The dataset was split chronologically, with training data covering the earlier period and test data covering 2025 onward.

A baseline ARIMA(5,1,0) model was trained and evaluated using:

- Mean Absolute Error
- Root Mean Squared Error
- Mean Absolute Percentage Error

This provides a starting point for the final modelling phase, where ARIMA/SARIMA and LSTM models will be compared.

## 8. Conclusion and Next Steps

The interim analysis successfully completed data extraction, cleaning, exploratory analysis, stationarity testing, volatility analysis, and risk metric calculation. Initial time series forecasting progress was also completed using ARIMA.

For the final submission, the project will be extended to include SARIMA and LSTM modelling, future price forecasting, portfolio optimization using Modern Portfolio Theory, Efficient Frontier visualization, and backtesting against a 60% SPY / 40% BND benchmark.