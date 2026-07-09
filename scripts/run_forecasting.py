import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import download_data
from src.forecasting import (
    train_arima_model,
    forecast_future_prices,
    train_lstm_model
)

os.makedirs("data/processed", exist_ok=True)

prices = download_data()
tsla = prices["TSLA"]

# ARIMA model
arima_model, arima_forecast, arima_metrics = train_arima_model(tsla)

arima_forecast.to_csv("data/processed/tsla_arima_test_forecast.csv")
arima_metrics.to_csv("data/processed/arima_results.csv", index=False)

# Future forecast using ARIMA
future_forecast, confidence_intervals = forecast_future_prices(arima_model, steps=180)

future_forecast.to_csv("data/processed/tsla_future_forecast.csv")
confidence_intervals.to_csv("data/processed/tsla_forecast_confidence_intervals.csv")

# LSTM model
lstm_model, lstm_forecast, lstm_metrics = train_lstm_model(
    tsla,
    epochs=10,
    batch_size=32
)

lstm_forecast.to_csv("data/processed/tsla_lstm_test_forecast.csv")
lstm_metrics.to_csv("data/processed/lstm_results.csv", index=False)

# Model comparison
model_comparison = pd.concat(
    [arima_metrics, lstm_metrics],
    ignore_index=True
)

model_comparison.to_csv("data/processed/model_comparison.csv", index=False)

print("Forecasting completed successfully.")
print(model_comparison)