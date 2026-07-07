import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import download_data
from src.forecasting import train_arima_model, forecast_future_prices

os.makedirs("data/processed", exist_ok=True)

prices = download_data()
tsla = prices["TSLA"]

model, forecast, metrics = train_arima_model(tsla)

forecast.to_csv("data/processed/tsla_arima_test_forecast.csv")
metrics.to_csv("data/processed/arima_results.csv", index=False)

future_forecast, confidence_intervals = forecast_future_prices(model, steps=180)
future_forecast.to_csv("data/processed/tsla_future_forecast.csv")
confidence_intervals.to_csv("data/processed/tsla_forecast_confidence_intervals.csv")

print("Forecasting completed successfully.")
print(metrics)