import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Task 2: ARIMA baseline forecasting model for Tesla price prediction
def train_arima_model(series, order=(5, 1, 0), split_date="2024-12-31"):
    train = series.loc[:split_date]
    test = series.loc["2025-01-01":]

    model = ARIMA(train, order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test))
    forecast.index = test.index

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    metrics = pd.DataFrame({
        "Model": [f"ARIMA{order}"],
        "MAE": [mae],
        "RMSE": [rmse],
        "MAPE": [mape]
    })

    return fitted_model, forecast, metrics

def forecast_future_prices(fitted_model, steps=180):
    """
    Generate future forecasts with confidence intervals.
    """
    forecast_result = fitted_model.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    confidence_intervals = forecast_result.conf_int()

    return forecast_mean, confidence_intervals


def evaluate_model(actual, predicted, model_name="Model"):
    """
    Evaluate forecast model using MAE, RMSE and MAPE.
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return pd.DataFrame({
        "Model": [model_name],
        "MAE": [mae],
        "RMSE": [rmse],
        "MAPE": [mape]
    })