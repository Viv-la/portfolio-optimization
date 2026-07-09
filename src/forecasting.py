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

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_lstm_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])

    X = np.array(X)
    y = np.array(y)

    return X.reshape((X.shape[0], X.shape[1], 1)), y


def train_lstm_model(series, split_date="2024-12-31", window_size=60, epochs=10, batch_size=32):
    train = series.loc[:split_date]
    test = series.loc["2025-01-01":]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

    X_train, y_train = create_lstm_sequences(train_scaled, window_size)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    full_scaled = scaler.transform(series.values.reshape(-1, 1))

    test_start_index = len(train)
    X_test = []

    for i in range(test_start_index, len(series)):
        X_test.append(full_scaled[i-window_size:i, 0])

    X_test = np.array(X_test).reshape((len(X_test), window_size, 1))

    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()

    predictions = pd.Series(predictions, index=test.index)

    metrics = evaluate_model(test, predictions, model_name="LSTM")

    return model, predictions, metrics

from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima_model(
    series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 5),
    split_date="2024-12-31"
):
    """
    Train a SARIMA model using chronological train-test split.
    The seasonal period is set to 5 to approximate weekly trading-day seasonality.
    """
    train = series.loc[:split_date]
    test = series.loc["2025-01-01":]

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False)

    forecast = fitted_model.forecast(steps=len(test))
    forecast.index = test.index

    metrics = evaluate_model(
        test,
        forecast,
        model_name=f"SARIMA{order}x{seasonal_order}"
    )

    return fitted_model, forecast, metrics