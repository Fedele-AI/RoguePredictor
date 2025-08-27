import pandas as pd
from tsfm_public import TimeSeriesForecastingPipeline
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
import torch

def test_granite_ts_forecast():
    # Load sample data
    data = pd.read_csv('data/sample_wave_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # Select relevant columns for forecasting, e.g., wave_height
    target = 'wave_height'
    context_length = 512  # Adjust based on model
    forecast_length = 96  # Predict next 96 hours

    # Initialize the model
    model = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2")

    # Create TimeSeriesPreprocessor
    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        id_columns=[],
        target_columns=[target],
        context_length=context_length,
        prediction_length=forecast_length,
        freq="1h",  # Assuming hourly data
        scaling=True,
    )

    # Train the preprocessor on the data
    tsp.train(data.reset_index())

    # Create pipeline
    pipeline = TimeSeriesForecastingPipeline(
        model=model,
        feature_extractor=tsp,
        explode_forecasts=False,
        inverse_scale_outputs=True,
        device="cpu"  # Use CPU for CI/CD
    )

    # Prepare data as DataFrame with proper structure
    input_data = data.reset_index()  # Reset index to have timestamp as column

    # Make forecast
    forecast = pipeline(input_data)

    # Assert that forecast is generated
    assert not forecast.empty, "Forecast DataFrame is empty"
    assert f"{target}_prediction" in forecast.columns, f"Prediction column {target}_prediction not found"

    # Additional checks: e.g., forecast values are reasonable (not NaN, within range)
    predictions = forecast[f"{target}_prediction"].iloc[0]
    assert not any(pd.isna(predictions)), "Forecast contains NaN values"
    assert all(p >= 0 for p in predictions), "Forecast contains negative values (invalid for wave height)"

    print("GraniteTS forecast test passed!")

if __name__ == "__main__":
    test_granite_ts_forecast()
