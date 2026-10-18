from pathlib import Path
import random
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from typing import Sequence, Optional
from gluonts.dataset.common import ListDataset
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Is CUDA available?:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
from helpers.forecast_eval import (
    ForecastResult,
    aggregate_metrics_by_asset_type,
    aggregate_chunked_block_results_by_item,
    build_chunk_holdout_entries,
    ensure_paths,
    listdataset,
    load_dataset_frame,
    load_taildart_from_ckpt,
    macro_summary,
    materialize_forecast_results,
    run_forecasts_raw,
    run_one_step_probabilistic_forecasts,
    DEFAULT_METRIC_COLUMNS,
)
from helpers.plotter import plot_metrics_by_asset_type, plot_metrics_heatmap_by_asset_type, plot_model_comparison_by_asset_type


TAILDART_ROOT, PROJECT_ROOT = ensure_paths()
print(TAILDART_ROOT)

from taildart.training import load_model_config

DEVICE            = "cuda"  # set to 'mps' on macbook runs
BATCH_SIZE        = 8
PREDICTION_LENGTH = 1 # 20 --> ValueError: use_single_pass_sampling=True is only valid for prediction_length=1;  multi-step forecasts would condition all samples on a shared mean trajectory. # one-step to mirror the taildart returns notebook
EVAL_NUM_SAMPLES  = 100
CONTEXT_LENGTH    = 250
HORIZON           = PREDICTION_LENGTH #1 #20
TRAIN_SIZE        = 800
DEMO_FREQ         = "D"
######################################################################################

######################################################################################
# Prepare a time series
# Input:  a csv/excel file, input context (N, e.g. 250 time points), output horizon (H, e.g. 1), sliding window (e.g. 1)
# Output: a dataframe of time series: columns: X_0, X_1, ..., X_N, Y_0, Y_1, ..., Y_H
######################################################################################
# -------------------------------------------------------------------
# Create a synthetic daily time series
# -------------------------------------------------------------------
def create_synthetic_series(
    n_observations: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Daily synthetic return-like series with seasonal structure,
    autocorrelation, and random shocks.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2020-01-01", periods=n_observations, freq=DEMO_FREQ)
    noise = rng.normal(loc=0.0, scale=0.01, size=n_observations)
    
    values = np.zeros(n_observations)
    
    for t in range(1, n_observations):
        seasonal_signal = 0.01 * np.sin(2 * np.pi * t / 60)
        values[t] = (
            0.50 * values[t - 1]
            + seasonal_signal
            + noise[t]
        )

    return pd.DataFrame(
        {
            "date": dates,
            "values": values,
        }
    )


# -------------------------------------------------------------------
# Convert one time series into rolling context windows
# -------------------------------------------------------------------
def create_context_windows(ts_df: pd.DataFrame,
                           context_length: int = 250,
                           horizon: int = 20,
                           first_forecast_index: int | None = None,
                           ) -> pd.DataFrame:
    """
    Creates one row per forecasting origin.

    Each row contains:
    - item_id
    - forecast_start: timestamp of the first predicted value
    - X_0 ... X_(context_length - 1): historical context

    The function produces only origins for which a complete future horizon
    exists in the source series.
    """
    df = ts_df.sort_values("date").reset_index(drop=True)

    if first_forecast_index is None:
        first_forecast_index = context_length

    if first_forecast_index < context_length:
        raise ValueError("first_forecast_index must be at least context_length.")

    rows = []

    # `origin` is the index of the first future value to forecast.
    for origin in range(first_forecast_index, len(df) - horizon + 1):
        context = df["values"].iloc[
            origin - context_length : origin
        ].to_numpy()

        row = {
            "item_id": f"window_{origin}",
            "forecast_start": df["date"].iloc[origin],
        }

        row.update(
            {
                f"X_{i}": context[i]
                for i in range(context_length)
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


# Create the underlying synthetic series.
synthetic_ts_df = create_synthetic_series(n_observations=1200,
                                          seed=42,
                                          )

# Create forecasting origins only after the nominal training period.
# In practice, `predictor` should have been trained on observations before
# TRAIN_SIZE, never on future observations.
test_windows_df = create_context_windows(ts_df=synthetic_ts_df,
                                         context_length=CONTEXT_LENGTH,
                                         horizon=HORIZON,
                                         first_forecast_index=TRAIN_SIZE,
                                         )

context_columns = [f"X_{i}" for i in range(CONTEXT_LENGTH)]
#####################################################################################

######################################################################################
# Set the model from the checkpoint
######################################################################################
base_ckpt_path    = TAILDART_ROOT / "final_checkpoints/taildart_dense_base_johnsonsu.ckpt"
model_config_path = TAILDART_ROOT / "configs" / "taildart_dense_johnsonsu.json"
model_cfg         = load_model_config(model_config_path, model_size="base") # ! ATTENTION ! model_size is manually set 

estimator, module, predictor = load_taildart_from_ckpt(
    ckpt_path=base_ckpt_path,
    prediction_length=PREDICTION_LENGTH,
    context_length=CONTEXT_LENGTH,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    num_parallel_samples=EVAL_NUM_SAMPLES,
    use_single_pass_sampling=True,
    model_config=model_cfg,
)
######################################################################################

######################################################################################
# Zero-Shot Forecasting of the time series data stored in the dataframe using the model 
######################################################################################
# -------------------------------------------------------------------
# Probabilistic Forecast 
# -------------------------------------------------------------------
def run_forecasts_from_context_windows(
        *,
        predictor,
        windows_df: pd.DataFrame,
        context_columns: Sequence[str],
        freq: str,
        item_id_col: Optional[str] = None,
        forecast_start_col: Optional[str] = None,
        num_samples: int = 100,
) -> list:
    """
    Generate probabilistic multi-step forecasts for context windows stored
    row-wise in a DataFrame.

    Parameters
    ----------
    predictor
        A trained predictor. Its prediction_length determines the
        number of future periods forecast for every input row.

    windows_df
        One row per forecast origin/context window.

    context_columns
        Ordered columns containing the historical input window, for example:
        ['X_0', 'X_1', ..., 'X_249'].

    freq
        Frequency string, e.g. 'B', 'D', 'W', or 'M'.

    item_id_col
        Optional column containing a unique identifier for each window.
        If omitted, row numbers are used.

    forecast_start_col
        Optional column giving the timestamp of the first forecasted value.
        This is recommended when the model uses calendar/time features.

    num_samples
        Number of predictive sample paths to draw per window.

    Returns
    -------
    list
        One GluonTS probabilistic Forecast object per DataFrame row.
        Each forecast contains `predictor.prediction_length` future steps.
    """
    missing = set(context_columns) - set(windows_df.columns)
    if missing:
        raise ValueError(f"Missing context columns: {sorted(missing)}")
    
    if item_id_col is not None and item_id_col not in windows_df.columns:
        raise ValueError(f"item_id_col '{item_id_col}' is not in windows_df.")
    
    if forecast_start_col is not None and forecast_start_col not in windows_df.columns:
        raise ValueError(
            f"forecast_start_col '{forecast_start_col}' is not in windows_df."
        )
    
    context_length = len(context_columns)
    dataset_entries = []
    
    for row_index, row in windows_df.iterrows():
        context = row.loc[list(context_columns)].to_numpy(dtype=np.float32)
        
        if np.isnan(context).any():
            raise ValueError(
                f"Context window at row {row_index} contains missing values."
            )
        
        item_id = (
            str(row[item_id_col])
            if item_id_col is not None
            else f"window_{row_index}"
        )
        
        # GluonTS requires a start time even if no explicit dates are available.
        # When the forecast start is known, derive the start of its context window.
        if forecast_start_col is not None:
            forecast_start = pd.Timestamp(row[forecast_start_col])
            
            context_start = pd.date_range(
                end=forecast_start,
                periods=context_length + 1,
                freq=freq,
            )[0]
        else:
            # Valid for models without calendar-dependent features.
            context_start = pd.Timestamp("2000-01-01")

        dataset_entries.append(
            {
                "start": pd.Timestamp(context_start),
                "target": context,
                "item_id": item_id,
            }
        )

    context_ds = ListDataset(dataset_entries, freq=freq)

    # This performs the actual model inference.
    forecasts = list(
        predictor.predict(
            context_ds,
            num_samples=int(num_samples),
        )
    )

    return forecasts


# -------------------------------------------------------------------
# Convert GluonTS Forecast objects into prediction columns
# -------------------------------------------------------------------
def forecasts_to_dataframe(
    windows_df: pd.DataFrame,
    forecasts: list,
    horizon: int,
) -> pd.DataFrame:
    """
    Adds median, mean, and 80% prediction intervals to each context window.
    """
    if len(windows_df) != len(forecasts):
        raise ValueError(
            "Number of context windows and forecasts must be identical."
        )

    output_df = windows_df.copy()

    for row_index, forecast in enumerate(forecasts):
        if forecast.prediction_length != horizon:
            raise ValueError(
                f"Expected horizon={horizon}, but received "
                f"{forecast.prediction_length}."
            )

        # Point forecasts
        forecast_mean = forecast.mean
        forecast_median = forecast.quantile(0.5)

        # Central 80% prediction interval
        lower_80 = forecast.quantile(0.1)
        upper_80 = forecast.quantile(0.9)

        for h in range(horizon):
            output_df.loc[row_index, f"Pred_Mean_Y_{h}"] = forecast_mean[h]
            output_df.loc[row_index, f"Pred_Median_Y_{h}"] = forecast_median[h]
            output_df.loc[row_index, f"PI80_Lower_Y_{h}"] = lower_80[h]
            output_df.loc[row_index, f"PI80_Upper_Y_{h}"] = upper_80[h]

    return output_df


forecasts = run_forecasts_from_context_windows(predictor=predictor,
                                               windows_df=test_windows_df,
                                               context_columns=context_columns,
                                               freq=DEMO_FREQ,
                                               item_id_col="item_id",
                                               forecast_start_col="forecast_start",
                                               #num_samples=NUM_SAMPLES,
                                               num_samples=EVAL_NUM_SAMPLES,
                                               )

# Add predicted horizons to the context-window DataFrame.
output_df = forecasts_to_dataframe(windows_df=test_windows_df,
                                   forecasts=forecasts,
                                   horizon=HORIZON,
                                   )

# Save inputs and forecasts together.
output_path = Path("synthetic_multihorizon_forecasts.csv")
output_df.to_csv(output_path, index=False)

print(f"Saved {len(output_df)} multi-horizon forecasts to: {output_path}")
print(output_df.head())
######################################################################################
