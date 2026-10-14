# src/features.py

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


DEFAULT_LOOKBACK = 30

# Features known at prediction time.
# Sales is included as historical input, never as a same-day input
# for predicting its own target.
BASE_FEATURES = [
    "Sales",
    "Open",
    "Promo",
    "SchoolHoliday",
    "StateHoliday",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "DayOfMonth_sin",
    "DayOfMonth_cos",
    "Month_sin",
    "Month_cos",
]


def create_calendar_features(df):
    """
    Extract calendar and business-condition features.

    StateHoliday is encoded as:
        0: no state holiday
        1: public holiday category 'a'
        2: Easter holiday category 'b'
        3: Christmas category 'c'
        other: unknown/unexpected category

    Returns a new DataFrame; does not mutate the input.
    """
    result = df.copy()

    result["Date"] = pd.to_datetime(result["Date"])
    dates = result["Date"]

    # Calendar variables.
    # Rossmann DayOfWeek: Monday=1, ..., Sunday=7.
    result["DayOfWeek"] = dates.dt.dayofweek + 1
    result["DayOfMonth"] = dates.dt.day
    result["Month"] = dates.dt.month
    result["DayOfYear"] = dates.dt.dayofyear

    # Cyclical encoding preserves periodicity.
    result["DayOfWeek_sin"] = np.sin(
        2 * np.pi * (result["DayOfWeek"] - 1) / 7
    )
    result["DayOfWeek_cos"] = np.cos(
        2 * np.pi * (result["DayOfWeek"] - 1) / 7
    )

    result["DayOfMonth_sin"] = np.sin(
        2 * np.pi * (result["DayOfMonth"] - 1) / 31
    )
    result["DayOfMonth_cos"] = np.cos(
        2 * np.pi * (result["DayOfMonth"] - 1) / 31
    )

    result["Month_sin"] = np.sin(
        2 * np.pi * (result["Month"] - 1) / 12
    )
    result["Month_cos"] = np.cos(
        2 * np.pi * (result["Month"] - 1) / 12
    )

    # Explicitly encode StateHoliday.
    holiday_map = {"0": 0, "a": 1, "b": 2, "c": 3}
    result["StateHoliday"] = (
        result["StateHoliday"]
        .astype(str)
        .str.strip()
        .map(holiday_map)
        .fillna(4)
        .astype(np.float32)
    )

    # Ensure business indicators are numeric.
    for col in ("Open", "Promo", "SchoolHoliday"):
        result[col] = pd.to_numeric(
            result[col], errors="coerce"
        )

    # If an indicator is unavailable, keep preprocessing deterministic.
    # data.py has already filled standard Rossmann missing indicators.
    result[["Open", "Promo", "SchoolHoliday"]] = (
        result[["Open", "Promo", "SchoolHoliday"]]
        .fillna(0)
        .astype(np.float32)
    )

    return result


def add_lag_features(df, lags=(1, 7, 14, 30), rolling_windows=(7, 30)):
    """
    Add lag and rolling features from past sales only.

    Rolling statistics are shifted by one row to avoid including
    the current day's target sales.
    """
    result = df.sort_values("Date").copy()

    for lag in lags:
        result[f"Sales_lag_{lag}"] = result["Sales"].shift(lag)

    past_sales = result["Sales"].shift(1)

    for window in rolling_windows:
        result[f"Sales_rolling_mean_{window}"] = (
            past_sales.rolling(window=window, min_periods=window).mean()
        )
        result[f"Sales_rolling_std_{window}"] = (
            past_sales.rolling(window=window, min_periods=window).std()
        )

    return result


def get_feature_columns(include_lags=False):
    """Return the ordered list of model input features."""
    features = list(BASE_FEATURES)

    if include_lags:
        features += [
            "Sales_lag_1",
            "Sales_lag_7",
            "Sales_lag_14",
            "Sales_lag_30",
            "Sales_rolling_mean_7",
            "Sales_rolling_std_7",
            "Sales_rolling_mean_30",
            "Sales_rolling_std_30",
        ]

    return features


def prepare_features(df, include_lags=False):
    """
    Create model features from a Store 1 dataframe.

    The official test data has no Sales column. This function is
    intended for labeled historical data; use calendar features
    directly for official test rows.
    """
    if "Sales" not in df.columns:
        raise ValueError(
            "Sales is required to build historical sales features. "
            "Official test data has no target; use "
            "create_calendar_features() for test covariates."
        )

    result = create_calendar_features(df)

    if include_lags:
        result = add_lag_features(result)

    return result


def fit_feature_scaler(
    train_df,
    feature_columns=None,
    include_lags=False,
):
    """
    Fit a StandardScaler using training observations only.

    Rows with missing feature values (e.g. initial lag windows) are
    excluded when fitting. The scaler is never fit on validation,
    holdout, or official test data.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(include_lags)

    missing = set(feature_columns) - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    fit_data = train_df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if fit_data.empty:
        raise ValueError("No complete rows available to fit the scaler.")

    scaler = StandardScaler()
    scaler.fit(fit_data)

    return scaler


def transform_features(
    df,
    scaler,
    feature_columns=None,
    include_lags=False,
):
    """
    Apply a previously fitted scaler.

    Does not fit or refit the scaler.
    Returns a float32 NumPy array with NaNs preserved.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(include_lags)

    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    values = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    ).to_numpy(dtype=np.float64)

    valid_rows = np.isfinite(values).all(axis=1)
    transformed = np.full(values.shape, np.nan, dtype=np.float32)

    if valid_rows.any():
        transformed[valid_rows] = scaler.transform(
            values[valid_rows]
        ).astype(np.float32)

    return transformed


def fit_target_scaler(train_df):
    """Fit a separate scaler on training Sales only."""
    sales = train_df["Sales"].dropna().to_numpy().reshape(-1, 1)

    if len(sales) == 0:
        raise ValueError("No valid Sales values available for scaling.")

    scaler = StandardScaler()
    scaler.fit(sales)

    return scaler


def transform_target(sales, scaler):
    """Scale a target array using the training-fitted scaler."""
    values = np.asarray(sales, dtype=np.float64).reshape(-1, 1)
    return scaler.transform(values).astype(np.float32)


def inverse_transform_target(scaled_sales, scaler):
    """Convert scaled model outputs back to original Sales units."""
    values = np.asarray(
        scaled_sales, dtype=np.float64
    ).reshape(-1, 1)

    return scaler.inverse_transform(values).ravel()


def make_sequences(
    feature_array,
    target_array,
    dates,
    target_indices,
    lookback=DEFAULT_LOOKBACK,
):
    """
    Construct LSTM sequences for selected target row indices.

    Each sample uses the previous `lookback` consecutive calendar
    days to predict Sales on the target date.

    Historical context from earlier splits is allowed, but the
    target must belong to target_indices. A sequence is skipped
    when any feature is missing or a calendar date is missing.

    Parameters
    ----------
    feature_array : np.ndarray, shape (n_rows, n_features)
        Already transformed feature matrix.
    target_array : np.ndarray, shape (n_rows,) or (n_rows, 1)
        Target Sales values, scaled with the training target scaler.
    dates : array-like
        Dates corresponding to each row.
    target_indices : iterable of int
        Row indices whose targets should be predicted.
    lookback : int
        Number of preceding days used as context.

    Returns
    -------
    X : np.ndarray, shape (n_samples, lookback, n_features)
    y : np.ndarray, shape (n_samples, 1)
    kept_indices : np.ndarray
        Target row indices retained in the sequence dataset.
    """
    feature_array = np.asarray(feature_array, dtype=np.float32)
    target_array = np.asarray(
        target_array, dtype=np.float32
    ).reshape(-1)

    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)

    if len(feature_array) != len(target_array) or len(dates) != len(target_array):
        raise ValueError("Features, targets and dates must have equal lengths.")

    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    X, y, kept_indices = [], [], []

    for idx in target_indices:
        idx = int(idx)
        start = idx - lookback

        if start < 0:
            continue

        # Only preceding rows are inputs; current target is excluded.
        x_window = feature_array[start:idx]
        y_value = target_array[idx]

        if len(x_window) != lookback:
            continue

        if not np.isfinite(x_window).all() or not np.isfinite(y_value):
            continue

        # Ensure observations are on consecutive calendar days.
        window_dates = dates.iloc[start:idx + 1]
        day_diffs = window_dates.diff().dropna().dt.days.to_numpy()

        if not np.all(day_diffs == 1):
            continue

        X.append(x_window)
        y.append([y_value])
        kept_indices.append(idx)

    if not X:
        n_features = feature_array.shape[1]
        return (
            np.empty((0, lookback, n_features), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(kept_indices, dtype=np.int64),
    )