# src/data.py

from pathlib import Path
import warnings

import pandas as pd


# Project root: inventory-optimization-lstm/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def load_rossmann_data(
    data_dir=None,
    store_id=1,
):
    """
    Load and prepare Rossmann train/test data for a single store.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing train.csv and test.csv.
    store_id : int, default=1
        Store identifier to retain.

    Returns
    -------
    train_df : pandas.DataFrame
        Labeled historical sales data for the selected store.
    test_df : pandas.DataFrame
        Official Kaggle test data for the selected store.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_train = {
        "Store", "Date", "Sales", "Customers", "Open",
        "Promo", "StateHoliday", "SchoolHoliday",
    }
    required_test = {
        "Store", "Date", "Open",
        "Promo", "StateHoliday", "SchoolHoliday",
    }

    missing_train = required_train - set(train_df.columns)
    missing_test = required_test - set(test_df.columns)

    if missing_train:
        raise ValueError(
            f"Missing training columns: {sorted(missing_train)}"
        )

    if missing_test:
        raise ValueError(
            f"Missing test columns: {sorted(missing_test)}"
        )

    # Select Store 1 only.
    train_df = train_df.loc[
        train_df["Store"] == store_id
    ].copy()

    test_df = test_df.loc[
        test_df["Store"] == store_id
    ].copy()

    if train_df.empty:
        raise ValueError(f"No training data found for Store {store_id}")

    # Parse dates and remove rows with invalid dates.
    train_df["Date"] = pd.to_datetime(
        train_df["Date"], errors="coerce"
    )
    test_df["Date"] = pd.to_datetime(
        test_df["Date"], errors="coerce"
    )

    train_df = train_df.dropna(subset=["Date"])
    test_df = test_df.dropna(subset=["Date"])

    # Sort chronologically before any temporal split.
    train_df = train_df.sort_values("Date").reset_index(drop=True)
    test_df = test_df.sort_values("Date").reset_index(drop=True)

    # A store should have at most one record per date.
    if train_df.duplicated(["Store", "Date"]).any():
        raise ValueError(
            "Duplicate Store/Date records found in training data."
        )

    if test_df.duplicated(["Store", "Date"]).any():
        raise ValueError(
            "Duplicate Store/Date records found in test data."
        )

    # Missing targets are not interpreted as zero sales.
    missing_sales = train_df["Sales"].isna().sum()

    if missing_sales:
        warnings.warn(
            f"Dropping {missing_sales} rows with missing Sales. "
            "Missing targets are not replaced with zero."
        )
        train_df = train_df.dropna(subset=["Sales"]).copy()

    # Sales should not be negative. Do not silently clip or modify them.
    negative_sales = (train_df["Sales"] < 0).sum()

    if negative_sales:
        warnings.warn(
            f"Found {negative_sales} negative Sales observations. "
            "Inspect these records before training."
        )

    # Fill missing exogenous variables conservatively.
    # Unknown categorical holidays are kept as an explicit category.
    for df in (train_df, test_df):
        if "StateHoliday" in df.columns:
            df["StateHoliday"] = (
                df["StateHoliday"]
                .fillna("0")
                .astype(str)
            )

        for col in ("Open", "Promo", "SchoolHoliday"):
            if col in df.columns:
                df[col] = df[col].fillna(0)

    # Customers is useful for EDA, but should not be a forecasting input.
    # Its missing values are intentionally not imputed here.

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def chronological_split(
    train_df,
    train_ratio=0.70,
    val_ratio=0.15,
    holdout_ratio=0.15,
):
    """
    Split labeled data chronologically into train, validation and holdout.

    No random shuffling is performed.

    Returns
    -------
    dict
        {"train": ..., "validation": ..., "holdout": ...}
    """
    total_ratio = train_ratio + val_ratio + holdout_ratio

    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")

    if min(train_ratio, val_ratio, holdout_ratio) <= 0:
        raise ValueError("All split ratios must be positive.")

    df = train_df.sort_values("Date").reset_index(drop=True)
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end < 1 or val_end <= train_end or val_end >= n:
        raise ValueError("Not enough observations for the requested split.")

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:val_end].copy()
    holdout = df.iloc[val_end:].copy()

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "holdout": holdout.reset_index(drop=True),
    }


def print_data_summary(train_df, test_df):
    """Print a concise summary for Store 1."""
    print("=== Rossmann Store Sales: Data Summary ===")
    print(f"Training observations: {len(train_df):,}")
    print(f"Test observations:     {len(test_df):,}")
    print(f"Training date range:   {train_df['Date'].min()} "
          f"to {train_df['Date'].max()}")

    if not test_df.empty:
        print(f"Test date range:       {test_df['Date'].min()} "
              f"to {test_df['Date'].max()}")

    print(f"Missing training values:\n{train_df.isna().sum()}")
    print(f"Missing test values:\n{test_df.isna().sum()}")

    if "Sales" in train_df.columns:
        print("\nSales statistics:")
        print(train_df["Sales"].describe())

    if "Open" in train_df.columns:
        print("\nOpen status counts:")
        print(train_df["Open"].value_counts(dropna=False))


if __name__ == "__main__":
    train_df, test_df = load_rossmann_data()
    print_data_summary(train_df, test_df)

    splits = chronological_split(train_df)

    print("\n=== Chronological Split ===")
    for name, subset in splits.items():
        print(
            f"{name:10s}: {len(subset):,} rows | "
            f"{subset['Date'].min().date()} "
            f"-> {subset['Date'].max().date()}"
        )