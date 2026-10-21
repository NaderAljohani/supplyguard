"""Light feature engineering on top of the upstream multi-product dataset.

The upstream file already contains all component risk scores. We only add:
  - import_anomaly_z_score (for anomaly detection)
  - event-proximity flags
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config


def add_import_anomaly_z(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score of monthly import value vs a 12-month rolling baseline,
    grouped by (product, supplier)."""
    df = df.copy()
    if "import_value_usd" not in df.columns:
        return df
    group = [c for c in ["product_name", "supplier_country"] if c in df.columns]
    if not group:
        return df
    df = df.sort_values(group + ["date"])
    g = df.groupby(group, group_keys=False)
    roll_mean = g["import_value_usd"].transform(lambda s: s.rolling(12, min_periods=3).mean())
    roll_std = g["import_value_usd"].transform(lambda s: s.rolling(12, min_periods=3).std())
    z = (df["import_value_usd"] - roll_mean) / roll_std.replace(0, np.nan)
    df["import_anomaly_z_score"] = z.fillna(0)
    df["import_anomaly_flag"] = (df["import_anomaly_z_score"].abs() >= config.Z_ABS_THRESHOLD).astype(int)
    return df


def add_event_proximity(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if events is None or events.empty or "date" not in events.columns:
        df["event_window_flag"] = 0
        df["days_to_nearest_event"] = np.nan
        df["disruption_event_name"] = ""
        return df

    ev = events.dropna(subset=["date"]).copy()
    ev_dates = ev["date"].values.astype("datetime64[ns]")
    ev_names = ev["event_name"].astype(str).values
    df_dates = df["date"].values.astype("datetime64[ns]")

    nearest_idx = np.empty(len(df_dates), dtype=int)
    nearest_dt = np.empty(len(df_dates), dtype="timedelta64[ns]")
    for i, d in enumerate(df_dates):
        diffs = np.abs(ev_dates - d)
        j = int(np.argmin(diffs))
        nearest_idx[i] = j
        nearest_dt[i] = diffs[j]

    df["days_to_nearest_event"] = pd.to_timedelta(nearest_dt).days
    df["disruption_event_name"] = ev_names[nearest_idx]
    delta_days = (df["date"].values.astype("datetime64[ns]") - ev_dates[nearest_idx]) / np.timedelta64(1, "D")
    df["event_window_flag"] = (
        (delta_days >= -config.EVENT_WINDOW_DAYS_BEFORE) &
        (delta_days <= config.EVENT_WINDOW_DAYS_AFTER)
    ).astype(int)
    return df


def build_all_features(df: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    print("[feature_engineering] adding anomaly z-score + event proximity")
    df = add_import_anomaly_z(df)
    df = add_event_proximity(df, events if events is not None else pd.DataFrame())
    print(f"  feature dataframe shape: {df.shape}")
    return df
