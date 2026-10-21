"""Clean & standardize the multi-product food imports dataset."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _standardize_supplier_names(df: pd.DataFrame) -> pd.DataFrame:
    if "supplier_country" not in df.columns:
        return df
    df["supplier_country"] = (
        df["supplier_country"]
        .astype(str)
        .str.strip()
        .replace(config.COUNTRY_RENAMES)
    )
    return df


def _build_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif {"year", "month"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            df[["year", "month"]].assign(day=1), errors="coerce"
        )
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def _risk_level(score: float) -> str:
    """Apply the current threshold bands (Low<25, Med 25-40, High 40-60, Crit>=60)."""
    if pd.isna(score):
        return "Low"
    for lo, hi, name in config.RISK_LEVELS:
        if lo <= score < hi:
            return name
    return "Critical"


def clean_main(df: pd.DataFrame) -> pd.DataFrame:
    """Clean & standardize the multi-product main dataset.

    Key: product_name × supplier_country × date.
    """
    print("[preprocessing] cleaning multi-product dataset")
    df = df.copy()
    df = _build_date(df)
    df = _standardize_supplier_names(df)

    # Aggregate-supplier rows (rare, but guard)
    if "supplier_country" in df.columns:
        agg = df["supplier_country"].isin(config.AGGREGATE_SUPPLIER_NAMES)
        if agg.any():
            print(f"  dropped {agg.sum()} aggregate-supplier rows")
            df = df.loc[~agg].copy()

    # Numeric coercion
    numeric_keywords = ("value", "kg", "share", "score", "hhi", "price", "lpi",
                         "stress", "zscore", "ratio", "pct", "mean", "std", "cv", "contribution")
    numeric_cols = [c for c in df.columns
                    if any(k in c.lower() for k in numeric_keywords) and df[c].dtype == object]
    df = _coerce_numeric(df, numeric_cols)

    # Validate share columns
    for share_col in ("supplier_share_value", "top3_supplier_share_value", "hhi_value"):
        if share_col in df.columns:
            mx = df[share_col].max(skipna=True)
            if pd.notna(mx) and mx > 1.5:
                df[share_col] = df[share_col] / 100.0
            df[share_col] = df[share_col].clip(lower=0.0, upper=1.0)

    # Clip *_score_0_100 columns
    for c in df.columns:
        if c.endswith("_score_0_100"):
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(0, 100)

    # Friendly alias columns the app expects
    if "supplier_share_value" in df.columns and "supplier_share" not in df.columns:
        df["supplier_share"] = df["supplier_share_value"]
    if "top3_supplier_share_value" in df.columns and "top_3_supplier_share" not in df.columns:
        df["top_3_supplier_share"] = df["top3_supplier_share_value"]
    if "hhi_value" in df.columns and "HHI" not in df.columns:
        df["HHI"] = df["hhi_value"]

    # Map upstream component names → canonical ones the rest of the app expects
    rename_map = {
        "supplier_dependency_risk_score_0_100": "supplier_dependency_score_0_100",
        "market_concentration_risk_score_0_100": "concentration_score_0_100",
        "import_anomaly_risk_score_0_100": "import_anomaly_score_0_100",
        "commodity_price_stress_score_0_100": "price_stress_score_0_100",
        "product_route_shipping_stress_score_0_100": "shipping_stress_score_0_100",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Re-apply current risk-level bands (the CSV was bucketed with old/upstream thresholds)
    if "supplyguard_risk_score_0_100" in df.columns:
        df["risk_level"] = df["supplyguard_risk_score_0_100"].apply(_risk_level)

    # Dedupe per (date, product, supplier)
    key = [c for c in ["date", "product_name", "supplier_country"] if c in df.columns]
    if key:
        before = len(df)
        df = df.drop_duplicates(subset=key, keep="last")
        if len(df) < before:
            print(f"  dropped {before - len(df)} duplicate {tuple(key)} rows")

    sort_key = [c for c in ["product_name", "supplier_country", "date"] if c in df.columns]
    df = df.sort_values(sort_key).reset_index(drop=True)
    print(f"  cleaned shape: {df.shape}")
    if "product_name" in df.columns:
        print(f"  products: {sorted(df['product_name'].dropna().unique())}")
    return df


def save_cleaned(df: pd.DataFrame) -> None:
    config.CLEANED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEANED_FILE, index=False)
    print(f"  saved -> {config.CLEANED_FILE}")
