"""Printed EDA summary across all products."""
from __future__ import annotations
import pandas as pd


def print_summary(df: pd.DataFrame, events: pd.DataFrame) -> None:
    print("\n--- EDA summary ---")
    if "date" in df.columns:
        print(f"date range: {df['date'].min()} -> {df['date'].max()}")
        print(f"n_months: {df['date'].nunique()}")
    if "product_name" in df.columns:
        print(f"products: {sorted(df['product_name'].dropna().unique())}")
        for p, sub in df.groupby("product_name"):
            print(f"  [{p}] suppliers={sub['supplier_country'].nunique()}, "
                  f"rows={len(sub)}, "
                  f"mean_risk={sub['supplyguard_risk_score_0_100'].mean():.2f}, "
                  f"max_risk={sub['supplyguard_risk_score_0_100'].max():.2f}")
    print(f"events: {len(events)}")
