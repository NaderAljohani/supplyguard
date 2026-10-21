"""Verify/refresh the composite risk score from upstream component columns.

The upstream multi-product CSV already contains:
  supplyguard_risk_score_0_100, risk_level, top_risk_driver,
  second_risk_driver, risk_explanation_text, contribution_*.

We:
  - re-apply the current risk-level bands (Low<25, Med 25-40, High 40-60, Crit>=60)
  - leave the composite + contributions untouched (upstream weights already match ours)
  - re-export with the multi-product modeling-dataset name
"""
from __future__ import annotations
import pandas as pd
from . import config


COMPONENT_LABELS = {
    "supplier_dependency_score_0_100": "Supplier dependency",
    "concentration_score_0_100": "Market concentration",
    "import_anomaly_score_0_100": "Import drop / anomaly",
    "price_stress_score_0_100": "Rice price stress",
    "logistics_risk_score_0_100": "Logistics (LPI)",
    "shipping_stress_score_0_100": "Shipping stress",
}


def _risk_level(score: float) -> str:
    if pd.isna(score): return "Low"
    for lo, hi, name in config.RISK_LEVELS:
        if lo <= score < hi:
            return name
    return "Critical"


def compute_supplyguard_score(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through: upstream already computed the composite + contributions
    using the same weights as config.RISK_WEIGHTS. We just refresh risk_level."""
    df = df.copy()
    if "supplyguard_risk_score_0_100" not in df.columns:
        # Defensive fallback: recompute from components if upstream score missing
        w = config.RISK_WEIGHTS
        wsum = sum(w.values())
        cols = {
            "supplier_dependency": "supplier_dependency_score_0_100",
            "market_concentration": "concentration_score_0_100",
            "import_anomaly": "import_anomaly_score_0_100",
            "price_stress": "price_stress_score_0_100",
            "logistics_risk": "logistics_risk_score_0_100",
            "shipping_stress": "shipping_stress_score_0_100",
        }
        score = sum(df.get(cols[k], 0).fillna(0) * (v / wsum) for k, v in w.items())
        df["supplyguard_risk_score_0_100"] = score.clip(0, 100).round(2)

    df["risk_level"] = df["supplyguard_risk_score_0_100"].apply(_risk_level)
    return df


def latest_supplier_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (product, supplier) = their most recent month."""
    if df.empty:
        return df
    key = [c for c in ["product_name", "supplier_country"] if c in df.columns]
    if not key:
        return df
    idx = df.groupby(key)["date"].idxmax()
    snap = df.loc[idx].sort_values(
        ["product_name", "supplyguard_risk_score_0_100"], ascending=[True, False]
    ).reset_index(drop=True)
    return snap


def save_modeling_dataset(df: pd.DataFrame) -> None:
    config.MODELING_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.MODELING_FILE, index=False)
    print(f"  saved modeling dataset -> {config.MODELING_FILE}")


def save_latest_snapshot(df: pd.DataFrame) -> None:
    snap = latest_supplier_snapshot(df)
    snap.to_csv(config.LATEST_RISK_FILE, index=False)
    print(f"  saved latest supplier snapshot -> {config.LATEST_RISK_FILE}")
