"""Unsupervised anomaly detection on multi-product supplier-month features.

Two feature sets are supported:

* ``ANOMALY_FEATURES_DRIVERS`` — driver-only inputs (import value, MoM/YoY
  changes, supplier share, price stress, shipping stress, logistics risk).
  This is the default; it lets the Isolation Forest find outliers without
  reading the composite risk score, which would be circular (the composite
  is itself built from these drivers).

* ``ANOMALY_FEATURES_WITH_COMPOSITE`` — drivers plus
  ``supplyguard_risk_score_0_100``. Available behind a flag for diagnostic
  comparison; not recommended for production interpretation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from . import config


# Default: driver-only features (no composite risk score, no circularity).
ANOMALY_FEATURES_DRIVERS = [
    "import_value_usd",
    "import_value_usd_mom_pct_change",
    "import_value_usd_yoy_pct_change",
    "supplier_share_value",
    "price_stress_score_0_100",
    "shipping_stress_score_0_100",
    "logistics_risk_score_0_100",
]

# Opt-in: includes the composite score — circular for interpretation, kept
# only as a diagnostic comparison option.
ANOMALY_FEATURES_WITH_COMPOSITE = ANOMALY_FEATURES_DRIVERS + [
    "supplyguard_risk_score_0_100",
]

# Back-compat alias — points to driver-only set so older callers get the
# safe default.
ANOMALY_FEATURES = ANOMALY_FEATURES_DRIVERS


def detect_z_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "import_anomaly_z_score" not in df.columns:
        df["import_anomaly_z_score"] = 0.0
    df["anomaly_z_flag"] = (df["import_anomaly_z_score"].abs() >= config.Z_ABS_THRESHOLD).astype(int)
    return df


def detect_isolation_forest(
    df: pd.DataFrame,
    include_composite_risk: bool = False,
) -> pd.DataFrame:
    """Fit one Isolation Forest per product so the contamination rate is meaningful.

    By default the model uses driver-only features (import value, MoM/YoY,
    supplier share, price stress, shipping stress, logistics risk). Pass
    ``include_composite_risk=True`` to add ``supplyguard_risk_score_0_100``
    back in — this is circular for interpretation and is provided only as
    a diagnostic comparison.
    """
    df = df.copy()
    feat_list = ANOMALY_FEATURES_WITH_COMPOSITE if include_composite_risk else ANOMALY_FEATURES_DRIVERS
    feats = [c for c in feat_list if c in df.columns]
    print(f"  isolation forest features ({'with composite' if include_composite_risk else 'drivers-only'}): {feats}")
    if not feats:
        df["iso_anomaly_score"] = 0.0
        df["iso_anomaly_flag"] = 0
        return df

    df["iso_anomaly_score"] = 0.0
    df["iso_anomaly_flag"] = 0

    if "product_name" not in df.columns:
        groups = [(None, df)]
    else:
        groups = list(df.groupby("product_name"))

    for prod, sub in groups:
        idx = sub.index
        X = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        if len(X) < 20:
            continue
        iso = IsolationForest(
            n_estimators=200, contamination=config.ISO_CONTAMINATION,
            random_state=42, n_jobs=-1,
        ).fit(X)
        raw = -iso.decision_function(X)
        rng = raw.max() - raw.min()
        df.loc[idx, "iso_anomaly_score"] = ((raw - raw.min()) / rng * 100) if rng > 0 else 0.0
        df.loc[idx, "iso_anomaly_flag"] = (iso.predict(X) == -1).astype(int)
    return df


def combine_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["anomaly_score"] = (
        0.5 * df.get("iso_anomaly_score", 0.0)
        + 50 * df.get("anomaly_z_flag", 0).astype(float)
    ).clip(0, 100)
    df["anomaly_flag"] = (
        (df.get("iso_anomaly_flag", 0) == 1) | (df.get("anomaly_z_flag", 0) == 1)
    ).astype(int)

    def _reason(r):
        bits = []
        if r.get("anomaly_z_flag", 0) == 1:
            bits.append(f"YoY-residual z={r.get('import_anomaly_z_score', 0):.1f}")
        if r.get("iso_anomaly_flag", 0) == 1:
            bits.append("isolation-forest outlier")
        return "; ".join(bits) if bits else ""

    df["anomaly_reason"] = df.apply(_reason, axis=1)
    return df


def run_anomaly_pipeline(
    df: pd.DataFrame,
    include_composite_risk: bool = False,
) -> pd.DataFrame:
    """Run the full anomaly pipeline. Driver-only Isolation Forest by default."""
    print("[anomaly_detection] z-score + Isolation Forest per product")
    df = detect_z_anomalies(df)
    df = detect_isolation_forest(df, include_composite_risk=include_composite_risk)
    df = combine_anomalies(df)
    n = int(df["anomaly_flag"].sum())
    print(f"  flagged {n} anomalous rows ({n/len(df)*100:.1f}%)")
    return df


def save_anomaly_results(df: pd.DataFrame) -> None:
    cols = [c for c in [
        "date", "product_name", "supplier_country", "import_value_usd",
        "supplyguard_risk_score_0_100", "risk_level",
        "anomaly_score", "anomaly_flag", "anomaly_reason",
        "import_anomaly_z_score", "iso_anomaly_flag",
    ] if c in df.columns]
    out = df.loc[df["anomaly_flag"] == 1, cols].sort_values(
        "anomaly_score", ascending=False
    )
    out.to_csv(config.ANOMALY_RESULTS_FILE, index=False)
    print(f"  saved -> {config.ANOMALY_RESULTS_FILE}")
