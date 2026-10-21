"""Sensitivity analysis: ±20% one-at-a-time weight perturbation per product.

For each component weight we (a) push it +20% and -20%, (b) renormalise weights
to sum to 1, (c) recompute the composite score, (d) rank suppliers in the
latest month, (e) compare with the base ranking. Saves a stability summary CSV
for the Methodology page.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from . import config


COMPONENT_COLS = {
    "supplier_dependency": "supplier_dependency_score_0_100",
    "market_concentration": "concentration_score_0_100",
    "import_anomaly": "import_anomaly_score_0_100",
    "price_stress": "price_stress_score_0_100",
    "logistics_risk": "logistics_risk_score_0_100",
    "shipping_stress": "shipping_stress_score_0_100",
}
COMPONENT_LABELS = {
    "supplier_dependency": "Supplier dependency",
    "market_concentration": "Market concentration",
    "import_anomaly": "Import anomaly",
    "price_stress": "Commodity price stress",
    "logistics_risk": "Logistics risk",
    "shipping_stress": "Shipping stress",
}


def _composite(row: pd.DataFrame, weights: dict) -> pd.Series:
    wsum = sum(weights.values())
    score = sum(row[COMPONENT_COLS[k]].fillna(0) * (v / wsum) for k, v in weights.items())
    return score.clip(0, 100)


def _materiality(df: pd.DataFrame) -> pd.Series:
    share = pd.to_numeric(df.get("supplier_share_value", 0), errors="coerce").fillna(0)
    value = pd.to_numeric(df.get("import_value_usd", 0), errors="coerce").fillna(0)
    return (share >= config.MATERIAL_SHARE_THRESHOLD) | (value >= config.MATERIAL_VALUE_THRESHOLD_USD)


def _latest_per_product(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("date").groupby(["product_name", "supplier_country"]).tail(1)


def run(modeling_df: pd.DataFrame) -> pd.DataFrame:
    """One-at-a-time ±20% sensitivity, computed on the latest month per product."""
    print("[sensitivity] ±20% weight perturbation, one component at a time")
    base_w = dict(config.RISK_WEIGHTS)
    snap = _latest_per_product(modeling_df).copy()
    snap = snap[_materiality(snap)]

    rows = []
    per_product_summary = []

    for prod, sub in snap.groupby("product_name"):
        if len(sub) < 5:
            continue
        sub = sub.copy()
        # Base ranking
        sub["score_base"] = _composite(sub, base_w)
        base_top10 = sub.sort_values("score_base", ascending=False).head(10)["supplier_country"].tolist()

        # Perturb each component ±20%
        overlaps = []
        spearmans = []
        rank_changes = []
        for comp in base_w:
            for direction, mul in (("+20%", 1.20), ("-20%", 0.80)):
                w_pert = dict(base_w)
                w_pert[comp] = w_pert[comp] * mul
                score_pert = _composite(sub, w_pert)
                sub_pert = sub.assign(score_pert=score_pert)
                top10_pert = sub_pert.sort_values("score_pert", ascending=False).head(10)["supplier_country"].tolist()
                overlap = len(set(base_top10) & set(top10_pert)) / max(len(base_top10), 1) * 100

                # Spearman across all material suppliers
                merged = pd.DataFrame({
                    "supplier": sub["supplier_country"].values,
                    "rank_base": sub["score_base"].rank(ascending=False).values,
                    "rank_pert": score_pert.rank(ascending=False).values,
                })
                try:
                    rho, _ = spearmanr(merged["rank_base"], merged["rank_pert"])
                except Exception:
                    rho = np.nan
                avg_rank_change = float((merged["rank_pert"] - merged["rank_base"]).abs().mean())

                rows.append({
                    "Product": prod,
                    "Component": COMPONENT_LABELS[comp],
                    "Perturbation": direction,
                    "Top-10 overlap %": round(overlap, 1),
                    "Spearman rho": round(float(rho) if not np.isnan(rho) else 0.0, 3),
                    "Avg rank change": round(avg_rank_change, 2),
                })
                overlaps.append(overlap)
                spearmans.append(float(rho) if not np.isnan(rho) else 0.0)
                rank_changes.append(avg_rank_change)

        per_product_summary.append({
            "Product": prod,
            "Mean top-10 overlap %": round(float(np.mean(overlaps)), 1),
            "Min top-10 overlap %": round(float(np.min(overlaps)), 1),
            "Mean Spearman rho": round(float(np.mean(spearmans)), 3),
            "Mean rank change": round(float(np.mean(rank_changes)), 2),
            "Stability score": _stability_score(np.mean(overlaps), np.mean(spearmans)),
        })

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(per_product_summary)

    detail.to_csv(config.OUTPUTS_DIR / "sensitivity_analysis_detail.csv", index=False)
    summary.to_csv(config.OUTPUTS_DIR / "sensitivity_analysis_summary.csv", index=False)
    print(f"  saved -> {config.OUTPUTS_DIR / 'sensitivity_analysis_summary.csv'}")
    if not summary.empty:
        print(summary.to_string(index=False))
    return summary


def _stability_score(overlap_pct: float, spearman: float) -> str:
    """Bucket the (overlap, spearman) pair into a plain-English label."""
    if overlap_pct >= 80 and spearman >= 0.85:
        return "Highly stable"
    if overlap_pct >= 70 and spearman >= 0.75:
        return "Stable"
    if overlap_pct >= 60:
        return "Moderately stable"
    return "Fragile"
