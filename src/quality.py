"""Data quality + missingness summaries — saved for the Methodology page."""
from __future__ import annotations
import pandas as pd
from . import config


# Friendly grouping of fields → high-level "did this group survive preprocessing?"
FIELD_GROUPS = {
    "Trade flow":      ["import_value_usd", "net_weight_kg", "supplier_share_value", "hhi_value", "top3_supplier_share_value"],
    "Commodity price": ["commodity_price_proxy_usd_per_mt", "commodity_price_stress_score_0_100", "price_stress_score_0_100"],
    "Logistics (LPI)": ["lpi_overall_score_filled", "logistics_risk_score_0_100"],
    "Maritime (PortWatch)": [
        "bab_el_mandeb_transit_calls_stress_score_0_100",
        "hormuz_transit_calls_stress_score_0_100",
        "jeddah_incoming_shipment_stress_score_0_100",
        "shipping_stress_score_0_100",
    ],
    "Composite + components": [
        "supplyguard_risk_score_0_100", "supplier_dependency_score_0_100",
        "concentration_score_0_100", "import_anomaly_score_0_100",
    ],
}


def build_quality_summary(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                          modeling_df: pd.DataFrame) -> pd.DataFrame:
    """Per-product row counts, supplier counts, date ranges, etc. Saved as CSV."""
    rows = []
    for prod, sub in modeling_df.groupby("product_name"):
        material = sub[(sub["supplier_share_value"].fillna(0) >= config.MATERIAL_SHARE_THRESHOLD) |
                       (sub["import_value_usd"].fillna(0) >= config.MATERIAL_VALUE_THRESHOLD_USD)]
        lpi_filled = int(sub["lpi_overall_score_filled"].notna().sum()) if "lpi_overall_score_filled" in sub.columns else 0
        rows.append({
            "Product": prod,
            "HS code": int(sub["hs_code"].iloc[0]) if "hs_code" in sub.columns else "",
            "Rows": len(sub),
            "Suppliers": sub["supplier_country"].nunique(),
            "Product-months": sub["date"].nunique(),
            "Date range": f"{sub['date'].min().strftime('%Y-%m')} → {sub['date'].max().strftime('%Y-%m')}",
            "Material supplier-months": len(material),
            "LPI values present": lpi_filled,
            "LPI missing": len(sub) - lpi_filled,
        })
    # Global summary row
    rows.append({
        "Product": "ALL",
        "HS code": "—",
        "Rows": len(modeling_df),
        "Suppliers": modeling_df["supplier_country"].nunique(),
        "Product-months": modeling_df.groupby("product_name")["date"].nunique().sum(),
        "Date range": f"{modeling_df['date'].min().strftime('%Y-%m')} → {modeling_df['date'].max().strftime('%Y-%m')}",
        "Material supplier-months": int(((modeling_df["supplier_share_value"].fillna(0) >= config.MATERIAL_SHARE_THRESHOLD) |
                                          (modeling_df["import_value_usd"].fillna(0) >= config.MATERIAL_VALUE_THRESHOLD_USD)).sum()),
        "LPI values present": int(modeling_df["lpi_overall_score_filled"].notna().sum()) if "lpi_overall_score_filled" in modeling_df.columns else 0,
        "LPI missing": int(modeling_df["lpi_overall_score_filled"].isna().sum()) if "lpi_overall_score_filled" in modeling_df.columns else 0,
    })
    summary = pd.DataFrame(rows)

    # Add dedup info (compare raw vs cleaned by product if possible)
    if "product_name" in raw_df.columns:
        dedup_drops = []
        for prod in summary["Product"]:
            if prod == "ALL":
                dedup_drops.append(len(raw_df) - len(cleaned_df))
                continue
            r = len(raw_df[raw_df["product_name"] == prod])
            c = len(cleaned_df[cleaned_df["product_name"] == prod])
            dedup_drops.append(max(r - c, 0))
        summary["Rows dropped in preprocessing"] = dedup_drops

    summary.to_csv(config.OUTPUTS_DIR / "data_quality_summary.csv", index=False)
    print(f"  saved -> {config.OUTPUTS_DIR / 'data_quality_summary.csv'}")
    return summary


def build_missingness(modeling_df: pd.DataFrame) -> pd.DataFrame:
    """Per-product missingness by field group (mean % nulls across the group's columns)."""
    rows = []
    for prod, sub in modeling_df.groupby("product_name"):
        for group, cols in FIELD_GROUPS.items():
            cols_present = [c for c in cols if c in sub.columns]
            if not cols_present:
                continue
            miss_pct = float(sub[cols_present].isna().mean().mean() * 100)
            rows.append({
                "Product": prod,
                "Field group": group,
                "Missing %": round(miss_pct, 2),
            })
    miss_df = pd.DataFrame(rows)
    miss_df.to_csv(config.OUTPUTS_DIR / "missingness_by_product.csv", index=False)
    print(f"  saved -> {config.OUTPUTS_DIR / 'missingness_by_product.csv'}")
    return miss_df


def run(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, modeling_df: pd.DataFrame) -> None:
    print("[quality] building data quality + missingness summaries")
    build_quality_summary(raw_df, cleaned_df, modeling_df)
    build_missingness(modeling_df)
