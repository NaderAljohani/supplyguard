"""Backtest the multi-product risk score against known events."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config


def backtest_event(df: pd.DataFrame, event_date: pd.Timestamp, *,
                   product: str | None = None,
                   country: str | None = None,
                   window_before: int = 90, window_after: int = 180) -> pd.DataFrame:
    if "date" not in df.columns:
        return pd.DataFrame()
    mask = (df["date"] >= event_date - pd.Timedelta(days=window_before)) & \
           (df["date"] <= event_date + pd.Timedelta(days=window_after))
    sub = df.loc[mask].copy()
    if product and "product_name" in sub.columns:
        products = {value.strip() for value in product.split(";")}
        sub = sub[sub["product_name"].isin(products)]
    if country and country.lower() not in ("global", "shipping route"):
        if "supplier_country" in sub.columns:
            countries = {value.strip().lower() for value in country.split("/")}
            sub = sub[sub["supplier_country"].str.lower().isin(countries)]
    if sub.empty:
        return sub
    agg_cols = [c for c in [
        "supplyguard_risk_score_0_100", "supplier_dependency_score_0_100",
        "concentration_score_0_100", "import_anomaly_score_0_100",
        "price_stress_score_0_100", "logistics_risk_score_0_100",
        "shipping_stress_score_0_100",
    ] if c in sub.columns]
    grp = sub.groupby("date")[agg_cols].mean().reset_index()
    grp["days_to_event"] = (grp["date"] - event_date).dt.days
    return grp


def precision_at_k(df: pd.DataFrame, events: pd.DataFrame, k: int = 10, window: int = 60) -> dict:
    if "anomaly_score" not in df.columns or events.empty:
        return {"k": k, "precision_at_k": None, "n_hits": 0}
    monthly = (df.groupby("date")["anomaly_score"].mean()
                 .sort_values(ascending=False).head(k))
    hits = 0
    for d in monthly.index:
        for ev in events["date"].dropna():
            if abs((d - ev).days) <= window:
                hits += 1
                break
    return {"k": k, "precision_at_k": round(hits / k, 3), "n_hits": hits}


def run_event_backtest(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    print("[evaluation] backtesting risk score against known events")
    if events.empty:
        return pd.DataFrame()
    rows = []
    product_data_max = df["date"].max() if "date" in df.columns else pd.NaT
    for _, ev in events.iterrows():
        if pd.isna(ev.get("date")):
            continue
        product = ev.get("affected_product") if isinstance(ev.get("affected_product"), str) else None
        country = ev.get("affected_country") if isinstance(ev.get("affected_country"), str) else None
        in_range = bool(pd.notna(product_data_max) and ev["date"] <= product_data_max + pd.Timedelta(days=30))
        sub = backtest_event(df, ev["date"], product=product, country=country)
        pre = sub[sub["days_to_event"] < 0]["supplyguard_risk_score_0_100"] if not sub.empty else pd.Series(dtype=float)
        during = sub[(sub["days_to_event"] >= 0) & (sub["days_to_event"] <= 60)]["supplyguard_risk_score_0_100"] if not sub.empty else pd.Series(dtype=float)
        post = sub[sub["days_to_event"] > 60]["supplyguard_risk_score_0_100"] if not sub.empty else pd.Series(dtype=float)
        rows.append({
            "event_name": ev.get("event_name"),
            "event_date": ev["date"],
            "affected_product": product,
            "affected_country": country,
            "event_in_product_data": in_range,
            "risk_pre_event_mean": round(pre.mean(), 2) if not pre.empty else np.nan,
            "risk_during_event_mean": round(during.mean(), 2) if not during.empty else np.nan,
            "risk_post_event_mean": round(post.mean(), 2) if not post.empty else np.nan,
            "delta_during_minus_pre": (round(during.mean() - pre.mean(), 2)
                                       if (not pre.empty and not during.empty) else np.nan),
            "n_months_in_window": len(sub),
        })
    out = pd.DataFrame(rows)
    pak = precision_at_k(df, events, k=10, window=60)
    print(f"  precision@10 (anomaly months within 60d of any event): {pak['precision_at_k']}")
    out.attrs["precision_at_10"] = pak
    out.to_csv(config.EVENT_BACKTEST_FILE, index=False)
    print(f"  saved -> {config.EVENT_BACKTEST_FILE}")
    return out
