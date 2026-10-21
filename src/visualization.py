"""Static (matplotlib) EDA figures. Generated for the default product only."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from . import config


def _save(fig, name: str) -> None:
    out = config.FIGURES_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure -> {out}")


def _filter_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    if "product_name" in df.columns and product:
        return df[df["product_name"] == product].copy()
    return df


def plot_top_suppliers(df, product, n=10):
    df = _filter_product(df, product)
    if df.empty or "import_value_usd" not in df.columns: return
    tot = df.groupby("supplier_country")["import_value_usd"].sum().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(8, 5))
    tot.plot(kind="barh", ax=ax, color="#00A6C8")
    ax.invert_yaxis()
    ax.set_title(f"Top {n} Saudi {product} suppliers — total import value (USD)")
    ax.set_xlabel("Import value (USD)")
    _save(fig, f"top_suppliers_{product.lower()}.png")


def plot_concentration(df, product):
    df = _filter_product(df, product)
    if df.empty or not {"HHI", "top_3_supplier_share"}.issubset(df.columns): return
    g = df.groupby("date")[["HHI", "top_3_supplier_share"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(g["date"], g["HHI"], label="HHI", color="#5E60CE")
    ax.plot(g["date"], g["top_3_supplier_share"], label="Top-3 share", color="#00A6C8")
    ax.legend()
    ax.set_title(f"Saudi {product} supplier concentration")
    _save(fig, f"concentration_{product.lower()}.png")


def plot_risk_trend(df, product, events):
    df = _filter_product(df, product)
    if df.empty or "supplyguard_risk_score_0_100" not in df.columns: return
    g = df.groupby("date")["supplyguard_risk_score_0_100"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(g["date"], g["supplyguard_risk_score_0_100"], color="#00A6C8", linewidth=2)
    for _, ev in events.dropna(subset=["date"]).iterrows():
        ax.axvline(ev["date"], color="#C03A2B", alpha=0.3, linestyle="--")
        ax.text(ev["date"], ax.get_ylim()[1]*0.95, str(ev["event_name"])[:25],
                rotation=90, fontsize=7, va="top")
    ax.set_title(f"Mean SupplyGuard risk score — {product}")
    _save(fig, f"risk_trend_{product.lower()}.png")


def plot_shipping_stress(df):
    cols = [c for c in [
        "bab_el_mandeb_transit_calls_stress_score_0_100",
        "hormuz_transit_calls_stress_score_0_100",
        "jeddah_incoming_shipment_stress_score_0_100",
        "king_abdullah_incoming_shipment_stress_score_0_100",
        "dammam_incoming_shipment_stress_score_0_100",
    ] if c in df.columns]
    if not cols: return
    g = df.groupby("date")[cols].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in cols:
        ax.plot(g["date"], g[c], label=c.replace("_stress_score_0_100", "").replace("_", " "))
    ax.legend(fontsize=7)
    ax.set_title("PortWatch monthly stress scores (0..100)")
    _save(fig, "shipping_stress.png")


def generate_all_figures(df, events, products=None):
    print("[visualization] generating per-product EDA figures")
    if products is None:
        products = sorted(df["product_name"].dropna().unique().tolist()) if "product_name" in df.columns else [None]
    for p in products:
        plot_top_suppliers(df, p)
        plot_concentration(df, p)
        plot_risk_trend(df, p, events)
    plot_shipping_stress(df)
