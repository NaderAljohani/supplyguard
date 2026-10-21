"""SupplyGuard Saudi — Strategic Food Import Risk Early-Warning System.

Multi-product dashboard. Run from project root:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config

# ============================================================
# Page config + theme
# ============================================================
st.set_page_config(
    page_title="SupplyGuard Saudi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B1F33"
TEAL = "#00A6C8"
TEAL_DARK = "#007A99"
INK = "#1F2D3D"
MUTED = "#5C6B7A"
SURFACE = "#FFFFFF"
SOFT = "#F4F7FB"
BORDER = "#E1E8EF"

RISK_COLORS = {"Low": "#2E9E5B", "Medium": "#E5A23B", "High": "#E0762B", "Critical": "#C03A2B"}
RISK_BG = {"Low": "#E7F5EC", "Medium": "#FBF1DD", "High": "#FBE6D8", "Critical": "#F8D9D4"}

PLOTLY_TEMPLATE = "plotly_white"
PRIMARY_SEQUENCE = [TEAL, "#5E60CE", "#48BFE3", "#80FFDB", "#56CFE1", "#7400B8", "#9D4EDD"]

PRODUCT_ICON = {"Rice": "🌾", "Wheat": "🌾", "Barley": "🌿", "Maize": "🌽", "Sugar": "🍬"}

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {SOFT}; }}
      section[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, {NAVY} 0%, #16314A 100%);
      }}
      section[data-testid="stSidebar"] * {{ color: #E6EDF3 !important; }}
      section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
          background-color: rgba(255,255,255,0.10) !important;
          border-color: rgba(255,255,255,0.30) !important;
          color: #FFFFFF !important;
      }}
      section[data-testid="stSidebar"] label {{ font-weight: 600; opacity: 0.9; }}
      .sg-title {{ font-size: 28px; font-weight: 700; color: {NAVY}; margin-bottom: 2px; line-height: 1.2; }}
      .sg-subtitle {{ font-size: 14px; color: {MUTED}; margin-bottom: 14px; }}
      hr.sg-rule {{ border: 0; border-top: 1px solid {BORDER}; margin: 4px 0 18px 0; }}

      .sg-kpi {{
          background: {SURFACE}; border: 1px solid {BORDER}; border-left: 4px solid {TEAL};
          border-radius: 10px; padding: 14px 16px; height: 100%;
          box-shadow: 0 1px 2px rgba(11,31,51,0.04);
      }}
      .sg-kpi .lbl {{ font-size: 11.5px; color: {MUTED}; text-transform: uppercase; letter-spacing: .04em; }}
      .sg-kpi .val {{ font-size: 22px; font-weight: 700; color: {NAVY}; margin: 4px 0 2px 0; }}
      .sg-kpi .sub {{ font-size: 12px; color: {MUTED}; }}

      .sg-badge {{
          display: inline-block; padding: 4px 12px; border-radius: 999px;
          font-size: 13px; font-weight: 600; letter-spacing: .02em;
      }}

      .sg-card {{
          background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
          padding: 18px 20px; margin-bottom: 12px;
          box-shadow: 0 1px 2px rgba(11,31,51,0.04);
      }}
      .sg-callout {{
          background: #EAF6FA; border-left: 4px solid {TEAL_DARK};
          padding: 12px 16px; border-radius: 8px; color: {INK};
          font-size: 14px; margin: 10px 0 4px 0;
      }}
      .sg-warn {{
          background: #FDF4E2; border-left: 4px solid #E5A23B;
          padding: 12px 16px; border-radius: 8px; color: {INK};
          font-size: 14px; margin: 10px 0 4px 0;
      }}
      .sg-foot {{
          color: {MUTED}; font-size: 11.5px; text-align: center;
          padding: 18px 0 4px 0; border-top: 1px solid {BORDER}; margin-top: 30px;
      }}
      [data-testid="stDataFrame"] {{ border-radius: 10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Loaders (cached)
# ============================================================

@st.cache_data(show_spinner=False)
def load_modeling() -> pd.DataFrame:
    if not config.MODELING_FILE.exists(): return pd.DataFrame()
    return pd.read_csv(config.MODELING_FILE, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    if not config.EVENTS_FILE.exists(): return pd.DataFrame()
    return pd.read_csv(config.EVENTS_FILE, parse_dates=["date"]).dropna(how="all")


@st.cache_data(show_spinner=False)
def load_portwatch_long() -> pd.DataFrame:
    if not config.PORTWATCH_LONG_FILE.exists(): return pd.DataFrame()
    return pd.read_csv(config.PORTWATCH_LONG_FILE, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_portwatch_wide() -> pd.DataFrame:
    if not config.PORTWATCH_WIDE_FILE.exists(): return pd.DataFrame()
    return pd.read_csv(config.PORTWATCH_WIDE_FILE, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_methodology() -> pd.DataFrame:
    return load_csv(config.METHODOLOGY_FILE).dropna(how="all") if config.METHODOLOGY_FILE.exists() else pd.DataFrame()


# ============================================================
# Helpers
# ============================================================

def fmt_money(x) -> str:
    if x is None or pd.isna(x): return "—"
    x = float(x)
    if abs(x) >= 1e9: return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6: return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def fmt_pct(x, dec: int = 1) -> str:
    if x is None or pd.isna(x): return "—"
    return f"{float(x)*100:.{dec}f}%"


def risk_level(score: float) -> str:
    if pd.isna(score): return "Low"
    for lo, hi, name in config.RISK_LEVELS:
        if lo <= score < hi:
            return name
    return "Critical"


def risk_badge_html(level: str) -> str:
    c = RISK_COLORS.get(level, MUTED)
    bg = RISK_BG.get(level, "#EEE")
    return f'<span class="sg-badge" style="color:{c};background:{bg};">{level}</span>'


def kpi_html(label: str, value: str, sub: str = "", border_color: str = TEAL) -> str:
    return (
        f'<div class="sg-kpi" style="border-left-color:{border_color};">'
        f'<div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def callout(text: str, kind: str = "info") -> None:
    css = "sg-callout" if kind == "info" else "sg-warn"
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="sg-title">{title}</div>'
        f'<div class="sg-subtitle">{subtitle}</div>'
        f'<hr class="sg-rule" />',
        unsafe_allow_html=True,
    )


def materiality_mask(df: pd.DataFrame) -> pd.Series:
    share = pd.to_numeric(df.get("supplier_share_value", 0), errors="coerce").fillna(0)
    value = pd.to_numeric(df.get("import_value_usd", 0), errors="coerce").fillna(0)
    return (share >= config.MATERIAL_SHARE_THRESHOLD) | (value >= config.MATERIAL_VALUE_THRESHOLD_USD)


def gauge(value: float, title: str, height: int = 220) -> go.Figure:
    v = float(value or 0)
    lvl = risk_level(v)
    bar_color = RISK_COLORS[lvl]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        number={"font": {"size": 30, "color": NAVY}},
        title={"text": f"<b>{title}</b>", "font": {"size": 13, "color": MUTED}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": MUTED,
                     "tickfont": {"size": 10, "color": MUTED}},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "white", "borderwidth": 1, "bordercolor": BORDER,
            "steps": [
                {"range": [0, 25], "color": RISK_BG["Low"]},
                {"range": [25, 40], "color": RISK_BG["Medium"]},
                {"range": [40, 60], "color": RISK_BG["High"]},
                {"range": [60, 100], "color": RISK_BG["Critical"]},
            ],
        },
    ))
    fig.update_layout(height=height, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor=SURFACE, font={"family": "sans serif"})
    return fig


def style_fig(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=height,
        margin=dict(t=40 if title else 20, b=30, l=10, r=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": "sans serif", "color": INK, "size": 12},
        title={"text": f"<b>{title}</b>", "x": 0.01, "font": {"size": 14, "color": NAVY}} if title else None,
        colorway=PRIMARY_SEQUENCE,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    return fig


COMPONENT_COLS = {
    "supplier_dependency_score_0_100": "Supplier dependency",
    "concentration_score_0_100": "Market concentration",
    "import_anomaly_score_0_100": "Import drop / anomaly",
    "price_stress_score_0_100": "Commodity price stress",
    "logistics_risk_score_0_100": "Logistics (LPI)",
    "shipping_stress_score_0_100": "Shipping stress",
}
COMPONENT_WEIGHTS = {
    "supplier_dependency_score_0_100": config.RISK_WEIGHTS["supplier_dependency"],
    "concentration_score_0_100": config.RISK_WEIGHTS["market_concentration"],
    "import_anomaly_score_0_100": config.RISK_WEIGHTS["import_anomaly"],
    "price_stress_score_0_100": config.RISK_WEIGHTS["price_stress"],
    "logistics_risk_score_0_100": config.RISK_WEIGHTS["logistics_risk"],
    "shipping_stress_score_0_100": config.RISK_WEIGHTS["shipping_stress"],
}


def classify_event(ev: pd.Series) -> str:
    event_type = str(ev.get("event_type", "") or "")
    name = str(ev.get("event_name", "") or "")
    if "future" in name.lower() or pd.Timestamp(ev["date"]) > pd.Timestamp.today():
        return "Future / forward-looking annotation"
    cls = config.EVENT_TYPE_CLASS.get(event_type)
    if cls: return cls
    if "global" in event_type.lower() or str(ev.get("affected_country", "")).lower() == "global":
        return "Global shock"
    if "ship" in event_type.lower() or "chokepoint" in event_type.lower() or "route" in str(ev.get("affected_route", "")).lower():
        return "Maritime route event"
    return "Product-specific event"


def event_relevant_for_product(ev: pd.Series, product: str) -> bool:
    cls = classify_event(ev)
    if cls in ("Maritime route event", "Global shock", "Future / forward-looking annotation"):
        return True
    affected = str(ev.get("affected_product", "") or "")
    if affected.lower() == product.lower(): return True
    name = str(ev.get("event_name", "") or "").lower()
    if "russia" in name or "ukraine" in name or "black sea" in name:
        return product in ("Wheat", "Barley", "Maize")
    if "sugar" in name: return product == "Sugar"
    return False


DEFAULT_EVENT_FOR_PRODUCT = {
    "Rice":   "India non-basmati rice export ban",
    "Wheat":  "Russia-Ukraine war / Black Sea grain disruption",
    "Barley": "Russia-Ukraine war / Black Sea grain disruption",
    "Maize":  "Russia-Ukraine war / Black Sea grain disruption",
    "Sugar":  "Sugar price stress",
}


# ============================================================
# Load
# ============================================================
df_all = load_modeling()
events = load_events()
pw_long = load_portwatch_long()
pw_wide = load_portwatch_wide()
methodology = load_methodology()
fc_metrics = load_csv(config.OUTPUTS_DIR / "forecast_model_metrics.csv")
fc_predictions = load_csv(config.OUTPUTS_DIR / "next_month_risk_forecast.csv")
fc_actual = load_csv(config.OUTPUTS_DIR / "forecast_actual_vs_predicted.csv")
fc_importance = load_csv(config.OUTPUTS_DIR / "forecast_feature_importance.csv")
fc_leakage_audit = load_csv(config.OUTPUTS_DIR / "forecast_leakage_audit.csv")
dq_summary = load_csv(config.OUTPUTS_DIR / "data_quality_summary.csv")
miss_summary = load_csv(config.OUTPUTS_DIR / "missingness_by_product.csv")
sens_summary = load_csv(config.OUTPUTS_DIR / "sensitivity_analysis_summary.csv")
sens_detail = load_csv(config.OUTPUTS_DIR / "sensitivity_analysis_detail.csv")

if df_all.empty:
    st.error("Modeling dataset not found. Run the pipeline first:\n\n```\npython run_pipeline.py\n```")
    st.stop()

df_all["risk_level"] = df_all["supplyguard_risk_score_0_100"].apply(risk_level)

if "as_of_date" in fc_predictions.columns:
    fc_predictions["as_of_date"] = pd.to_datetime(fc_predictions["as_of_date"], errors="coerce")
    fc_predictions["forecast_for_date"] = pd.to_datetime(fc_predictions["forecast_for_date"], errors="coerce")
if "date" in fc_actual.columns:
    fc_actual["date"] = pd.to_datetime(fc_actual["date"], errors="coerce")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 6px 4px 14px 4px;">
            <div style="font-size: 22px; font-weight: 700;">🛡️ SupplyGuard Saudi</div>
            <div style="font-size: 12px; opacity: 0.85; margin-top: 2px;">
                Historical Food Import Risk · Research Prototype
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_products = sorted(df_all["product_name"].dropna().unique().tolist())
    default_product = config.DEFAULT_PRODUCT if config.DEFAULT_PRODUCT in available_products else available_products[0]
    product = st.selectbox(
        "Product (HS code)",
        available_products,
        format_func=lambda p: f"{PRODUCT_ICON.get(p,'•')}  {p}  —  HS {config.PRODUCT_HS.get(p,'—')}",
        index=available_products.index(default_product),
    )

    df = df_all[df_all["product_name"] == product].copy()
    n_suppliers = df["supplier_country"].nunique()
    latest_date_product = df["date"].max() if not df.empty else None

    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.07); border-radius: 8px; padding: 10px 12px; margin: 6px 0 14px 0; font-size: 12px;">
            <div style="opacity:.75;">Selected product</div>
            <div style="margin-top:4px;font-size:14px;"><b>{PRODUCT_ICON.get(product,'•')} {product}</b> · HS {config.PRODUCT_HS.get(product,'—')}</div>
            <div style="margin-top:8px;opacity:.75;">Data coverage</div>
            <div style="margin-top:2px;"><b>{df['date'].min().strftime('%b %Y') if not df.empty else '—'} → {df['date'].max().strftime('%b %Y') if not df.empty else '—'}</b></div>
            <div>Suppliers: <b>{n_suppliers}</b></div>
            <div>Latest month: <b>{latest_date_product.strftime('%b %Y') if latest_date_product is not None else '—'}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "📈  Executive Overview",
            "🏷️  Supplier Risk Analysis",
            "📊  Market Concentration",
            "🚢  Maritime Stress",
            "📜  Event Backtesting",
            "🚨  Anomaly Detection",
            "🔮  Forecasting",
            "🧪  Scenario Simulator",
            "📘  Methodology",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Hackathon build · 2026")

latest_date = df["date"].max() if not df.empty else None
latest_snap_all = df[df["date"] == latest_date].copy() if latest_date is not None else df.iloc[0:0].copy()
latest_snap_material = latest_snap_all[materiality_mask(latest_snap_all)].copy()

# Per-product forecast helpers
fc_for_product = fc_predictions[fc_predictions["product"] == product] if "product" in fc_predictions.columns else pd.DataFrame()
fc_metrics_product = fc_metrics[fc_metrics["product"] == product] if "product" in fc_metrics.columns else pd.DataFrame()


def product_weighted_forecast(prod: str) -> float | None:
    """Share-weighted next-month forecasted risk for the given product."""
    if fc_predictions.empty: return None
    sub = fc_predictions[fc_predictions["product"] == prod].copy()
    if sub.empty: return None
    sub["w"] = sub["supplier_share_value"].fillna(0)
    if sub["w"].sum() <= 0: return float(sub["forecast_next_month_risk_score"].mean())
    return float((sub["forecast_next_month_risk_score"] * sub["w"]).sum() / sub["w"].sum())


# ============================================================
# 1. Executive Overview
# ============================================================
if "Executive Overview" in page:
    page_header(
        f"Executive Overview — Saudi {product} Import Risk",
        "An explainable early-warning index combining supplier dependency, concentration, "
        "import anomalies, commodity-price stress, logistics performance, and maritime stress.",
    )

    if not latest_snap_material.empty:
        w = latest_snap_material["supplier_share_value"].fillna(0)
        saudi_score = float((latest_snap_material["supplyguard_risk_score_0_100"] * w).sum() / max(w.sum(), 1e-9))
    elif not latest_snap_all.empty:
        saudi_score = float(latest_snap_all["supplyguard_risk_score_0_100"].mean())
    else:
        saudi_score = 0.0
    saudi_level = risk_level(saudi_score)
    forecast_score = product_weighted_forecast(product)
    forecast_level = risk_level(forecast_score) if forecast_score is not None else None
    forecast_delta = (forecast_score - saudi_score) if forecast_score is not None else None

    top_supplier_row = latest_snap_all.sort_values("supplier_share_value", ascending=False).head(1)
    top_supplier_name = top_supplier_row["supplier_country"].iloc[0] if not top_supplier_row.empty else "—"
    top_supplier_share = top_supplier_row["supplier_share_value"].iloc[0] if not top_supplier_row.empty else np.nan
    top3 = latest_snap_all.nlargest(3, "supplier_share_value")["supplier_share_value"].sum() if not latest_snap_all.empty else np.nan
    hhi = latest_snap_all["HHI"].max() if "HHI" in latest_snap_all.columns and not latest_snap_all.empty else np.nan

    # Row 1 KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: st.markdown(kpi_html(f"{product} risk (now)", f"{saudi_score:.1f}", "share-weighted", RISK_COLORS[saudi_level]), unsafe_allow_html=True)
    with k2: st.markdown(
        f'<div class="sg-kpi" style="border-left-color:{RISK_COLORS[saudi_level]};">'
        f'<div class="lbl">Risk level</div><div class="val">{risk_badge_html(saudi_level)}</div>'
        f'<div class="sub">empirical bands</div></div>', unsafe_allow_html=True)
    with k3:
        if forecast_score is not None:
            sign = "+" if forecast_delta >= 0 else ""
            st.markdown(kpi_html("Next-month forecast", f"{forecast_score:.1f}",
                                 f"{sign}{forecast_delta:.1f} vs now · {forecast_level}",
                                 RISK_COLORS[forecast_level]), unsafe_allow_html=True)
        else:
            st.markdown(kpi_html("Next-month forecast", "—", "no forecast available"), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html("Top supplier", f"{top_supplier_name}", fmt_pct(top_supplier_share)), unsafe_allow_html=True)
    with k5: st.markdown(kpi_html("Top-3 share", fmt_pct(top3), "concentration"), unsafe_allow_html=True)
    with k6: st.markdown(kpi_html("HHI", f"{(hhi or 0):.3f}", "Herfindahl-Hirschman"), unsafe_allow_html=True)

    st.markdown("###")

    c1, c2 = st.columns([1.05, 1.95])
    with c1:
        st.plotly_chart(gauge(saudi_score, f"Saudi-wide {product} risk score", height=280), width="stretch")
    with c2:
        st.markdown(f"##### Top material high-risk {product.lower()} suppliers — {latest_date.strftime('%b %Y') if latest_date is not None else ''}")
        top_mat = latest_snap_material.sort_values("supplyguard_risk_score_0_100", ascending=False).head(8)
        if top_mat.empty:
            st.info("No material suppliers in the latest month.")
        else:
            view = pd.DataFrame({
                "Supplier": top_mat["supplier_country"],
                "Share": top_mat["supplier_share_value"].apply(fmt_pct),
                "Import Value": top_mat["import_value_usd"].apply(fmt_money),
                "Risk Score": top_mat["supplyguard_risk_score_0_100"].round(1),
                "Level": top_mat["risk_level"],
                "Top Driver": top_mat["top_risk_driver"],
            })
            st.dataframe(view, width="stretch", hide_index=True)
        callout(
            "Materiality filter: suppliers contributing ≥1% of monthly imports OR ≥USD 1M. "
            "This removes statistical anomalies from tiny suppliers that aren't operationally relevant."
        )

    # ----- Business value card -----
    total_exposed_value = float(latest_snap_material["import_value_usd"].sum()) if not latest_snap_material.empty else 0.0
    top_supplier_value = float(top_supplier_row["import_value_usd"].iloc[0]) if not top_supplier_row.empty else 0.0
    diversify_15pct_top = top_supplier_value * 0.15
    st.markdown(
        f"""
        <div class="sg-card" style="border-left:4px solid {TEAL_DARK};">
            <div style="font-size:14px;color:{MUTED};text-transform:uppercase;letter-spacing:.04em;">Business value · {product}</div>
            <div style="font-size:15px;color:{INK};margin-top:6px;line-height:1.6;">
                <b>Current exposed monthly import value (material suppliers):</b> {fmt_money(total_exposed_value)}.<br>
                <b>Top supplier value (potential diversification target):</b> {fmt_money(top_supplier_value)} from <b>{top_supplier_name}</b>.<br>
                <b>15% reallocation indicator:</b> {fmt_money(diversify_15pct_top)} of monthly flow would shift to alternative suppliers under a 15% reassignment scenario.
            </div>
            <div style="font-size:12.5px;color:{MUTED};margin-top:8px;">
                Operational impact: reducing reliance on the dominant supplier lowers the dependency component
                and improves resilience against single-country shocks. See the Scenario Simulator to size the effect.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Risk trend -----
    st.markdown(f"##### Saudi-wide {product} risk trend (share-weighted across material suppliers)")
    monthly = []
    for d, sub in df.groupby("date"):
        msub = sub[materiality_mask(sub)]
        if msub.empty: continue
        w = msub["supplier_share_value"].fillna(0)
        s = float((msub["supplyguard_risk_score_0_100"] * w).sum() / max(w.sum(), 1e-9))
        monthly.append({"date": d, "risk": s})
    trend = pd.DataFrame(monthly)
    if not trend.empty:
        fig = px.line(trend, x="date", y="risk", labels={"date": "Month", "risk": "Risk Score"})
        fig.update_traces(line=dict(color=TEAL, width=2.5))
        for _, ev in events.dropna(subset=["date"]).iterrows():
            if event_relevant_for_product(ev, product):
                fig.add_vline(x=ev["date"], line_dash="dash", line_color="#C03A2B", opacity=0.35)
        style_fig(fig, height=340)
        st.plotly_chart(fig, width="stretch")

    # ============================================================
    # Saudi Food Import Portfolio section
    # ============================================================
    st.markdown("###")
    st.markdown(
        f'<div class="sg-title" style="font-size:22px;">Saudi Food Import Portfolio</div>'
        f'<div class="sg-subtitle">Cross-product comparison — the engine is HS-code agnostic and works on Rice, Wheat, Barley, Maize, Sugar.</div>'
        f'<hr class="sg-rule" />',
        unsafe_allow_html=True,
    )

    portfolio_rows = []
    for prod in available_products:
        psub = df_all[df_all["product_name"] == prod]
        latest_d = psub["date"].max()
        latest_p = psub[psub["date"] == latest_d]
        mat_p = latest_p[materiality_mask(latest_p)]
        if mat_p.empty:
            mat_p = latest_p
        top_s = latest_p.sort_values("supplier_share_value", ascending=False).head(1)
        top_3_share_p = latest_p.nlargest(3, "supplier_share_value")["supplier_share_value"].sum()
        hhi_p = float(latest_p["HHI"].max()) if "HHI" in latest_p.columns else np.nan
        w_p = mat_p["supplier_share_value"].fillna(0)
        risk_p = float((mat_p["supplyguard_risk_score_0_100"] * w_p).sum() / max(w_p.sum(), 1e-9))
        fc_p = product_weighted_forecast(prod)
        portfolio_rows.append({
            "Product": prod,
            "HS code": config.PRODUCT_HS.get(prod, "—"),
            "Latest month": latest_d.strftime("%b %Y") if pd.notna(latest_d) else "—",
            "Top supplier": top_s["supplier_country"].iloc[0] if not top_s.empty else "—",
            "Top supplier share": fmt_pct(top_s["supplier_share_value"].iloc[0] if not top_s.empty else np.nan),
            "Top-3 share": fmt_pct(top_3_share_p),
            "HHI": f"{hhi_p:.3f}" if pd.notna(hhi_p) else "—",
            "Risk (now)": round(risk_p, 1),
            "Forecast next month": round(fc_p, 1) if fc_p is not None else "—",
            "Level": risk_level(risk_p),
            "Main driver": (mat_p.sort_values("supplyguard_risk_score_0_100", ascending=False)
                            ["top_risk_driver"].iloc[0] if not mat_p.empty and "top_risk_driver" in mat_p.columns else "—"),
        })
    portfolio = pd.DataFrame(portfolio_rows)
    st.dataframe(portfolio, width="stretch", hide_index=True)

    pc1, pc2 = st.columns(2)
    with pc1:
        rank_df = portfolio.sort_values("Risk (now)", ascending=True)
        fig = go.Figure(go.Bar(
            x=rank_df["Risk (now)"], y=rank_df["Product"], orientation="h",
            marker_color=[RISK_COLORS[l] for l in rank_df["Level"]],
            text=[f"{v:.1f}" for v in rank_df["Risk (now)"]], textposition="outside",
        ))
        fig.update_layout(xaxis_title="Risk Score (now)")
        st.plotly_chart(style_fig(fig, height=320, title="Product risk ranking — now"), width="stretch")
    with pc2:
        if portfolio["Forecast next month"].apply(lambda v: isinstance(v, (int, float))).any():
            fc_df = portfolio[portfolio["Forecast next month"] != "—"].copy()
            fc_df["Forecast next month"] = pd.to_numeric(fc_df["Forecast next month"], errors="coerce")
            fc_df = fc_df.sort_values("Forecast next month", ascending=True)
            fc_df["Lvl"] = fc_df["Forecast next month"].apply(risk_level)
            fig = go.Figure(go.Bar(
                x=fc_df["Forecast next month"], y=fc_df["Product"], orientation="h",
                marker_color=[RISK_COLORS[l] for l in fc_df["Lvl"]],
                text=[f"{v:.1f}" for v in fc_df["Forecast next month"]], textposition="outside",
            ))
            fig.update_layout(xaxis_title="Forecast Risk Score")
            st.plotly_chart(style_fig(fig, height=320, title="Product risk ranking — next-month forecast"), width="stretch")

    with st.expander("About SupplyGuard Saudi"):
        st.markdown(
            "**SupplyGuard Saudi** helps logistics and procurement decision-makers monitor "
            "Saudi food import risk by combining trade dependency, market concentration, "
            "commodity price stress, supplier logistics performance, and maritime chokepoint stress "
            "into a single explainable **0–100 risk score** at the **product × supplier × month** level. "
            "Covers Rice, Wheat, Barley, Maize, and Sugar."
        )

    st.markdown('<div class="sg-foot">Hackathon prototype — built for explainable food import risk monitoring.</div>', unsafe_allow_html=True)

# ============================================================
# 2. Supplier Risk Analysis
# ============================================================
elif "Supplier Risk Analysis" in page:
    page_header(
        f"Supplier Risk Analysis — {product}",
        "Drill into any supplier country: risk trend, import flow, supplier share, "
        "and a per-month explanation card.",
    )

    suppliers = sorted(df["supplier_country"].dropna().unique().tolist())
    if not suppliers:
        st.warning(f"No suppliers found for {product}.")
        st.stop()
    default = "India" if "India" in suppliers else suppliers[0]
    chosen = st.selectbox("Supplier country", suppliers, index=suppliers.index(default))
    sub = df[df["supplier_country"] == chosen].sort_values("date").copy()
    if sub.empty:
        st.warning("No data for selected supplier.")
        st.stop()
    latest = sub.iloc[-1]
    level = latest["risk_level"]

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(kpi_html("Latest risk score", f"{latest['supplyguard_risk_score_0_100']:.1f}", "0–100 scale", RISK_COLORS[level]), unsafe_allow_html=True)
    with k2: st.markdown(
        f'<div class="sg-kpi" style="border-left-color:{RISK_COLORS[level]};">'
        f'<div class="lbl">Risk level</div><div class="val">{risk_badge_html(level)}</div>'
        f'<div class="sub">empirical bands</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(kpi_html("Latest share", fmt_pct(latest["supplier_share_value"]), f"of Saudi {product.lower()}"), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html("Latest import value", fmt_money(latest["import_value_usd"]), pd.Timestamp(latest["date"]).strftime("%b %Y")), unsafe_allow_html=True)
    with k5: st.markdown(kpi_html("Top driver", str(latest.get("top_risk_driver", "—")), "highest contribution"), unsafe_allow_html=True)

    st.markdown("###")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(sub, x="date", y="supplyguard_risk_score_0_100",
                      labels={"date": "Month", "supplyguard_risk_score_0_100": "Risk Score"})
        fig.update_traces(line=dict(color=TEAL, width=2.5))
        for _, ev in events.dropna(subset=["date"]).iterrows():
            if event_relevant_for_product(ev, product):
                fig.add_vline(x=ev["date"], line_dash="dash", line_color="#C03A2B", opacity=0.3)
        st.plotly_chart(style_fig(fig, height=320, title="Risk Score over time"), width="stretch")
    with c2:
        fig = px.area(sub, x="date", y="import_value_usd",
                      labels={"date": "Month", "import_value_usd": "Import Value (USD)"})
        fig.update_traces(line=dict(color=TEAL_DARK, width=1.5), fillcolor="rgba(0,166,200,0.18)")
        st.plotly_chart(style_fig(fig, height=320, title="Import Value (USD) over time"), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        fig = px.area(sub, x="date", y="supplier_share_value",
                      labels={"date": "Month", "supplier_share_value": "Supplier Share"})
        fig.update_traces(line=dict(color="#5E60CE", width=1.5), fillcolor="rgba(94,96,206,0.18)")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, height=320, title="Supplier share over time"), width="stretch")
    with c4:
        comp_vals = {COMPONENT_COLS[c]: float(latest[c]) * COMPONENT_WEIGHTS[c] / sum(COMPONENT_WEIGHTS.values())
                     for c in COMPONENT_COLS if c in sub.columns}
        comp_df = (pd.DataFrame({"Component": comp_vals.keys(), "Points": comp_vals.values()})
                   .sort_values("Points", ascending=True))
        fig = px.bar(comp_df, x="Points", y="Component", orientation="h",
                     color="Points", color_continuous_scale=[(0, "#E7F5EC"), (1, TEAL_DARK)],
                     labels={"Points": "Points contributed to risk score"})
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, height=320, title="Component contribution (latest month)"), width="stretch")

    st.markdown("###")
    st.markdown(
        f"""
        <div class="sg-card">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;flex-wrap:wrap;">
                <div style="font-size:18px;font-weight:700;color:{NAVY};">Explanation card · {product} · {chosen}</div>
                {risk_badge_html(level)}
                <div style="color:{MUTED};font-size:13px;">Month: <b>{pd.Timestamp(latest['date']).strftime('%b %Y')}</b></div>
                <div style="color:{MUTED};font-size:13px;">Risk: <b>{latest['supplyguard_risk_score_0_100']:.1f} / 100</b></div>
            </div>
            <div style="color:{INK};font-size:14px;line-height:1.55;">
                <b>Top drivers:</b> {latest.get('top_risk_driver','—')} → {latest.get('second_risk_driver','—')}<br>
                {latest.get('risk_explanation_text','')}<br>
                <b>Recommended action.</b> Monitor route stress and supplier-share evolution; if dependency exceeds 50%, consider partial diversification toward alternative suppliers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 3. Market Concentration
# ============================================================
elif "Market Concentration" in page:
    page_header(
        f"Market Concentration — Saudi {product} Imports",
        "How dependent are Saudi imports on a small number of supplier countries? "
        "High concentration means a single supplier shock can disrupt national supply.",
    )

    if latest_snap_all.empty:
        st.warning("No supplier data available.")
        st.stop()

    top10 = latest_snap_all.sort_values("supplier_share_value", ascending=False).head(10).copy()
    top_supplier = top10["supplier_country"].iloc[0]
    top10["color"] = top10["supplier_country"].apply(lambda c: TEAL if c == top_supplier else "#9AB2C5")
    fig = go.Figure(go.Bar(
        x=top10["supplier_share_value"], y=top10["supplier_country"],
        orientation="h", marker_color=top10["color"],
        text=[fmt_pct(v) for v in top10["supplier_share_value"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Share: %{x:.1%}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style_fig(fig, height=380,
                              title=f"Top {product} suppliers · {latest_date.strftime('%b %Y')}"),
                    width="stretch")
    callout(
        f"{top_supplier} is the dominant {product.lower()} supplier with **{fmt_pct(top10.iloc[0]['supplier_share_value'])}**. "
        f"Top-3 share is **{fmt_pct(top10.head(3)['supplier_share_value'].sum())}**."
    )

    c1, c2 = st.columns(2)
    with c1:
        top5 = df.groupby("supplier_country")["import_value_usd"].sum().nlargest(5).index.tolist()
        stack = df[df["supplier_country"].isin(top5)].copy()
        pivot = stack.pivot_table(index="date", columns="supplier_country",
                                  values="supplier_share_value", aggfunc="mean").fillna(0)
        fig = go.Figure()
        for i, c in enumerate(pivot.columns):
            color = TEAL if c == top_supplier else PRIMARY_SEQUENCE[(i+1) % len(PRIMARY_SEQUENCE)]
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[c], stackgroup="one",
                                     name=c, line=dict(width=0), fillcolor=color))
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, height=340, title="Supplier share over time — top 5"), width="stretch")
    with c2:
        if "HHI" in df.columns:
            g = df.groupby("date")["HHI"].mean().reset_index()
            fig = px.line(g, x="date", y="HHI", labels={"date": "Month", "HHI": "HHI (0–1)"})
            fig.update_traces(line=dict(color="#5E60CE", width=2.5))
            st.plotly_chart(style_fig(fig, height=340, title="HHI concentration index over time"), width="stretch")

    if "top_3_supplier_share" in df.columns:
        g = df.groupby("date")["top_3_supplier_share"].mean().reset_index()
        fig = px.line(g, x="date", y="top_3_supplier_share", labels={"top_3_supplier_share": "Top-3 supplier share"})
        fig.update_traces(line=dict(color=TEAL_DARK, width=2.5))
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, height=320, title="Top-3 supplier share over time"), width="stretch")
        callout("High supplier concentration means disruption in one supplier country can materially affect Saudi import resilience.")

# ============================================================
# 4. Maritime Stress
# ============================================================
elif "Maritime Stress" in page:
    page_header(
        "Maritime Stress (IMF PortWatch)",
        "Chokepoint and port stress signals. Asian suppliers (India, Pakistan, Thailand) typically route through "
        "Bab-el-Mandeb to Jeddah; the Gulf route via Hormuz / Dammam is the alternative.",
    )

    stress_cols = [c for c in [
        "bab_el_mandeb_transit_calls_stress_score_0_100",
        "hormuz_transit_calls_stress_score_0_100",
        "jeddah_incoming_shipment_stress_score_0_100",
        "king_abdullah_incoming_shipment_stress_score_0_100",
        "dammam_incoming_shipment_stress_score_0_100",
    ] if c in df_all.columns]
    pretty = {
        "bab_el_mandeb_transit_calls_stress_score_0_100": "Bab el-Mandeb",
        "hormuz_transit_calls_stress_score_0_100": "Strait of Hormuz",
        "jeddah_incoming_shipment_stress_score_0_100": "Jeddah",
        "king_abdullah_incoming_shipment_stress_score_0_100": "King Abdullah Port",
        "dammam_incoming_shipment_stress_score_0_100": "Dammam",
    }
    if stress_cols:
        g = df_all.groupby("date")[stress_cols].mean().reset_index()
        long = g.melt("date", var_name="metric", value_name="stress")
        long["metric"] = long["metric"].map(pretty)
        fig = px.line(long, x="date", y="stress", color="metric",
                      labels={"stress": "Stress (0–100)", "metric": ""})
        for _, ev in events.dropna(subset=["date"]).iterrows():
            fig.add_vline(x=ev["date"], line_dash="dash", line_color="#C03A2B", opacity=0.3)
        st.plotly_chart(style_fig(fig, height=380, title="Chokepoint & port stress over time"), width="stretch")
        callout(
            "Red Sea / Bab el-Mandeb stress increased sharply after late 2023, aligning with Houthi attacks and "
            "global rerouting around the Cape of Good Hope. Bab el-Mandeb mainly affects Asian-supplied rice and "
            "Black-Sea-routed grains; Hormuz / Dammam matter most for Gulf imports."
        )

    c1, c2 = st.columns(2)
    with c1:
        cols = [c for c in ["red_sea_shipping_stress_score_0_100", "gulf_shipping_stress_score_0_100"] if c in df_all.columns]
        if cols:
            g = df_all.groupby("date")[cols].mean().reset_index()
            long = g.melt("date", var_name="route", value_name="stress")
            long["route"] = long["route"].str.replace("_shipping_stress_score_0_100", "", regex=False).str.replace("_", " ").str.title()
            fig = px.area(long, x="date", y="stress", color="route",
                          labels={"stress": "Stress (0–100)", "route": ""},
                          color_discrete_map={"Red Sea": "#C03A2B", "Gulf": TEAL_DARK})
            st.plotly_chart(style_fig(fig, height=340, title="Red Sea vs Gulf shipping stress"), width="stretch")
    with c2:
        if not pw_long.empty:
            locs = sorted(pw_long["location"].dropna().unique())
            chosen_loc = st.selectbox("Location", locs, key="pw_loc")
            metrics = sorted(pw_long[pw_long["location"] == chosen_loc]["metric"].dropna().unique())
            chosen_metric = st.selectbox("Metric", metrics, key="pw_metric")
            psub = pw_long[(pw_long["location"] == chosen_loc) & (pw_long["metric"] == chosen_metric)]
            fig = px.line(psub.sort_values("date"), x="date", y="stress_score_0_100",
                          labels={"stress_score_0_100": "Stress (0–100)"})
            fig.update_traces(line=dict(color=TEAL, width=2.5))
            st.plotly_chart(style_fig(fig, height=340, title=f"{chosen_loc.replace('_',' ').title()} · {chosen_metric}"), width="stretch")

# ============================================================
# 5. Event Backtesting — with availability fallback
# ============================================================
elif "Event Backtesting" in page:
    page_header(
        f"Event Backtesting — {product}",
        "Did the risk score rise around known historical disruptions? "
        "This is directional validation and monitoring, not causal proof.",
    )

    if events.empty:
        st.warning("No events file loaded.")
        st.stop()

    ev_df_full = events.dropna(subset=["date"]).copy()
    ev_df_full["event_class"] = ev_df_full.apply(classify_event, axis=1)
    show_all = st.checkbox("Show all events (not only events relevant to selected product)", value=False)
    ev_df = ev_df_full if show_all else ev_df_full[ev_df_full.apply(lambda r: event_relevant_for_product(r, product), axis=1)]
    if ev_df.empty:
        st.info(f"No events relevant to {product}. Toggle 'Show all events' to see the full list.")
        st.stop()

    ev_df = ev_df.reset_index(drop=True)
    ev_df["label"] = ev_df.apply(lambda r: f"{r['event_name']}  ·  {pd.Timestamp(r['date']).strftime('%Y-%m-%d')}", axis=1)

    # Default event by product (fuzzy match on event name)
    default_event_name = DEFAULT_EVENT_FOR_PRODUCT.get(product, "")
    default_idx = 0
    for i, r in ev_df.iterrows():
        if default_event_name and default_event_name.lower() in str(r["event_name"]).lower():
            default_idx = int(i)
            break

    ev_idx = st.selectbox("Event", range(len(ev_df)),
                          format_func=lambda i: ev_df.iloc[i]["label"],
                          index=default_idx)
    ev = ev_df.iloc[ev_idx]

    product_max = df["date"].max() if not df.empty else pd.NaT
    product_min = df["date"].min() if not df.empty else pd.NaT
    pw_max = pw_wide["date"].max() if not pw_wide.empty else pd.NaT
    pw_min = pw_wide["date"].min() if not pw_wide.empty else pd.NaT
    in_product = bool(pd.notna(product_min) and pd.notna(product_max) and product_min <= ev["date"] <= product_max + pd.Timedelta(days=30))
    in_pw = bool(pd.notna(pw_min) and pd.notna(pw_max) and pw_min <= ev["date"] <= pw_max + pd.Timedelta(days=30))
    ev_class = ev["event_class"]

    st.markdown(
        f"""
        <div class="sg-card">
            <div style="font-size:16px;font-weight:700;color:{NAVY};">{ev['event_name']}</div>
            <div style="color:{MUTED};font-size:13px;margin-bottom:6px;">
                {pd.Timestamp(ev['date']).strftime('%d %B %Y')} · <b>{ev_class}</b> · {ev.get('event_type','—')} · Affected: {ev.get('affected_country','—')} / {ev.get('affected_product','—')}
            </div>
            <div style="color:{INK};font-size:14px;">{ev.get('notes','')}</div>
            <div style="margin-top:8px;color:{MUTED};font-size:12px;">
                Data availability — {product} risk: <b>{product_min.strftime('%Y-%m') if pd.notna(product_min) else '—'} → {product_max.strftime('%Y-%m') if pd.notna(product_max) else '—'}</b>
                ({'<span style="color:'+RISK_COLORS["Low"]+';">in range</span>' if in_product else '<span style="color:'+RISK_COLORS["High"]+';">out of range</span>'})
                · PortWatch: <b>{pw_min.strftime('%Y-%m') if pd.notna(pw_min) else '—'} → {pw_max.strftime('%Y-%m') if pd.notna(pw_max) else '—'}</b>
                ({'<span style="color:'+RISK_COLORS["Low"]+';">in range</span>' if in_pw else '<span style="color:'+RISK_COLORS["High"]+';">out of range</span>'})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if in_product:
        country = ev.get("affected_country")
        if isinstance(country, str) and country.lower() not in ("global", "shipping route"):
            sub = df[df["supplier_country"].str.lower() == country.lower()].copy()
            label = f"{product} from {country} — supplier-level risk"
        else:
            sub = df.groupby("date").mean(numeric_only=True).reset_index()
            label = f"Saudi {product} — mean risk across suppliers"
        sub["days_to_event"] = (sub["date"] - ev["date"]).dt.days
        sub = sub[(sub["days_to_event"] >= -180) & (sub["days_to_event"] <= 365)]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(sub, x="date", y="supplyguard_risk_score_0_100",
                          labels={"date": "Month", "supplyguard_risk_score_0_100": "Risk Score"})
            fig.update_traces(line=dict(color=TEAL, width=2.5))
            fig.add_vline(x=ev["date"], line_dash="dash", line_color="#C03A2B")
            st.plotly_chart(style_fig(fig, height=340, title=label), width="stretch")
        with c2:
            comp_cols = [c for c in COMPONENT_COLS if c in sub.columns]
            long = sub[["date"] + comp_cols].melt("date", var_name="component", value_name="value")
            long["component"] = long["component"].map(COMPONENT_COLS)
            fig = px.line(long, x="date", y="value", color="component",
                          labels={"value": "Component score (0–100)", "component": ""})
            fig.add_vline(x=ev["date"], line_dash="dash", line_color="#C03A2B")
            st.plotly_chart(style_fig(fig, height=340, title="Components around event"), width="stretch")

        pre = sub[sub["days_to_event"] < 0]["supplyguard_risk_score_0_100"]
        during = sub[(sub["days_to_event"] >= 0) & (sub["days_to_event"] <= 60)]["supplyguard_risk_score_0_100"]
        post = sub[sub["days_to_event"] > 60]["supplyguard_risk_score_0_100"]

        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(kpi_html("Pre-event (mean)", f"{pre.mean():.1f}" if not pre.empty else "—", f"{len(pre)} months"), unsafe_allow_html=True)
        with k2: st.markdown(kpi_html("During 0–60d", f"{during.mean():.1f}" if not during.empty else "—", f"{len(during)} months"), unsafe_allow_html=True)
        with k3: st.markdown(kpi_html("Post >60d", f"{post.mean():.1f}" if not post.empty else "—", f"{len(post)} months"), unsafe_allow_html=True)
        with k4:
            if not pre.empty and not during.empty:
                delta = during.mean() - pre.mean()
                color = RISK_COLORS["Critical"] if delta > 0 else RISK_COLORS["Low"]
                st.markdown(kpi_html("Δ during − pre", f"{delta:+.2f}", "risk-score points", color), unsafe_allow_html=True)
            else:
                st.markdown(kpi_html("Δ during − pre", "—", "insufficient data"), unsafe_allow_html=True)
        callout("Directional validation and monitoring — the index moved alongside the event. This is not causal proof.")

    elif in_pw:
        callout(
            f"<b>Product-level import risk data ends at {product_max.strftime('%Y-%m')}; "
            f"this event occurs after that date. Product-risk backtesting is not available. "
            f"Showing PortWatch maritime stress monitoring instead.</b>",
            kind="warn",
        )

        name_lower = str(ev.get("event_name", "")).lower()
        route_lower = str(ev.get("affected_route", "")).lower()
        if "hormuz" in name_lower or "hormuz" in route_lower:
            pw_cols = [c for c in [
                "hormuz_transit_calls_stress_score_0_100",
                "hormuz_trade_volume_stress_score_0_100",
                "dammam_incoming_shipment_stress_score_0_100",
            ] if c in pw_wide.columns]
            pw_pretty = {
                "hormuz_transit_calls_stress_score_0_100": "Hormuz transit calls",
                "hormuz_trade_volume_stress_score_0_100": "Hormuz trade volume",
                "dammam_incoming_shipment_stress_score_0_100": "Dammam incoming (Gulf comparison)",
            }
        elif "red sea" in name_lower or "bab" in name_lower:
            pw_cols = [c for c in [
                "bab_el_mandeb_transit_calls_stress_score_0_100",
                "bab_el_mandeb_trade_volume_stress_score_0_100",
                "jeddah_incoming_shipment_stress_score_0_100",
            ] if c in pw_wide.columns]
            pw_pretty = {
                "bab_el_mandeb_transit_calls_stress_score_0_100": "Bab el-Mandeb transit calls",
                "bab_el_mandeb_trade_volume_stress_score_0_100": "Bab el-Mandeb trade volume",
                "jeddah_incoming_shipment_stress_score_0_100": "Jeddah incoming",
            }
        else:
            pw_cols = [c for c in [
                "bab_el_mandeb_transit_calls_stress_score_0_100",
                "hormuz_transit_calls_stress_score_0_100",
                "jeddah_incoming_shipment_stress_score_0_100",
                "dammam_incoming_shipment_stress_score_0_100",
            ] if c in pw_wide.columns]
            pw_pretty = {c: c.replace("_stress_score_0_100", "").replace("_", " ").title() for c in pw_cols}

        pw_sub = pw_wide[(pw_wide["date"] >= ev["date"] - pd.Timedelta(days=180)) &
                        (pw_wide["date"] <= ev["date"] + pd.Timedelta(days=365))].copy()
        pw_sub["days_to_event"] = (pw_sub["date"] - ev["date"]).dt.days

        if pw_cols and not pw_sub.empty:
            g = pw_sub[["date"] + pw_cols].melt("date", var_name="series", value_name="stress")
            g["series"] = g["series"].map(pw_pretty)
            fig = px.line(g, x="date", y="stress", color="series",
                          labels={"stress": "Stress (0–100)", "series": ""})
            fig.add_vline(x=ev["date"], line_dash="dash", line_color="#C03A2B")
            st.plotly_chart(style_fig(fig, height=380, title="PortWatch maritime stress around event"), width="stretch")

            headline = pw_cols[0]
            pre = pw_sub[pw_sub["days_to_event"] < 0][headline]
            during = pw_sub[(pw_sub["days_to_event"] >= 0) & (pw_sub["days_to_event"] <= 60)][headline]
            post = pw_sub[pw_sub["days_to_event"] > 60][headline]
            k1, k2, k3, k4 = st.columns(4)
            with k1: st.markdown(kpi_html(f"{pw_pretty[headline]} · pre", f"{pre.mean():.1f}" if not pre.empty else "—", f"{len(pre)} months"), unsafe_allow_html=True)
            with k2: st.markdown(kpi_html(f"{pw_pretty[headline]} · during 0–60d", f"{during.mean():.1f}" if not during.empty else "—", f"{len(during)} months"), unsafe_allow_html=True)
            with k3: st.markdown(kpi_html(f"{pw_pretty[headline]} · post >60d", f"{post.mean():.1f}" if not post.empty else "—", f"{len(post)} months"), unsafe_allow_html=True)
            with k4:
                if not pre.empty and not during.empty:
                    delta = during.mean() - pre.mean()
                    color = RISK_COLORS["Critical"] if delta > 0 else RISK_COLORS["Low"]
                    st.markdown(kpi_html("Δ during − pre", f"{delta:+.2f}", "stress points", color), unsafe_allow_html=True)
                else:
                    st.markdown(kpi_html("Δ during − pre", "—", "insufficient data"), unsafe_allow_html=True)

        callout("Maritime signal monitoring, not product-risk validation.", kind="warn")
    else:
        st.markdown(
            f'<div class="sg-warn">'
            f'<b>Insufficient data to backtest this event.</b> The event date is outside both the '
            f'{product} product-risk data range and the PortWatch data range.</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# 6. Anomaly Detection
# ============================================================
elif "Anomaly Detection" in page:
    page_header(
        f"Anomaly Detection — {product}",
        "Z-score on rolling residuals + Isolation Forest on **driver-only** features (import value, "
        "MoM/YoY changes, supplier share, price stress, shipping stress, LPI). The composite "
        "risk score is deliberately excluded to avoid circular interpretation.",
    )

    tcol1, tcol2 = st.columns([2, 1])
    with tcol1:
        show_all = st.checkbox("Show all statistical anomalies (incl. tiny suppliers)", value=False)
    with tcol2:
        callout(
            "Pipeline default: <b>drivers-only</b> Isolation Forest. "
            "A diagnostic <i>with-composite</i> variant exists in code (<code>include_composite_risk=True</code>) "
            "but is not used in this dashboard because it would be circular.",
        )

    if "anomaly_score" not in df.columns:
        st.warning("Anomaly columns missing — re-run the pipeline.")
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c1:
        g = df.groupby("date")["anomaly_score"].mean().reset_index()
        fig = px.line(g, x="date", y="anomaly_score",
                      labels={"date": "Month", "anomaly_score": "Mean anomaly score"})
        fig.update_traces(line=dict(color=TEAL, width=2.5))
        for _, ev in events.dropna(subset=["date"]).iterrows():
            if event_relevant_for_product(ev, product):
                fig.add_vline(x=ev["date"], line_dash="dash", line_color="#C03A2B", opacity=0.3)
        st.plotly_chart(style_fig(fig, height=350, title=f"Mean anomaly score over time — {product}"), width="stretch")
    with c2:
        top = (df.groupby("date")["anomaly_score"].mean()
                 .sort_values(ascending=False).head(10).round(1).reset_index())
        top.columns = ["Month", "Mean anomaly score"]
        top["Month"] = top["Month"].dt.strftime("%Y-%m")
        st.markdown("##### Top anomalous months")
        st.dataframe(top, width="stretch", hide_index=True)

    st.markdown("##### " + ("All flagged anomalies" if show_all else f"Material {product.lower()} supplier anomalies"))
    flagged = df[df["anomaly_flag"] == 1].copy()
    if not show_all:
        flagged = flagged[materiality_mask(flagged)]
    flagged = flagged.sort_values("anomaly_score", ascending=False).head(60)
    if flagged.empty:
        st.info("No flagged anomalies under the current filter.")
    else:
        view = pd.DataFrame({
            "Month": flagged["date"].dt.strftime("%Y-%m"),
            "Supplier": flagged["supplier_country"],
            "Import Value": flagged["import_value_usd"].apply(fmt_money),
            "Share": flagged["supplier_share_value"].apply(fmt_pct),
            "Risk Score": flagged["supplyguard_risk_score_0_100"].round(1),
            "Anomaly Score": flagged["anomaly_score"].round(1),
            "Reason": flagged["anomaly_reason"],
        })
        st.dataframe(view, width="stretch", hide_index=True)
    callout("Materiality filter: supplier_share ≥ 1% OR import_value_usd ≥ $1M. Toggle above to see all statistical anomalies.")

# ============================================================
# 7. Forecasting — NEW PAGE
# ============================================================
elif "Forecasting" in page:
    page_header(
        f"Forecasting — {product} Next-Month Risk Prediction",
        "Supervised regression predicts next-month composite risk score per supplier. "
        "Selected on an earlier 3-month validation window; last 6 target months held out. Predicts an index, not disruption probability.",
    )

    callout(
        "This model forecasts the next-month composite risk score. It does not predict a binary "
        "disruption event because public disruption labels are not available."
    )

    if fc_metrics.empty or fc_predictions.empty:
        st.warning("No forecast outputs found. Run `python run_pipeline.py` to generate them.")
        st.stop()

    if fc_metrics_product.empty:
        st.info(f"No forecasting results for {product} — insufficient months for time-based split.")
        st.stop()

    selected_rows = fc_metrics_product.loc[fc_metrics_product["selected_on_validation"].astype(str).str.lower().eq("true")]
    best_row = selected_rows.iloc[0]
    best_name = best_row["model"]
    if best_row["n_test"] < 30:
        st.warning(f"Experimental forecast: only {int(best_row['n_test'])} evaluation rows for {product}. Estimates are illustrative and require more evidence.")
    baseline_row = fc_metrics_product[fc_metrics_product["model"] == "Baseline (persistence)"].iloc[0] \
                   if (fc_metrics_product["model"] == "Baseline (persistence)").any() else None

    # Supplier selector — restricted to material suppliers with a forecast row
    fc_material = fc_for_product[
        (fc_for_product["supplier_share_value"] >= config.MATERIAL_SHARE_THRESHOLD) |
        (fc_for_product["import_value_usd"] >= config.MATERIAL_VALUE_THRESHOLD_USD)
    ].copy()
    sup_options = sorted(fc_material["supplier_country"].unique().tolist()) if not fc_material.empty else \
                  sorted(fc_for_product["supplier_country"].unique().tolist())
    if not sup_options:
        st.info("No suppliers with forecasts.")
        st.stop()
    default = "India" if "India" in sup_options else sup_options[0]
    chosen_sup = st.selectbox("Supplier", sup_options, index=sup_options.index(default))
    sup_row = fc_for_product[fc_for_product["supplier_country"] == chosen_sup].iloc[0]
    current_risk = float(sup_row["current_risk_score"])
    forecast_risk = float(sup_row["forecast_next_month_risk_score"])
    delta_risk = forecast_risk - current_risk
    forecast_level = risk_level(forecast_risk)

    # KPI row 1 — supplier-specific forecast
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(kpi_html("Current risk", f"{current_risk:.1f}",
                                  pd.Timestamp(sup_row['as_of_date']).strftime('%b %Y')), unsafe_allow_html=True)
    with k2: st.markdown(kpi_html("Forecast next-month", f"{forecast_risk:.1f}",
                                  pd.Timestamp(sup_row['forecast_for_date']).strftime('%b %Y'),
                                  RISK_COLORS[forecast_level]), unsafe_allow_html=True)
    with k3:
        sign = "+" if delta_risk >= 0 else ""
        dcolor = RISK_COLORS["Critical"] if delta_risk > 0 else RISK_COLORS["Low"]
        st.markdown(kpi_html("Expected change", f"{sign}{delta_risk:.2f}", "risk-score points", dcolor), unsafe_allow_html=True)
    with k4: st.markdown(
        f'<div class="sg-kpi" style="border-left-color:{RISK_COLORS[forecast_level]};">'
        f'<div class="lbl">Forecast risk level</div><div class="val">{risk_badge_html(forecast_level)}</div>'
        f'<div class="sub">empirical bands</div></div>', unsafe_allow_html=True)
    with k5: st.markdown(kpi_html("Best model", best_name, f"selected by validation MAE"), unsafe_allow_html=True)

    st.markdown("###")

    # KPI row 2 — metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.markdown(kpi_html("MAE", f"{best_row['MAE']:.2f}", "lower is better"), unsafe_allow_html=True)
    with m2: st.markdown(kpi_html("RMSE", f"{best_row['RMSE']:.2f}", "lower is better"), unsafe_allow_html=True)
    with m3: st.markdown(kpi_html("R²", f"{best_row['R2']:.3f}", "higher is better"), unsafe_allow_html=True)
    with m4: st.markdown(kpi_html("Baseline MAE", f"{baseline_row['MAE']:.2f}" if baseline_row is not None else "—",
                                  "persistence model"), unsafe_allow_html=True)
    with m5:
        imp = (baseline_row["MAE"] - best_row["MAE"]) if baseline_row is not None else 0.0
        imp_color = RISK_COLORS["Low"] if imp > 0 else MUTED
        st.markdown(kpi_html("MAE improvement", f"{imp:+.2f}",
                             "vs persistence baseline", imp_color), unsafe_allow_html=True)

    # All-models table
    st.markdown("##### Model comparison (per product)")
    show_cols = ["product", "model", "validation_MAE", "selected_on_validation", "MAE", "RMSE", "R2", "n_test", "baseline_MAE", "mae_improvement_vs_baseline"]
    metrics_view = fc_metrics_product[show_cols].rename(columns={
        "product": "Product", "model": "Model", "n_test": "n test",
        "baseline_MAE": "Baseline MAE", "mae_improvement_vs_baseline": "Δ vs baseline (MAE)",
    })
    st.dataframe(metrics_view, width="stretch", hide_index=True)

    st.markdown("###")
    c1, c2 = st.columns([1.4, 1])

    with c1:
        # Actual vs predicted in held-out test window (across material suppliers)
        ap = fc_actual[fc_actual["product"] == product].copy() if not fc_actual.empty else pd.DataFrame()
        if not ap.empty:
            agg = ap.groupby("date").agg(
                actual=("actual_next_month_risk", "mean"),
                predicted=("predicted_next_month_risk", "mean"),
                baseline=("baseline_predicted", "mean"),
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=agg["date"], y=agg["actual"], name="Actual",
                                     line=dict(color=NAVY, width=2.5)))
            fig.add_trace(go.Scatter(x=agg["date"], y=agg["predicted"], name=f"Predicted ({best_name})",
                                     line=dict(color=TEAL, width=2.5, dash="dash")))
            fig.add_trace(go.Scatter(x=agg["date"], y=agg["baseline"], name="Baseline (persistence)",
                                     line=dict(color="#9AB2C5", width=1.5, dash="dot")))
            fig.update_layout(yaxis_title="Next-month risk score", xaxis_title="Month")
            st.plotly_chart(style_fig(fig, height=380,
                                      title=f"Actual vs Predicted — held-out test window ({product})"),
                            width="stretch")
        else:
            st.info("No held-out test predictions available.")

    with c2:
        # Feature importance — best model for selected product
        imp = fc_importance[(fc_importance["product"] == product) & (fc_importance["model"] == best_name)].copy() \
              if not fc_importance.empty else pd.DataFrame()
        if imp.empty:
            imp = fc_importance[fc_importance["product"] == product].copy() if not fc_importance.empty else pd.DataFrame()
        if not imp.empty:
            imp = imp.sort_values("importance", ascending=True).tail(12)
            # Friendly feature names
            label_map = {
                "risk_score_lag_1": "Risk score (lag 1)",
                "risk_score_lag_3": "Risk score (lag 3)",
                "risk_score_rolling_3m_mean": "Risk score (rolling 3m)",
                "risk_score_rolling_6m_mean": "Risk score (rolling 6m)",
                "supplier_share_lag_1": "Supplier share (lag 1)",
                "import_value_lag_1": "Import value (lag 1)",
                "import_value_yoy_change_lag_1": "Import YoY change (lag 1)",
                "hhi_lag_1": "HHI (lag 1)",
                "top3_supplier_share_lag_1": "Top-3 share (lag 1)",
                "commodity_price_stress_lag_1": "Price stress (lag 1)",
                "shipping_stress_lag_1": "Shipping stress (lag 1)",
                "import_anomaly_risk_lag_1": "Import anomaly (lag 1)",
                "logistics_risk_score_0_100": "Logistics risk (LPI)",
                "month_sin": "Seasonality (sin)",
                "month_cos": "Seasonality (cos)",
            }
            imp["Feature"] = imp["feature"].map(label_map).fillna(imp["feature"])
            fig = px.bar(imp, x="importance", y="Feature", orientation="h",
                         color="importance", color_continuous_scale=[(0, "#E7F5EC"), (1, TEAL_DARK)],
                         labels={"importance": "Importance (normalised)"})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, height=380, title=f"Feature importance ({best_name})"),
                            width="stretch")
            callout("The forecast is driven mainly by lagged risk score, supplier dependency, price stress, and shipping stress.")
        else:
            st.info("No feature importance available.")

    # Forecast table — all material suppliers
    st.markdown(f"##### Next-month forecast — all material {product.lower()} suppliers")
    if not fc_material.empty:
        view = pd.DataFrame({
            "Supplier": fc_material["supplier_country"],
            "Current share": fc_material["supplier_share_value"].apply(fmt_pct),
            "Latest import value": fc_material["import_value_usd"].apply(fmt_money),
            "Current risk": fc_material["current_risk_score"].round(1),
            "Forecast next-month": fc_material["forecast_next_month_risk_score"].round(1),
            "Δ": (fc_material["forecast_next_month_risk_score"] - fc_material["current_risk_score"]).round(2),
            "Level (forecast)": fc_material["forecast_next_month_risk_score"].apply(risk_level),
        }).sort_values("Forecast next-month", ascending=False)
        st.dataframe(view, width="stretch", hide_index=True)
    else:
        st.info("No material suppliers with forecasts.")

# ============================================================
# 8. Scenario Simulator
# ============================================================
elif "Scenario Simulator" in page:
    page_header(
        f"Scenario Simulator — {product} Import Risk",
        "What-if risk recalculation. Hypothetical stress test, not a forecast.",
    )

    callout("This is a what-if stress test, not an automatic forecast. It recalculates risk under user-defined assumptions. "
            "For the model-based next-month forecast, see the Forecasting page.")

    mode = st.radio("Scenario mode",
                    ["Supplier-level (recommended)", "Saudi-wide portfolio"],
                    horizontal=True)

    base = latest_snap_all.copy().reset_index(drop=True)
    if base.empty:
        st.warning("No latest data available.")
        st.stop()
    material_suppliers = sorted(latest_snap_material["supplier_country"].unique().tolist()) or sorted(base["supplier_country"].unique().tolist())
    default = "India" if "India" in material_suppliers else material_suppliers[0]

    SLIDER_KEYS = {
        "drop_pct": f"scn_drop_{product}",
        "ship_bump": f"scn_ship_{product}",
        "price_bump": f"scn_price_{product}",
        "diversify_pct": f"scn_div_{product}",
        "supplier": f"scn_sup_{product}",
    }
    for k, v in [("drop_pct", 0), ("ship_bump", 0), ("price_bump", 0), ("diversify_pct", 0)]:
        st.session_state.setdefault(SLIDER_KEYS[k], v)
    st.session_state.setdefault(SLIDER_KEYS["supplier"], default)

    st.markdown("##### Quick scenarios")
    pcols = st.columns(7)
    presets = [
        ("Reset", {"drop_pct": 0, "ship_bump": 0, "price_bump": 0, "diversify_pct": 0}),
        ("India ban-style shock", {"drop_pct": 50, "ship_bump": 10, "price_bump": 18, "diversify_pct": 0}),
        ("Red Sea escalation", {"drop_pct": 10, "ship_bump": 30, "price_bump": 10, "diversify_pct": 0}),
        ("Commodity price spike", {"drop_pct": 0, "ship_bump": 0, "price_bump": 25, "diversify_pct": 0}),
        ("Top supplier disruption", {"drop_pct": 40, "ship_bump": 20, "price_bump": 15, "diversify_pct": 0}),
        ("Diversification improvement", {"drop_pct": 0, "ship_bump": 0, "price_bump": 0, "diversify_pct": 25}),
        ("Severe combined stress", {"drop_pct": 60, "ship_bump": 45, "price_bump": 30, "diversify_pct": 0}),
    ]
    for col, (label, preset) in zip(pcols, presets):
        if col.button(label, use_container_width=True, key=f"preset_{label}_{product}"):
            for k, v in preset.items():
                st.session_state[SLIDER_KEYS[k]] = v
            st.rerun()

    chosen = st.selectbox("Focus supplier", material_suppliers, key=SLIDER_KEYS["supplier"])

    s1, s2 = st.columns(2)
    with s1:
        drop_pct = st.slider(f"Reduce {chosen} imports by", 0, 80, step=5, format="%d%%", key=SLIDER_KEYS["drop_pct"])
        ship_bump = st.slider("Add Red Sea / Bab el-Mandeb shipping stress", 0, 60, step=5, format="+%d pts", key=SLIDER_KEYS["ship_bump"])
    with s2:
        price_bump = st.slider("Commodity price shock", 0, 50, step=5, format="+%d%%", key=SLIDER_KEYS["price_bump"])
        diversify_pct = st.slider(f"Reassign {chosen} share to next 3 suppliers", 0, 40, step=5, format="%d%%", key=SLIDER_KEYS["diversify_pct"])

    # ---- Apply scenario ----
    sim_all = base.copy()
    idx_focus = sim_all.index[sim_all["supplier_country"] == chosen]
    base_focus_value = float(base.loc[idx_focus, "import_value_usd"].iloc[0]) if len(idx_focus) > 0 else 0.0

    if len(idx_focus) > 0:
        sim_all.loc[idx_focus, "import_value_usd"] = base_focus_value * (1 - drop_pct / 100)
        if diversify_pct > 0:
            others_idx = sim_all[sim_all["supplier_country"] != chosen].sort_values("supplier_share_value", ascending=False).head(3).index
            redirect_value = base_focus_value * (diversify_pct / 100)
            sim_all.loc[idx_focus, "import_value_usd"] = max(sim_all.loc[idx_focus, "import_value_usd"].iloc[0] - redirect_value, 0.0)
            if len(others_idx) > 0:
                per = redirect_value / len(others_idx)
                sim_all.loc[others_idx, "import_value_usd"] += per

    total = sim_all["import_value_usd"].sum()
    if total > 0:
        sim_all["supplier_share_value"] = sim_all["import_value_usd"] / total
    sim_all["supplier_dependency_score_0_100"] = (sim_all["supplier_share_value"] * 100).clip(0, 100)
    new_top3 = float(sim_all.nlargest(3, "supplier_share_value")["supplier_share_value"].sum())
    sim_all["top_3_supplier_share"] = new_top3
    sim_all["HHI"] = float((sim_all["supplier_share_value"] ** 2).sum())
    sim_all["concentration_score_0_100"] = (new_top3 * 100)
    sim_all["shipping_stress_score_0_100"] = (sim_all["shipping_stress_score_0_100"].fillna(0) + ship_bump).clip(0, 100)
    sim_all["price_stress_score_0_100"] = (sim_all["price_stress_score_0_100"].fillna(0) + price_bump * 0.7).clip(0, 100)
    if drop_pct > 0 and len(idx_focus) > 0:
        sim_all.loc[idx_focus, "import_anomaly_score_0_100"] = (
            sim_all.loc[idx_focus, "import_anomaly_score_0_100"].fillna(0) + float(drop_pct)
        ).clip(0, 100)

    w_sum = sum(COMPONENT_WEIGHTS.values())
    sim_all["supplyguard_risk_score_0_100"] = sum(
        sim_all[c].fillna(0) * (w / w_sum) for c, w in COMPONENT_WEIGHTS.items()
    ).clip(0, 100)

    if mode.startswith("Supplier-level"):
        before_score = float(base.loc[base["supplier_country"] == chosen, "supplyguard_risk_score_0_100"].iloc[0])
        after_score = float(sim_all.loc[sim_all["supplier_country"] == chosen, "supplyguard_risk_score_0_100"].iloc[0])
        before_components = {c: float(base.loc[base["supplier_country"] == chosen, c].iloc[0]) for c in COMPONENT_WEIGHTS}
        after_components = {c: float(sim_all.loc[sim_all["supplier_country"] == chosen, c].iloc[0]) for c in COMPONENT_WEIGHTS}
        scope_label = f"{chosen} (supplier-level)"
    else:
        w_b = base["supplier_share_value"].fillna(0)
        w_a = sim_all["supplier_share_value"].fillna(0)
        before_score = float((base["supplyguard_risk_score_0_100"] * w_b).sum() / max(w_b.sum(), 1e-9))
        after_score = float((sim_all["supplyguard_risk_score_0_100"] * w_a).sum() / max(w_a.sum(), 1e-9))
        before_components = {c: float((base[c] * w_b).sum() / max(w_b.sum(), 1e-9)) for c in COMPONENT_WEIGHTS}
        after_components = {c: float((sim_all[c] * w_a).sum() / max(w_a.sum(), 1e-9)) for c in COMPONENT_WEIGHTS}
        scope_label = f"Saudi-wide {product} portfolio"

    delta = after_score - before_score
    delta_color = RISK_COLORS["Critical"] if delta > 0.5 else (RISK_COLORS["Low"] if delta < -0.5 else MUTED)
    level_before, level_after = risk_level(before_score), risk_level(after_score)
    deltas = {c: after_components[c] - before_components[c] for c in COMPONENT_WEIGHTS}
    main_driver_col = max(deltas, key=lambda c: abs(deltas[c] * COMPONENT_WEIGHTS[c]))
    main_driver = COMPONENT_COLS[main_driver_col]
    main_driver_delta = deltas[main_driver_col]

    st.markdown("###")
    g1, g2, g3 = st.columns([1, 1, 1])
    with g1: st.plotly_chart(gauge(before_score, f"Before · {scope_label}", height=260), width="stretch")
    with g2: st.plotly_chart(gauge(after_score, f"After · {scope_label}", height=260), width="stretch")
    with g3:
        sign = "+" if delta >= 0 else ""
        st.markdown(
            f"""
            <div class="sg-kpi" style="border-left-color:{delta_color};height:260px;display:flex;flex-direction:column;justify-content:center;">
                <div class="lbl">Δ Risk Score</div>
                <div style="font-size:42px;font-weight:800;color:{delta_color};">{sign}{delta:.2f}</div>
                <div class="sub">Before: {before_score:.1f} → After: {after_score:.1f}</div>
                <div style="margin-top:10px;display:flex;gap:8px;">
                    {risk_badge_html(level_before)} <span style="color:{MUTED};">→</span> {risk_badge_html(level_after)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    r1, r2, r3 = st.columns(3)
    with r1: st.markdown(kpi_html("Main driver of change", main_driver, f"Δ {main_driver_delta:+.1f} pts"), unsafe_allow_html=True)
    with r2: st.markdown(kpi_html("Risk level before", level_before, "", RISK_COLORS[level_before]), unsafe_allow_html=True)
    with r3: st.markdown(kpi_html("Risk level after", level_after, "", RISK_COLORS[level_after]), unsafe_allow_html=True)

    comp_change = pd.DataFrame({
        "Component": [COMPONENT_COLS[c] for c in COMPONENT_WEIGHTS],
        "Before": [before_components[c] for c in COMPONENT_WEIGHTS],
        "After": [after_components[c] for c in COMPONENT_WEIGHTS],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", y=comp_change["Component"], x=comp_change["Before"],
                         orientation="h", marker_color="#9AB2C5",
                         text=[f"{v:.0f}" for v in comp_change["Before"]], textposition="auto"))
    fig.add_trace(go.Bar(name="After", y=comp_change["Component"], x=comp_change["After"],
                         orientation="h", marker_color=TEAL,
                         text=[f"{v:.0f}" for v in comp_change["After"]], textposition="auto"))
    fig.update_layout(barmode="group", xaxis_title="Component score (0–100)")
    st.plotly_chart(style_fig(fig, height=380, title="Component change · Before vs After"), width="stretch")

    # ---- What changed? ----
    biggest = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:3]
    bits = [f"{COMPONENT_COLS[c]} {('+' if d>=0 else '')}{d:.1f}" for c, d in biggest if abs(d) >= 0.5]
    what_changed_text = "; ".join(bits) if bits else "no component changed meaningfully"
    st.markdown(
        f"""
        <div class="sg-card">
            <div style="font-size:14px;color:{MUTED};text-transform:uppercase;letter-spacing:.04em;">What changed?</div>
            <div style="font-size:15px;color:{INK};margin-top:6px;line-height:1.5;">
                The scenario {('increases' if delta > 0 else ('decreases' if delta < 0 else 'leaves unchanged'))}
                the {scope_label} risk score because: {what_changed_text}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Driver-aware Operational Recommendation ----
    DRIVER_RECS = {
        "supplier_dependency_score_0_100": (
            f"Reduce dependence on {chosen} by reallocating part of procurement to lower-risk alternative suppliers; "
            f"target a top-supplier share below 50%."),
        "concentration_score_0_100": (
            f"Diversify the top-3 supplier mix — current concentration leaves the {product} supply chain exposed to a single-country shock."),
        "import_anomaly_score_0_100": (
            f"Investigate recent {chosen} shipment performance, verify upcoming shipment reliability, and prepare contingency volumes from secondary suppliers."),
        "price_stress_score_0_100": (
            f"Review procurement timing, contract exposure (spot vs forward), and inventory strategy due to {product.lower()} commodity-price pressure."),
        "logistics_risk_score_0_100": (
            f"Reassess supplier logistics-performance assumptions for {chosen}; consider shifting share to suppliers with stronger LPI."),
        "shipping_stress_score_0_100": (
            f"Monitor Red Sea / Bab el-Mandeb route conditions; pre-position inventory buffers or arrange Cape-of-Good-Hope routing contingency."),
    }
    if delta > 5:
        rec_intro = "Activate contingency sourcing. "
        rcolor = RISK_COLORS["Critical"]
    elif delta > 0.5:
        rec_intro = "Monitor the situation closely and prepare secondary options. "
        rcolor = RISK_COLORS["High"]
    elif delta < -3:
        rec_intro = "Diversification improves resilience under this scenario. "
        rcolor = RISK_COLORS["Low"]
    else:
        rec_intro = "Scenario impact is marginal under current assumptions. "
        rcolor = MUTED
    rec_main = DRIVER_RECS.get(main_driver_col, "Re-evaluate the dominant driver in the component breakdown above.")
    rec = rec_intro + rec_main

    # Business value impact text
    if delta < -0.5:
        biz_text = "Operational impact: diversification reduces exposure to a single supplier and improves resilience."
    elif delta > 5:
        biz_text = "Operational impact: this scenario materially worsens supply-chain exposure and may require contingency sourcing or inventory buffering."
    elif delta > 0.5:
        biz_text = "Operational impact: elevated risk warrants closer monitoring and secondary-supplier readiness."
    else:
        biz_text = "Operational impact: the scenario is small and unlikely to require operational action."

    # Exposed value
    if len(idx_focus) > 0:
        exposed_value_now = base_focus_value
        exposed_value_after = float(sim_all.loc[sim_all["supplier_country"] == chosen, "import_value_usd"].iloc[0])
        biz_text += (f"<br><b>Current exposed monthly import value from {chosen}:</b> {fmt_money(exposed_value_now)}. "
                     f"Scenario adjusts this to {fmt_money(exposed_value_after)}.")

    st.markdown(
        f"""
        <div class="sg-card" style="border-left:4px solid {rcolor};">
            <div style="font-size:14px;color:{MUTED};text-transform:uppercase;letter-spacing:.04em;">Operational Recommendation</div>
            <div style="font-size:15px;color:{INK};margin-top:6px;line-height:1.55;">{rec}</div>
            <div style="font-size:13.5px;color:{MUTED};margin-top:10px;line-height:1.55;">{biz_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Alternative suppliers
    alts = (latest_snap_material[latest_snap_material["supplier_country"] != chosen]
            .sort_values("supplier_share_value", ascending=False).head(3))
    if not alts.empty:
        st.markdown(f"##### Alternative {product.lower()} suppliers (excluding {chosen})")
        alt_view = pd.DataFrame({
            "Supplier": alts["supplier_country"],
            "Current share": alts["supplier_share_value"].apply(fmt_pct),
            "Latest import value": alts["import_value_usd"].apply(fmt_money),
            "Current risk": alts["supplyguard_risk_score_0_100"].round(1),
            "Risk level": alts["risk_level"],
        })
        st.dataframe(alt_view, width="stretch", hide_index=True)

    # Download summary
    summary = {
        "product": product, "scenario_mode": mode, "focus_supplier": chosen,
        "scope": scope_label,
        "sliders": {
            "import_reduction_pct": drop_pct,
            "shipping_stress_bump_pts": ship_bump,
            "price_shock_pct": price_bump,
            "diversify_pct": diversify_pct,
        },
        "before_risk_score": round(before_score, 2),
        "after_risk_score": round(after_score, 2),
        "delta_risk_score": round(delta, 2),
        "risk_level_before": level_before, "risk_level_after": level_after,
        "main_driver_of_change": main_driver,
        "main_driver_delta": round(main_driver_delta, 2),
        "components_before": {COMPONENT_COLS[c]: round(before_components[c], 2) for c in COMPONENT_WEIGHTS},
        "components_after": {COMPONENT_COLS[c]: round(after_components[c], 2) for c in COMPONENT_WEIGHTS},
        "recommendation": rec, "operational_impact": biz_text.replace("<br>", " ").replace("<b>", "").replace("</b>", ""),
    }
    csv_buf = io.StringIO()
    flat_rows = [{"key": k, "value": v if not isinstance(v, dict) else json.dumps(v)} for k, v in summary.items()]
    pd.DataFrame(flat_rows).to_csv(csv_buf, index=False)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇  Download summary (CSV)", data=csv_buf.getvalue(),
                           file_name=f"supplyguard_scenario_{product}_{chosen}.csv",
                           mime="text/csv", use_container_width=True)
    with d2:
        st.download_button("⬇  Download summary (JSON)", data=json.dumps(summary, indent=2),
                           file_name=f"supplyguard_scenario_{product}_{chosen}.json",
                           mime="application/json", use_container_width=True)

    # ============================================================
    # AI-Enhanced Recommendation (optional, advisory only)
    # ============================================================
    st.markdown("###")
    st.markdown(
        f'<div class="sg-title" style="font-size:22px;">AI-Enhanced Recommendation</div>'
        f'<div class="sg-subtitle">Optional natural-language summary of the scenario above, produced by Google Gemini. '
        f'Core risk scoring, forecasting, and scenario calculations remain deterministic and explainable.</div>'
        f'<hr class="sg-rule" />',
        unsafe_allow_html=True,
    )

    # Lazy import so the dashboard works even if google-genai is not installed.
    try:
        from src.ai_recommendation import (
            load_gemini_client, build_decision_packet,
            generate_gemini_recommendation, fallback_rule_based_recommendation,
        )
        _ai_available = True
        _ai_import_error = None
    except Exception as e:
        _ai_available = False
        _ai_import_error = str(e)[:160]

    if _ai_available:
        gemini_client, gemini_status = load_gemini_client()
    else:
        gemini_client, gemini_status = None, None

    # Status indicator
    if not _ai_available:
        status_color, status_text = RISK_COLORS["High"], f"Module unavailable: {_ai_import_error}"
    elif gemini_client is not None and gemini_status and gemini_status.connected:
        status_color, status_text = RISK_COLORS["Low"], f"Gemini: Connected · model={gemini_status.model}"
    else:
        msg = gemini_status.message if gemini_status else "unknown"
        status_color, status_text = RISK_COLORS["Medium"], f"Gemini: Not configured ({msg})"

    s1, s2 = st.columns([2, 1])
    with s1:
        st.markdown(
            f'<div class="sg-kpi" style="border-left-color:{status_color};">'
            f'<div class="lbl">AI status</div>'
            f'<div class="val" style="font-size:14px;">{status_text}</div>'
            f'<div class="sub">Optional advisory layer. Fallback recommendation is always available.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with s2:
        generate_clicked = st.button(
            "🤖 Generate AI-Enhanced Recommendation",
            use_container_width=True,
            disabled=not _ai_available,
            key=f"gemini_btn_{product}_{chosen}",
        )

    callout(
        "AI-enhanced recommendation is optional. Core risk scoring, forecasting, and scenario "
        "calculations remain deterministic and explainable. Gemini only converts the existing "
        "structured analytics output into a clearer operational summary."
    )

    # Build the decision packet (used both for the API call and the expander)
    if _ai_available:
        # Alternative suppliers (top-3 material, excluding focus)
        alts_pkt = (latest_snap_material[latest_snap_material["supplier_country"] != chosen]
                    .sort_values("supplier_share_value", ascending=False).head(3))
        alt_list = [
            {
                "supplier_name": r["supplier_country"],
                "current_share": float(r["supplier_share_value"]),
                "latest_import_value_usd": float(r["import_value_usd"]),
                "current_risk_score": float(r["supplyguard_risk_score_0_100"]),
                "risk_level": r["risk_level"],
            }
            for _, r in alts_pkt.iterrows()
        ] if not alts_pkt.empty else []

        # Per-supplier next-month forecast (if we have one for the focus supplier)
        fc_for_chosen = fc_for_product[fc_for_product["supplier_country"] == chosen] \
                        if not fc_for_product.empty else pd.DataFrame()
        forecast_score_for_supplier = float(fc_for_chosen["forecast_next_month_risk_score"].iloc[0]) \
                                       if not fc_for_chosen.empty else None
        model_note = (f"Best per-product model: {fc_for_chosen['model'].iloc[0]}"
                      if not fc_for_chosen.empty else None)

        packet = build_decision_packet(
            product_name=product,
            hs_code=config.PRODUCT_HS.get(product, "—"),
            selected_supplier=chosen,
            scenario_mode=mode,
            current_risk_score=float(base.loc[base["supplier_country"] == chosen,
                                              "supplyguard_risk_score_0_100"].iloc[0])
                                if (base["supplier_country"] == chosen).any() else None,
            forecasted_next_month_risk_score=forecast_score_for_supplier,
            before_scenario_risk=before_score,
            after_scenario_risk=after_score,
            risk_level_before=level_before,
            risk_level_after=level_after,
            top_risk_driver=main_driver,
            second_risk_driver=None,
            component_scores_before={COMPONENT_COLS[c]: before_components[c] for c in COMPONENT_WEIGHTS},
            component_scores_after={COMPONENT_COLS[c]: after_components[c] for c in COMPONENT_WEIGHTS},
            supplier_share=float(base.loc[base["supplier_country"] == chosen,
                                          "supplier_share_value"].iloc[0])
                            if (base["supplier_country"] == chosen).any() else None,
            import_value_usd=base_focus_value,
            alternative_suppliers=alt_list,
            scenario_parameters={
                "import_reduction_pct": drop_pct,
                "price_shock_pct": price_bump,
                "shipping_stress_shock_points": ship_bump,
                "reassignment_pct": diversify_pct,
            },
            model_forecast_note=model_note,
        )
    else:
        packet = None

    # Generate only on click — never auto-fire on slider changes
    if generate_clicked and _ai_available and packet is not None:
        with st.spinner("Calling Gemini…"):
            result = generate_gemini_recommendation(packet, client=gemini_client)
        st.session_state[f"_ai_result_{product}_{chosen}"] = result

    result = st.session_state.get(f"_ai_result_{product}_{chosen}")
    if result is not None:
        if result.source == "gemini":
            card_color = TEAL_DARK
            badge = f'<span class="sg-badge" style="color:white;background:{TEAL_DARK};">Gemini · {result.model_used}</span>'
        else:
            card_color = RISK_COLORS["Medium"]
            err_part = f" — {result.error}" if result.error else ""
            badge = f'<span class="sg-badge" style="color:{RISK_COLORS["Medium"]};background:{RISK_BG["Medium"]};">Fallback (rule-based){err_part}</span>'

        body_html = (result.text or "")
        # Convert basic markdown bullets / headings to HTML for the card
        body_html = re.sub(r"^(\d\.\s+\S.*)$", r"<h5 style='margin:14px 0 4px;color:#0B1F33;'>\1</h5>",
                           body_html, flags=re.MULTILINE)
        body_html = re.sub(r"^-\s+(.*)$", r"<li>\1</li>", body_html, flags=re.MULTILINE)
        body_html = re.sub(r"(<li>.*?</li>(\n)?)+",
                           lambda m: "<ul>" + m.group(0) + "</ul>", body_html, flags=re.DOTALL)
        body_html = body_html.replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="sg-card" style="border-left:4px solid {card_color};">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                    <div style="font-size:14px;color:{MUTED};text-transform:uppercase;letter-spacing:.04em;">AI-Enhanced Recommendation</div>
                    {badge}
                </div>
                <div style="font-size:14.5px;color:{INK};line-height:1.6;">{body_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for w in (result.warnings or []):
            callout(f"⚠ {w}", kind="warn")

    # Decision packet expander (never includes the API key — it's not part of the packet)
    if _ai_available and packet is not None:
        with st.expander("View structured decision packet (sent to Gemini)"):
            st.json(packet)


# ============================================================
# 9. Methodology
# ============================================================
elif "Methodology" in page:
    page_header("Methodology",
                "Transparent composite operational risk index — not a supervised disruption label.")

    st.markdown("#### 🎯 Target variable")
    st.markdown(
        "**`supplyguard_risk_score_0_100`** is a *transparent composite operational risk index* "
        "computed at the **product × supplier country × month** level. "
        "Public data does not provide ground-truth `disruption / no-disruption` labels, so we "
        "do not train a binary classifier and we do not report accuracy / F1 / AUC. "
        "The score is **calibrated for operational warning, not a supervised probability of disruption**. "
        "Risk levels are calibrated to the observed empirical distribution and operational interpretability."
    )

    st.markdown("#### 📥 Data sources")
    src_table = pd.DataFrame([
        ["UN Comtrade", "Bilateral monthly import volume/value per product (HS 1006, 1001, 1003, 1005, 1701), reporter SAU"],
        ["World Bank Pink Sheet", "Monthly commodity-price proxy per product"],
        ["World Bank Logistics Performance Index", "Static supplier logistics performance score (LPI 2023)"],
        ["IMF PortWatch", "Monthly chokepoint + port stress (Bab-el-Mandeb, Hormuz, Jeddah, KAP, Dammam)"],
        ["Curated event annotations", "India ban, Suez, COVID, Red Sea, Hormuz, Black Sea grain, sugar price stress"],
    ], columns=["Source", "Used for"])
    st.dataframe(src_table, width="stretch", hide_index=True)

    st.markdown("#### 🧹 Data preprocessing")
    if not dq_summary.empty:
        st.markdown("**Per-product data coverage (after preprocessing):**")
        st.dataframe(dq_summary, width="stretch", hide_index=True)
    if not miss_summary.empty:
        st.markdown("**Missingness by field group:**")
        try:
            pivot = miss_summary.pivot(index="Field group", columns="Product", values="Missing %").fillna(0)
            fig = px.imshow(pivot, color_continuous_scale=[(0, "#E7F5EC"), (1, "#C03A2B")],
                            aspect="auto", labels=dict(color="Missing %"),
                            text_auto=".1f")
            st.plotly_chart(style_fig(fig, height=320, title="Missing % by product × field group"), width="stretch")
        except Exception:
            st.dataframe(miss_summary, width="stretch", hide_index=True)

    st.markdown("#### 🛠️ Feature engineering")
    st.markdown(
        "Six feature groups feed the composite: **supplier dependency** (share), "
        "**market concentration** (HHI, top-3 share), **import trend / anomaly** (YoY %, "
        "rolling 12-month z-score on residuals), **commodity price stress** (Pink Sheet "
        "z-score + volatility), **logistics risk** (LPI transformation), and **maritime "
        "shipping stress** (PortWatch chokepoint + port stress, product-route weighted)."
    )

    st.markdown("#### 🧮 Risk score formula")
    st.code(
        "supplyguard_risk_score_0_100 = Σ (component_i × weight_i)\n\n"
        "where each component_i ∈ [0, 100], Σ weights = 1.0",
        language="text",
    )
    weights_table = pd.DataFrame([
        ["Supplier dependency", "20%", "supplier_share × 100"],
        ["Market concentration", "15%", "top_3_share × 100 (HHI fallback)"],
        ["Import drop / anomaly", "20%", "max(0, −YoY%) × 100"],
        ["Commodity price stress", "15%", "0.7·pos_z(price) + 0.3·volatility, scaled"],
        ["Logistics risk", "10%", "((5 − LPI) / 4) × 100"],
        ["Maritime shipping stress", "20%", "product-route blend of Red Sea / Gulf PortWatch"],
    ], columns=["Component", "Weight", "Formula"])
    st.dataframe(weights_table, width="stretch", hide_index=True)

    st.markdown("#### 🎚️ Risk levels (empirical bands)")
    levels_table = pd.DataFrame([
        ["Low", "< 25", "Routine operational state"],
        ["Medium", "25 – 40", "Watchlist; monitor key drivers"],
        ["High", "40 – 60", "Elevated risk; diversification or hedging warranted"],
        ["Critical", "≥ 60", "Material exposure; act on dominant driver immediately"],
    ], columns=["Level", "Score range", "Operational interpretation"])
    st.dataframe(levels_table, width="stretch", hide_index=True)
    callout("Bands are calibrated to the observed empirical distribution, not the textbook 0/25/50/75.")

    st.markdown("#### 🚨 Anomaly detection")
    st.markdown(
        "Two unsupervised methods run side by side per product:\n"
        "1. **Rolling-residual z-score** — z of monthly import value vs trailing 12-month mean per supplier.\n"
        "2. **Isolation Forest** — multivariate outlier detection on **driver variables only** "
        "(import value, MoM/YoY change, supplier share, price stress, shipping stress, logistics risk). "
        "The composite `supplyguard_risk_score_0_100` is deliberately **excluded** from the Isolation Forest "
        "feature set, because the composite is itself built from those drivers — feeding it back in would "
        "produce a circular anomaly interpretation. A flag exists to add it back for diagnostic comparison only.\n\n"
        "A row is flagged if **either** method fires. Materiality filter (share ≥ 1% or value ≥ $1M) is applied by default in the dashboard."
    )

    st.markdown("#### 🔮 Forecasting layer")
    st.markdown(
        "Per-product supervised regression predicts `next_month_supplyguard_risk_score_0_100`. "
        "Time-based split (last 6 months held out, no random shuffling). Models compared: "
        "**Baseline (persistence)**, **LinearRegression** (scaled), **RandomForest**, and "
        "**XGBoost** (falls back to HistGradientBoosting if XGB not installed). Best model per "
        "product is selected by lowest validation MAE, including persistence. The last 6 target months are held out for evaluation. Feature importance is reported on the Forecasting page."
    )

    st.markdown("#### 🛡️ Leakage prevention")
    st.markdown(
        "Forecasting predicts **next-month** risk. The target is\n\n"
        "```\n"
        "next_month_supplyguard_risk_score_0_100 = "
        "groupby([product_name, supplier_country]).supplyguard_risk_score_0_100.shift(-1)\n"
        "```\n\n"
        "All predictors are **lagged** (`risk_score_lag_1`, `supplier_share_lag_1`, "
        "`price_stress_lag_1`, `shipping_stress_lag_1`, `import_anomaly_risk_lag_1`, "
        "`hhi_lag_1`, `top3_supplier_share_lag_1`, `import_value_lag_1`) or **as-of T** "
        "(`logistics_risk_score_0_100` — slowly-changing LPI; `month_sin`/`month_cos` — calendar). "
        "Rolling features use `series.shift(1).rolling(window).mean()` so the window never sees future data.\n\n"
        "We **do not** train any model to reproduce the same-month composite score from its own "
        "components — doing so is circular and would inflate apparent accuracy. The "
        "**persistence baseline** (predicted next-month = current risk) is kept as a valid time-series "
        "reference. Every pipeline run regenerates a feature-by-feature leakage audit at "
        "`data/outputs/forecast_leakage_audit.csv`; any feature marked `REMOVE` is dropped before training."
    )
    if not fc_metrics.empty:
        best_per_product = fc_metrics.loc[fc_metrics["selected_on_validation"].astype(str).str.lower().eq("true")].copy()
        st.markdown("**Model selected per product (validation MAE):**")
        st.dataframe(best_per_product[["product", "model", "MAE", "RMSE", "R2",
                                        "baseline_MAE", "mae_improvement_vs_baseline"]]
                     .rename(columns={"product": "Product", "model": "Best model",
                                      "baseline_MAE": "Baseline MAE",
                                      "mae_improvement_vs_baseline": "Δ vs baseline (MAE)"}),
                     width="stretch", hide_index=True)

    st.markdown("#### 🛡️ Leakage prevention")
    st.markdown(
        "Forecasting predicts **next-month** risk (`next_month_supplyguard_risk_score_0_100`, "
        "constructed by shifting the composite score by −1 per product × supplier series). "
        "All predictors are **lagged or known as of the current month**: lag-1 / lag-3 of risk, "
        "supplier share, import value, HHI, top-3 share, price stress, shipping stress, and "
        "import-anomaly score; rolling means use `series.shift(1).rolling(...)` so the window "
        "only sees past values; LPI and seasonality (sin/cos of month) are legitimate as-of-T "
        "statics. We **do not** train any model to reproduce the same-month composite score from "
        "its own components — that would be circular.\n\n"
        "Every column passing through the pipeline is classified by `audit_features()` as "
        "**OK / WARNING / REMOVE** before training; any REMOVE-classified feature is automatically "
        "dropped from the feature set. The full audit is saved to "
        "`data/outputs/forecast_leakage_audit.csv`."
    )
    if not fc_leakage_audit.empty:
        counts = fc_leakage_audit["leakage_risk"].value_counts().to_dict()
        in_model_rows = fc_leakage_audit[fc_leakage_audit["in_model"] == True]
        n_remove_in_model = int(((fc_leakage_audit["leakage_risk"] == "REMOVE") &
                                  (fc_leakage_audit["in_model"] == True)).sum())
        a1, a2, a3 = st.columns(3)
        with a1: st.markdown(kpi_html("Audited columns", str(len(fc_leakage_audit)), "all pipeline columns"), unsafe_allow_html=True)
        with a2:
            ok_count = counts.get("OK", 0)
            color = RISK_COLORS["Low"] if ok_count == len(fc_leakage_audit) else RISK_COLORS["High"]
            st.markdown(kpi_html("Classified OK", f"{ok_count}", f"of {len(fc_leakage_audit)}", color), unsafe_allow_html=True)
        with a3:
            color = RISK_COLORS["Critical"] if n_remove_in_model > 0 else RISK_COLORS["Low"]
            st.markdown(kpi_html("Leakage in active model", f"{n_remove_in_model}", "REMOVE-flagged features used", color), unsafe_allow_html=True)
        st.markdown("**Features actually used by the forecasting model:**")
        view = in_model_rows[["feature", "lagged_or_as_of", "leakage_risk", "reason"]].rename(
            columns={"feature": "Feature", "lagged_or_as_of": "Type",
                     "leakage_risk": "Risk", "reason": "Reason"}
        )
        st.dataframe(view, width="stretch", hide_index=True)
        if n_remove_in_model == 0:
            callout("All forecasting features are lag, rolling-shifted, or as-of-T statics. No leakage detected.")
        else:
            callout(f"⚠ {n_remove_in_model} REMOVE-classified features detected in the active model. Re-run the pipeline.", kind="warn")
        with st.expander("Full audit (all 116 columns including those NOT used as features)"):
            st.dataframe(fc_leakage_audit, width="stretch", hide_index=True)
    else:
        callout("Audit CSV not found — run `python run_pipeline.py` to generate.", kind="warn")

    st.markdown("#### 🎛️ Sensitivity analysis (±20% one-at-a-time)")
    st.markdown(
        "Each component weight is perturbed +20% and -20%, weights are renormalised to sum to 1, "
        "the composite is recomputed, and the top-10 material-supplier ranking is compared with "
        "the base ranking. Stable rankings indicate the index is robust to reasonable weight choices."
    )
    if not sens_summary.empty:
        st.dataframe(sens_summary, width="stretch", hide_index=True)
        stable = sens_summary["Stability score"].astype(str).str.contains("stable", case=False).all()
        callout(
            "The ranking remains stable under reasonable weight perturbations, supporting the robustness "
            "of the composite index." if stable else
            "Some products show fragility under weight perturbation — review the detail CSV."
        )
        if not sens_detail.empty:
            with st.expander("Detail (per product × component × perturbation)"):
                st.dataframe(sens_detail, width="stretch", hide_index=True)

    st.markdown("#### 🧪 Scenario simulator")
    st.markdown(
        "Deterministic what-if engine: re-applies user-defined shocks (import reduction, shipping stress, "
        "price shock, diversification reallocation) to the latest month, then re-aggregates the composite. "
        "Not a forecast — used for decision-support stress testing only."
    )

    st.markdown("#### ✅ Validation & limitations")
    st.markdown(
        "1. **Historical event backtesting** — India non-basmati ban, Suez, Red Sea / Bab-el-Mandeb, COVID-19, Russia-Ukraine grain disruption.\n"
        "2. **Anomaly precision@k** — fraction of top-k anomaly months within ±60 days of any known event.\n"
        "3. **Per-product time-based forecasting** — last 6 months held out; MAE / RMSE / R² vs persistence baseline.\n"
        "4. **Sensitivity analysis** — weights varied ±20% per OECD/JRC handbook.\n\n"
        "Limitations: trade data lags 3–6 months (early-warning, not real-time); bilateral aggregates only; "
        "LPI 2023 latest; composite weighting judgmental; event impact = proximity in time, not causal; "
        "some events (e.g. 2026 Hormuz) sit outside the trade-data range — those use the PortWatch fallback."
    )

    st.markdown("#### 🤖 AI-Enhanced Recommendation Layer (optional)")
    st.markdown(
        "The default recommendation engine is **transparent and rule-based** — driver-aware text "
        "tailored to the largest component change in the scenario. Optionally, the Scenario "
        "Simulator can call **Google Gemini** to convert the scenario output into a more polished "
        "natural-language recommendation.\n\n"
        "- Gemini receives a **compact structured decision packet**, not the full dataset.\n"
        "- Gemini **does not** calculate the risk score, forecast, anomaly score, or scenario impact.\n"
        "- The prompt explicitly tells the model to **use only the provided packet** and not to "
        "invent suppliers, values, events, or savings.\n"
        "- If Gemini is unavailable (no `GEMINI_API_KEY`, SDK missing, network failure, or model "
        "error), the dashboard falls back to the deterministic rule-based recommendation.\n\n"
        "**The AI layer is used for natural-language decision support only. The quantitative "
        "analytics remain deterministic and auditable.**"
    )

    st.markdown("#### ❓ Why no classification accuracy")
    st.markdown(
        "Because we **do not have labels.** Public datasets do not tell us, for every supplier × product × month, "
        "whether a 'disruption' occurred. Inventing such labels would be misleading; reporting accuracy "
        "against invented labels would be dishonest. The OECD/JRC composite-indicator approach is the "
        "standard way to handle this situation."
    )

    callout(
        "<b>Judge answer:</b> Our target is a transparent composite operational risk index from 0 to 100 "
        "at the product × supplier × month level. Public datasets do not provide reliable "
        "disruption/no-disruption labels, so we do not train a binary classifier or report accuracy/F1/AUC. "
        "We validate the index through historical event backtesting, anomaly precision around known events, "
        "sensitivity analysis, and next-month risk-score forecasting against a persistence baseline."
    )

    if not methodology.empty:
        with st.expander("Original methodology CSV"):
            st.dataframe(methodology, width="stretch", hide_index=True)

    st.markdown('<div class="sg-foot">Hackathon prototype — built for explainable food import risk monitoring.</div>',
                unsafe_allow_html=True)
