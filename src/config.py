"""Centralized paths, constants, column names, and component weights."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

for d in (PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _resolve(name: str) -> Path:
    """Prefer raw/ copies; fall back to data/ root."""
    p = RAW_DIR / name
    return p if p.exists() else DATA_DIR / name


# --- Multi-product (current source of truth, May 2026) ---
MAIN_FILE = _resolve("supplyguard_food_imports_multiproduct_features_2018_2025_with_barley.csv")
EVENTS_FILE = _resolve("known_disruption_events_multiproduct_supplyguard_with_barley.csv")
COVERAGE_FILE = _resolve("supplyguard_multiproduct_coverage_summary_with_barley.csv")
PRICES_FILE = _resolve("world_bank_pink_sheet_food_price_proxies_2018_2025_with_barley.csv")
ANNUAL_TOP_FILE = _resolve("supplyguard_food_imports_annual_top_suppliers_summary_with_barley.csv")
LATEST_MATERIAL_FILE = _resolve("supplyguard_latest_material_supplier_risk_multiproduct_with_barley.csv")

PORTWATCH_WIDE_FILE = _resolve("portwatch_monthly_shipping_features_2019_2026.csv")
PORTWATCH_LONG_FILE = _resolve("portwatch_monthly_long_2019_2026.csv")

# --- Archive / reference only (rice-only legacy) ---
LEGACY_MAIN_FILE = _resolve("supplyguard_rice_final_features_with_portwatch_risk_2018_2025.csv")
METHODOLOGY_FILE = _resolve("supplyguard_risk_score_methodology.csv")

# --- Output paths ---
CLEANED_FILE = PROCESSED_DIR / "supplyguard_cleaned.csv"
MODELING_FILE = PROCESSED_DIR / "supplyguard_multiproduct_modeling_dataset.csv"
LATEST_RISK_FILE = PROCESSED_DIR / "supplyguard_latest_supplier_risk.csv"
MODEL_METRICS_FILE = OUTPUTS_DIR / "model_metrics.csv"
ANOMALY_RESULTS_FILE = OUTPUTS_DIR / "anomaly_results.csv"
EVENT_BACKTEST_FILE = OUTPUTS_DIR / "event_backtest_summary.csv"

# --- Composite-score weights (sum = 1.0) ---
RISK_WEIGHTS = {
    "supplier_dependency": 0.20,
    "market_concentration": 0.15,
    "import_anomaly": 0.20,
    "price_stress": 0.15,
    "logistics_risk": 0.10,
    "shipping_stress": 0.20,
}

# --- Risk-level thresholds (empirically calibrated) ---
RISK_LEVELS = [
    (0, 25, "Low"),
    (25, 40, "Medium"),
    (40, 60, "High"),
    (60, 101, "Critical"),
]

# --- Materiality filter ---
MATERIAL_SHARE_THRESHOLD = 0.01
MATERIAL_VALUE_THRESHOLD_USD = 1_000_000

# --- Anomaly thresholds ---
Z_ABS_THRESHOLD = 2.5
ISO_CONTAMINATION = 0.05

# --- Backtest windows ---
EVENT_WINDOW_DAYS_BEFORE = 90
EVENT_WINDOW_DAYS_AFTER = 180

# --- Products covered ---
PRODUCTS = ["Rice", "Wheat", "Barley", "Maize", "Sugar"]
DEFAULT_PRODUCT = "Rice"
PRODUCT_HS = {"Rice": 1006, "Wheat": 1001, "Barley": 1003, "Maize": 1005, "Sugar": 1701}

# --- Event-type classification mapping ---
EVENT_TYPE_CLASS = {
    "Export restriction": "Product-specific event",
    "Shipping disruption": "Maritime route event",
    "Chokepoint disruption": "Maritime route event",
    "Chokepoint tension": "Maritime route event",
    "De-escalation / recovery marker": "Maritime route event",
    "Global logistics disruption": "Global shock",
}

# Which products are affected by each event class (used for the product-aware event filter)
EVENT_PRODUCT_MAPPING = {
    "India non-basmati rice export ban": {"Rice"},
    "Russia-Ukraine war / Black Sea grain disruption": {"Wheat", "Barley", "Maize"},
    "Sugar price stress": {"Sugar"},
    # Maritime/global events default to ALL products
}

# --- Aggregate label for "World" reporter ---
AGGREGATE_SUPPLIER_NAMES = {"World", "WORLD", "world", "Total", "TOTAL"}

COUNTRY_RENAMES = {
    "Viet Nam": "Vietnam",
    "United States of America": "United States",
    "USA": "United States",
    "US": "United States",
    "Russian Federation": "Russia",
    "Iran (Islamic Republic of)": "Iran",
    "Korea, Rep.": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "United Arab Emirates": "UAE",
    "United Kingdom": "UK",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
}
