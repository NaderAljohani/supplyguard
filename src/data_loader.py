"""Load raw CSVs with light validation. Robust to column-name drift."""
from __future__ import annotations
import pandas as pd
from . import config


def _read_csv(path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    print(f"  loaded {path.name}: shape={df.shape}")
    return df


def load_main() -> pd.DataFrame:
    df = _read_csv(config.MAIN_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_portwatch_wide() -> pd.DataFrame:
    df = _read_csv(config.PORTWATCH_WIDE_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_portwatch_long() -> pd.DataFrame:
    df = _read_csv(config.PORTWATCH_LONG_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_events() -> pd.DataFrame:
    df = _read_csv(config.EVENTS_FILE)
    df = df.dropna(how="all")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_coverage() -> pd.DataFrame:
    if not config.COVERAGE_FILE.exists():
        return pd.DataFrame()
    return _read_csv(config.COVERAGE_FILE)


def load_annual_top() -> pd.DataFrame:
    if not config.ANNUAL_TOP_FILE.exists():
        return pd.DataFrame()
    return _read_csv(config.ANNUAL_TOP_FILE)


def load_prices() -> pd.DataFrame:
    if not config.PRICES_FILE.exists():
        return pd.DataFrame()
    df = _read_csv(config.PRICES_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_latest_material() -> pd.DataFrame:
    if not config.LATEST_MATERIAL_FILE.exists():
        return pd.DataFrame()
    df = _read_csv(config.LATEST_MATERIAL_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_methodology() -> pd.DataFrame:
    if not config.METHODOLOGY_FILE.exists():
        return pd.DataFrame()
    return _read_csv(config.METHODOLOGY_FILE).dropna(how="all")


def describe_frame(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} ---")
    print(f"shape: {df.shape}")
    print(f"columns ({len(df.columns)}): {list(df.columns)[:12]}{'...' if len(df.columns) > 12 else ''}")
    if "date" in df.columns:
        print(f"date range: {df['date'].min()} -> {df['date'].max()}")
    if "product_name" in df.columns:
        print(f"products: {sorted(df['product_name'].dropna().unique())}")
    miss = df.isna().mean().mul(100).round(1).sort_values(ascending=False)
    top_miss = miss[miss > 0].head(5)
    if not top_miss.empty:
        print("top missing %:")
        for k, v in top_miss.items():
            print(f"  {k}: {v}%")
