"""Smoke-test every Streamlit page across every product."""
from __future__ import annotations
import importlib
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

PAGES = [
    "📈  Executive Overview",
    "🏷️  Supplier Risk Analysis",
    "📊  Market Concentration",
    "🚢  Maritime Stress",
    "📜  Event Backtesting",
    "🚨  Anomaly Detection",
    "🔮  Forecasting",
    "🧪  Scenario Simulator",
    "📘  Methodology",
]
PRODUCTS = ["Rice", "Wheat", "Barley", "Maize", "Sugar"]


def run_one(page_value: str, product_value: str) -> tuple[bool, str]:
    orig_radio = st.radio
    orig_selectbox = st.selectbox
    orig_slider = st.slider
    orig_toggle = getattr(st, "toggle", None)
    orig_checkbox = st.checkbox
    orig_button = st.button
    orig_download = st.download_button

    def fake_radio(label, options, **kw):
        if isinstance(label, str) and ("Navigation" in label or "navigation" in label.lower()):
            return page_value
        if isinstance(label, str) and "Scenario mode" in label:
            return options[0]
        return options[0] if options else None

    def fake_selectbox(label, options, **kw):
        if isinstance(label, str) and "Product" in label:
            return product_value if product_value in options else (options[0] if options else None)
        return options[0] if options else None

    def fake_slider(label, *args, **kw):
        if "value" in kw and kw["value"] is not None:
            return kw["value"]
        if len(args) >= 3:
            return args[2]
        return 0

    def fake_toggle(label, *args, **kw):
        return kw.get("value", False)

    def fake_checkbox(label, *args, **kw):
        return kw.get("value", False)

    def fake_button(label, *args, **kw):
        return False  # never click presets in smoke test

    def fake_download(label, **kw):
        return False

    st.radio = fake_radio
    st.selectbox = fake_selectbox
    st.slider = fake_slider
    if orig_toggle is not None:
        st.toggle = fake_toggle
    st.checkbox = fake_checkbox
    st.button = fake_button
    st.download_button = fake_download

    try:
        if "app_module" in sys.modules:
            del sys.modules["app_module"]
        spec = importlib.util.spec_from_file_location("app_module", ROOT / "app" / "streamlit_app.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return True, "ok"
    except SystemExit as e:
        return True, f"st.stop() called: {e}"
    except Exception as e:
        import traceback
        return False, traceback.format_exc()
    finally:
        st.radio = orig_radio
        st.selectbox = orig_selectbox
        st.slider = orig_slider
        if orig_toggle is not None:
            st.toggle = orig_toggle
        st.checkbox = orig_checkbox
        st.button = orig_button
        st.download_button = orig_download


def main() -> int:
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)

    fail = 0
    print(f"Smoke testing {len(PAGES)} pages × {len(PRODUCTS)} products = {len(PAGES)*len(PRODUCTS)} runs")
    print("=" * 72)
    for prod in PRODUCTS:
        for p in PAGES:
            ok, msg = run_one(p, prod)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}]  {prod:<8}  {p}")
            if not ok:
                fail += 1
                print(msg)
    print("=" * 72)
    print(f"Result: {len(PAGES)*len(PRODUCTS) - fail} passed, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
