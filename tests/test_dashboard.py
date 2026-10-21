from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = ["Executive Overview", "Supplier Risk Analysis", "Market Concentration",
         "Maritime Stress", "Event Backtesting", "Anomaly Detection",
         "Forecasting", "Scenario Simulator", "Methodology"]


@pytest.mark.parametrize("product", ["Rice", "Wheat", "Barley", "Maize", "Sugar"])
@pytest.mark.parametrize("page", PAGES)
def test_dashboard_pages_render_for_each_product(product, page, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    product_widget = next(item for item in app.selectbox if item.label.startswith("Product"))
    navigation = next(item for item in app.radio if item.label == "Navigation")
    product_widget.set_value(product)
    navigation.set_value(next(option for option in navigation.options if page in option))
    app.run()
    assert not app.exception, [error.message for error in app.exception]
    assert not app.error, [error.value for error in app.error]
