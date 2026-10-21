"""Live Gemini call — makes one real API call.
Only runs if GEMINI_API_KEY is present (read from .env via the module).
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ai_recommendation as ai


def main() -> int:
    client, status = ai.load_gemini_client()
    if client is None:
        print(f"SKIP: Gemini not configured — {status.message}")
        return 0
    print(f"Connected — model='{status.model}'")

    packet = ai.build_decision_packet(
        product_name="Rice", hs_code=1006, selected_supplier="India",
        scenario_mode="Supplier-level (recommended)",
        current_risk_score=42.0, forecasted_next_month_risk_score=44.5,
        before_scenario_risk=42.0, after_scenario_risk=55.3,
        risk_level_before="High", risk_level_after="High",
        top_risk_driver="Shipping stress", second_risk_driver="Commodity price stress",
        component_scores_before={"Supplier dependency": 60, "Market concentration": 90,
                                  "Import drop / anomaly": 10, "Commodity price stress": 20,
                                  "Logistics (LPI)": 40, "Shipping stress": 30},
        component_scores_after={"Supplier dependency": 60, "Market concentration": 90,
                                 "Import drop / anomaly": 40, "Commodity price stress": 35,
                                 "Logistics (LPI)": 40, "Shipping stress": 60},
        supplier_share=0.73, import_value_usd=120_000_000,
        alternative_suppliers=[
            {"supplier_name": "Pakistan", "current_share": 0.12,
             "latest_import_value_usd": 18_000_000, "current_risk_score": 35.0, "risk_level": "Medium"},
            {"supplier_name": "United States", "current_share": 0.08,
             "latest_import_value_usd": 14_000_000, "current_risk_score": 28.0, "risk_level": "Medium"},
        ],
        scenario_parameters={"import_reduction_pct": 30, "price_shock_pct": 15,
                             "shipping_stress_shock_points": 25, "reassignment_pct": 0},
        model_forecast_note="Best per-product model: LinearRegression",
    )

    result = ai.generate_gemini_recommendation(packet, client=client)
    print(f"\nsource={result.source}   model_used={result.model_used}   error={result.error}")
    print(f"warnings={result.warnings}")
    print("\n--- output (first 800 chars) ---")
    print((result.text or "")[:800])

    # Quick structure sanity check — does it contain the expected sections?
    expected = ["Executive recommendation", "Why risk changed",
                "Immediate operational", "Procurement", "Monitoring", "Caveat"]
    missing = [s for s in expected if s.lower() not in result.text.lower()]
    print(f"\nMissing sections: {missing if missing else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
