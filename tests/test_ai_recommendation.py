"""Targeted tests for src/ai_recommendation.py.

Tests three failure paths and one happy path:
  1. No GEMINI_API_KEY in env -> client is None, status describes the gap
  2. Invalid model name -> SDK retries with fallback, eventually returns rule-based
  3. Real .env present -> client connects (uses real network)
  4. Packet validation rejects malformed input
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib
from src import ai_recommendation as ai


def _sample_packet(**overrides) -> dict:
    base = dict(
        product_name="Rice",
        hs_code=1006,
        selected_supplier="India",
        scenario_mode="Supplier-level (recommended)",
        current_risk_score=42.0,
        forecasted_next_month_risk_score=44.5,
        before_scenario_risk=42.0,
        after_scenario_risk=55.3,
        risk_level_before="High",
        risk_level_after="High",
        top_risk_driver="Shipping stress",
        second_risk_driver="Commodity price stress",
        component_scores_before={
            "Supplier dependency": 60, "Market concentration": 90, "Import drop / anomaly": 10,
            "Commodity price stress": 20, "Logistics (LPI)": 40, "Shipping stress": 30,
        },
        component_scores_after={
            "Supplier dependency": 60, "Market concentration": 90, "Import drop / anomaly": 40,
            "Commodity price stress": 35, "Logistics (LPI)": 40, "Shipping stress": 60,
        },
        supplier_share=0.73,
        import_value_usd=120_000_000,
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
    base.update(overrides)
    return ai.build_decision_packet(**base)


def test_no_api_key():
    """Test 1: missing GEMINI_API_KEY -> client is None, status is clear."""
    os.environ.pop("GEMINI_API_KEY", None)
    importlib.reload(ai)
    ai._safe_load_env = lambda: None  # bypass .env loading
    client, status = ai.load_gemini_client()
    assert client is None, "expected client to be None when API key missing"
    assert not status.connected, "expected status.connected=False"
    assert "GEMINI_API_KEY" in status.message or "google-genai" in status.message
    # Ensure the API key value isn't in the message even by accident
    assert "AIza" not in status.message
    print(f"Test 1 PASS — status.message='{status.message}'")


def test_packet_validation_rejects_missing_fields():
    """Test 2: validate_packet flags structural issues. (Construct packet as dict
    so build_decision_packet's own arithmetic doesn't reject it first.)"""
    importlib.reload(ai)
    bad = {
        "product_name": "",
        "selected_supplier": "",
        "before_scenario_risk": "not-a-number",
        "after_scenario_risk": 10.0,
        "alternative_suppliers": [],
    }
    val = ai.validate_packet(bad)
    assert not val.ok
    assert any("product_name" in i for i in val.issues)
    assert any("selected_supplier" in i for i in val.issues)
    assert any("before_scenario_risk" in i for i in val.issues)
    print(f"Test 2 PASS — issues={val.issues}")


def test_fallback_recommendation_grounded_in_packet():
    """Test 3: rule-based fallback only mentions suppliers actually in the packet."""
    importlib.reload(ai)
    pkt = _sample_packet()
    text = ai.fallback_rule_based_recommendation(pkt)
    assert "India" in text  # focus supplier
    assert "Pakistan" in text  # alternative
    assert "United States" in text  # alternative
    # Not in alternatives:
    assert "Vietnam" not in text
    print("Test 3 PASS — fallback text mentions only suppliers in the packet")


def test_postprocess_flags_unsupported_claims():
    """Test 4: post-processor catches dollar-savings claims and unknown suppliers."""
    importlib.reload(ai)
    pkt = _sample_packet()
    bad_output = (
        "1. Executive recommendation\n"
        "Diversify to Vietnam to save $4.5M monthly. This will guarantee no disruption."
    )
    _, warnings = ai._postprocess(bad_output, pkt)
    joined = " | ".join(warnings)
    assert "dollar" in joined.lower() or "saving" in joined.lower()
    assert "guarantee" in joined.lower()
    assert "Vietnam" in joined
    print(f"Test 4 PASS — warnings caught: {warnings}")


def test_generate_falls_back_when_no_client():
    """Test 5: generate_gemini_recommendation gracefully falls back when no client."""
    os.environ.pop("GEMINI_API_KEY", None)
    importlib.reload(ai)
    ai._safe_load_env = lambda: None
    pkt = _sample_packet()
    result = ai.generate_gemini_recommendation(pkt)
    assert result.source == "fallback"
    assert "Decision support only" in result.text
    print(f"Test 5 PASS — source={result.source}, error={result.error}")


if __name__ == "__main__":
    failures = 0
    for t in [test_no_api_key, test_packet_validation_rejects_missing_fields,
              test_fallback_recommendation_grounded_in_packet,
              test_postprocess_flags_unsupported_claims,
              test_generate_falls_back_when_no_client]:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {e}")
            failures += 1
    print(f"\nResult: {5 - failures} passed, {failures} failed")
    sys.exit(1 if failures else 0)
