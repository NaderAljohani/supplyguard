"""Optional AI-enhanced recommendation layer for the Scenario Simulator.

Design rules (must not be violated):
  * Gemini never calculates the risk score, forecast, anomaly score, or scenario result.
  * Gemini only converts the existing structured analytics output into a clearer
    natural-language recommendation.
  * If the SDK isn't installed, the env vars aren't set, or the API call fails,
    callers get the deterministic rule-based recommendation instead.
  * The API key is never logged, printed, displayed, or returned to the caller.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

# python-dotenv is optional — if it's missing we still try env vars from the OS.
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False

# google-genai is optional — if it's missing the module reports a clean status.
try:
    from google import genai
    _HAS_SDK = True
except Exception:
    genai = None  # type: ignore[assignment]
    _HAS_SDK = False


# ============================================================
# Client
# ============================================================

@dataclass
class GeminiStatus:
    """What the dashboard needs to render the status indicator."""
    connected: bool
    message: str
    model: str = ""


def _safe_load_env() -> None:
    """Load .env from project root if python-dotenv is available."""
    if not _HAS_DOTENV:
        return
    try:
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        env_path = root / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except Exception:
        # Never fail the app because of env loading.
        pass


def load_gemini_client() -> tuple[Any | None, GeminiStatus]:
    """Return (client, status). client is None if anything is missing/broken.

    Status messages are safe to display in the UI; they never contain the API key.
    """
    _safe_load_env()
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    if not _HAS_SDK:
        return None, GeminiStatus(False, "google-genai package not installed", model)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, GeminiStatus(False, "GEMINI_API_KEY not set in environment", model)

    try:
        client = genai.Client(api_key=api_key)
        return client, GeminiStatus(True, "Connected", model)
    except Exception as e:
        # Strip the api_key out of any error message just in case.
        safe = re.sub(r"AIza[\w-]+", "<redacted>", str(e))
        return None, GeminiStatus(False, f"Client init failed: {safe[:120]}", model)


# ============================================================
# Decision packet
# ============================================================

def _round(x, n=2):
    if x is None:
        return None
    try:
        return round(float(x), n)
    except Exception:
        return None


def build_decision_packet(
    *,
    product_name: str,
    hs_code: int | str,
    selected_supplier: str,
    scenario_mode: str,
    current_risk_score: float | None,
    forecasted_next_month_risk_score: float | None,
    before_scenario_risk: float,
    after_scenario_risk: float,
    risk_level_before: str,
    risk_level_after: str,
    top_risk_driver: str | None,
    second_risk_driver: str | None,
    component_scores_before: dict,
    component_scores_after: dict,
    supplier_share: float | None,
    import_value_usd: float | None,
    alternative_suppliers: list[dict],
    scenario_parameters: dict,
    model_forecast_note: str | None = None,
) -> dict:
    """Compact, model-agnostic dict that summarises one scenario for the LLM."""
    delta_risk = after_scenario_risk - before_scenario_risk
    component_changes = {
        k: _round(component_scores_after.get(k, 0) - component_scores_before.get(k, 0), 2)
        for k in component_scores_before
    }
    exposed_value = None
    if import_value_usd is not None and supplier_share is not None:
        try:
            exposed_value = float(import_value_usd)
        except Exception:
            exposed_value = None

    return {
        "product_name": product_name,
        "hs_code": hs_code,
        "selected_supplier": selected_supplier,
        "scenario_mode": scenario_mode,
        "current_risk_score": _round(current_risk_score),
        "forecasted_next_month_risk_score": _round(forecasted_next_month_risk_score),
        "before_scenario_risk": _round(before_scenario_risk),
        "after_scenario_risk": _round(after_scenario_risk),
        "delta_risk": _round(delta_risk),
        "risk_level_before": risk_level_before,
        "risk_level_after": risk_level_after,
        "top_risk_driver": top_risk_driver,
        "second_risk_driver": second_risk_driver,
        "component_scores_before": {k: _round(v) for k, v in component_scores_before.items()},
        "component_scores_after": {k: _round(v) for k, v in component_scores_after.items()},
        "component_changes": component_changes,
        "supplier_share": _round(supplier_share, 4),
        "import_value_usd": _round(import_value_usd, 0),
        "exposed_monthly_import_value": _round(exposed_value, 0),
        "alternative_suppliers": [
            {
                "supplier_name": a.get("supplier_name"),
                "current_share": _round(a.get("current_share"), 4),
                "latest_import_value_usd": _round(a.get("latest_import_value_usd"), 0),
                "current_risk_score": _round(a.get("current_risk_score")),
                "risk_level": a.get("risk_level"),
            }
            for a in (alternative_suppliers or [])
        ],
        "maritime_shipping_stress": _round(component_scores_after.get("Shipping stress")),
        "commodity_price_stress": _round(component_scores_after.get("Commodity price stress")),
        "scenario_parameters": {
            "import_reduction_pct": scenario_parameters.get("import_reduction_pct"),
            "price_shock_pct": scenario_parameters.get("price_shock_pct"),
            "shipping_stress_shock_points": scenario_parameters.get("shipping_stress_shock_points"),
            "reassignment_pct": scenario_parameters.get("reassignment_pct"),
        },
        "model_forecast_note": model_forecast_note,
    }


# ============================================================
# Packet validation
# ============================================================

@dataclass
class PacketValidation:
    ok: bool
    issues: list[str] = field(default_factory=list)


def validate_packet(packet: dict) -> PacketValidation:
    issues = []
    if not packet.get("product_name"):
        issues.append("product_name missing")
    if not packet.get("selected_supplier"):
        issues.append("selected_supplier missing")
    for field_name in ("before_scenario_risk", "after_scenario_risk"):
        v = packet.get(field_name)
        if v is None or not isinstance(v, (int, float)):
            issues.append(f"{field_name} is not numeric")
    if not packet.get("alternative_suppliers"):
        # Not blocking — just a note for the prompt.
        pass
    return PacketValidation(ok=(len(issues) == 0), issues=issues)


# ============================================================
# Prompt construction
# ============================================================

_PROMPT_PREAMBLE = """You are a senior food supply-chain risk advisor for Saudi Arabia.

You will receive a structured DECISION PACKET produced by SupplyGuard Saudi.
The packet already contains the deterministic risk-score calculations, the
scenario before/after values, the component-level drivers, and the list of
alternative suppliers. Your job is to write a clear, professional
recommendation grounded ONLY in that packet.

STRICT RULES:
- Use only the provided decision packet. Do not invent suppliers, values,
  events, causes, savings, prices, or external facts. If information is
  missing, say it is not available in the current packet.
- Do not change or recompute any numerical risk score, forecast, anomaly
  score, or scenario value. SupplyGuard already computed those.
- Do not claim guaranteed disruption. The score is an early-warning index,
  not a binary disruption prediction.
- Mention alternative suppliers ONLY if they appear in alternative_suppliers.
  If alternative_suppliers is empty, state that no alternatives are listed.
- Keep tone professional, concise, and operational.

OUTPUT FORMAT (use these exact section headings, in this order):

1. Executive recommendation
(2-3 sentences)

2. Why risk changed
(1 short paragraph naming the main component drivers from the packet)

3. Immediate operational actions
- bullet 1
- bullet 2
- bullet 3

4. Procurement / diversification actions
- bullet 1
- bullet 2
- bullet 3
(Use only suppliers listed in alternative_suppliers. If none provided, say so.)

5. Monitoring signals
- bullet 1
- bullet 2
- bullet 3

6. Caveat
(One sentence stating this is decision support, not a guaranteed disruption prediction.)

DECISION PACKET (JSON):
"""


def _build_prompt(packet: dict) -> str:
    return _PROMPT_PREAMBLE + json.dumps(packet, indent=2, default=str)


# ============================================================
# Generation
# ============================================================

@dataclass
class AIRecommendationResult:
    text: str
    source: str            # "gemini" | "fallback"
    model_used: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def _postprocess(text: str, packet: dict) -> tuple[str, list[str]]:
    """Detect suspicious claims (exact $ savings, suppliers not in packet, guarantees).
    Returns (text, warnings). Does not block output."""
    warnings_list = []
    suppliers_allowed = {(packet.get("selected_supplier") or "").lower()}
    for alt in packet.get("alternative_suppliers", []):
        if alt.get("supplier_name"):
            suppliers_allowed.add(alt["supplier_name"].lower())

    # 1. Exact dollar savings claims
    if re.search(r"(save|saving|savings)[^.\n]{0,60}\$\s?\d", text, re.IGNORECASE):
        warnings_list.append("Output may claim an exact dollar saving that is not in the packet.")

    # 2. Guarantee language (covers "guarantee", "guaranteed", "guaranteeing", etc.)
    if re.search(
        r"\b(guarantee\w*|certain to|will definitely|prevents disruption|no risk of disruption)\b",
        text, re.IGNORECASE,
    ):
        warnings_list.append("Output uses guarantee-style wording inconsistent with an early-warning index.")

    # 3. Supplier names not in packet (heuristic — look for "Country names" patterns we know are common
    #    but missing from the alternative-supplier list)
    common_suppliers = ["India", "Pakistan", "Thailand", "Vietnam", "United States", "Russia",
                        "Ukraine", "Brazil", "Australia", "Argentina", "Germany", "France",
                        "China", "Turkey", "UAE", "Egypt"]
    mentioned_unknown = []
    for s in common_suppliers:
        if s.lower() not in suppliers_allowed and re.search(rf"\b{re.escape(s)}\b", text):
            mentioned_unknown.append(s)
    if mentioned_unknown:
        warnings_list.append(
            f"Output mentions supplier(s) not in the alternatives list: {', '.join(mentioned_unknown)}."
        )

    return text, warnings_list


def generate_gemini_recommendation(packet: dict, client: Any | None = None) -> AIRecommendationResult:
    """Call Gemini with a strict prompt. Falls back to the rule-based recommendation
    on any failure. Never raises."""
    val = validate_packet(packet)
    if not val.ok:
        return AIRecommendationResult(
            text=fallback_rule_based_recommendation(packet),
            source="fallback", warnings=val.issues,
            error="packet validation failed",
        )

    if client is None:
        client, status = load_gemini_client()
    else:
        status = GeminiStatus(True, "Connected", os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))

    if client is None:
        return AIRecommendationResult(
            text=fallback_rule_based_recommendation(packet),
            source="fallback", error=status.message,
        )

    prompt = _build_prompt(packet)
    primary_model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    fallback_model = "gemini-2.5-flash"

    # Try low temperature if the SDK accepts it via config; ignore failures of the
    # config arg (different SDK versions accept different shapes).
    def _call(model_name: str):
        try:
            from google.genai import types as _types  # type: ignore
            cfg = _types.GenerateContentConfig(temperature=0.2)
            return client.models.generate_content(
                model=model_name, contents=prompt, config=cfg,
            )
        except Exception:
            return client.models.generate_content(model=model_name, contents=prompt)

    last_error = None
    for model_name in (primary_model, fallback_model):
        if not model_name:
            continue
        try:
            response = _call(model_name)
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                last_error = f"empty response from {model_name}"
                continue
            text, warns = _postprocess(text, packet)
            return AIRecommendationResult(
                text=text, source="gemini", model_used=model_name, warnings=warns,
            )
        except Exception as e:
            safe = re.sub(r"AIza[\w-]+", "<redacted>", str(e))[:200]
            last_error = f"{model_name}: {safe}"
            continue

    return AIRecommendationResult(
        text=fallback_rule_based_recommendation(packet),
        source="fallback", error=last_error,
    )


# ============================================================
# Rule-based fallback (mirrors the dashboard's existing logic)
# ============================================================

_DRIVER_TEMPLATES = {
    "Supplier dependency": (
        "Reduce dependence on {supplier} by reallocating part of procurement to lower-risk "
        "alternative suppliers; target a top-supplier share below 50%."),
    "Market concentration": (
        "Diversify the top-3 supplier mix — current concentration leaves the {product} supply "
        "chain exposed to a single-country shock."),
    "Import drop / anomaly": (
        "Investigate recent {supplier} shipment performance, verify upcoming shipment "
        "reliability, and prepare contingency volumes from secondary suppliers."),
    "Commodity price stress": (
        "Review procurement timing, contract exposure (spot vs forward), and inventory "
        "strategy due to {product_lower} commodity-price pressure."),
    "Logistics (LPI)": (
        "Reassess supplier logistics-performance assumptions for {supplier}; consider "
        "shifting share to suppliers with stronger LPI."),
    "Shipping stress": (
        "Monitor Red Sea / Bab el-Mandeb route conditions; pre-position inventory buffers "
        "or arrange Cape-of-Good-Hope routing contingency."),
}


def fallback_rule_based_recommendation(packet: dict) -> str:
    """Return the same deterministic recommendation the dashboard already shows."""
    delta = packet.get("delta_risk") or 0.0
    supplier = packet.get("selected_supplier", "the focus supplier")
    product = packet.get("product_name", "this product")
    product_lower = product.lower()

    if delta > 5:
        intro = "Activate contingency sourcing."
    elif delta > 0.5:
        intro = "Monitor the situation closely and prepare secondary options."
    elif delta < -3:
        intro = "Diversification improves resilience under this scenario."
    else:
        intro = "Scenario impact is marginal under current assumptions."

    # Choose driver by largest |change| (matches the dashboard's main_driver_col).
    changes = packet.get("component_changes", {})
    if changes:
        main_driver = max(changes, key=lambda k: abs(changes[k] or 0))
    else:
        main_driver = packet.get("top_risk_driver", "")

    template = _DRIVER_TEMPLATES.get(main_driver,
        "Re-evaluate the dominant driver in the component breakdown.")
    body = template.format(supplier=supplier, product=product, product_lower=product_lower)

    alts = packet.get("alternative_suppliers") or []
    if alts:
        alt_names = ", ".join(a["supplier_name"] for a in alts if a.get("supplier_name"))
        alt_line = f"\n\n**Alternative suppliers available:** {alt_names}."
    else:
        alt_line = "\n\n_No alternative suppliers listed in the current packet._"

    return f"{intro} {body}{alt_line}\n\n_Decision support only — not a guaranteed disruption prediction._"
