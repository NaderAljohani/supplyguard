"""Per-product next-month risk-score forecasting.

Three artifacts go to data/outputs/ so the Forecasting dashboard page can render
without re-training:
  - forecast_model_metrics.csv     (MAE/RMSE/R² per product per model)
  - next_month_risk_forecast.csv   (per-supplier predicted next-month risk)
  - forecast_feature_importance.csv (best-model feature importances per product)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from . import config

warnings.filterwarnings("ignore")


# ============================================================
# Lag / rolling / seasonality features
# ============================================================

LAG_SPEC = {
    "supplyguard_risk_score_0_100":    ("risk_score", (1, 3)),
    "supplier_share_value":            ("supplier_share", (1,)),
    "import_value_usd":                ("import_value", (1,)),
    "import_value_usd_yoy_pct_change": ("import_value_yoy_change", (1,)),
    "hhi_value":                       ("hhi", (1,)),
    "top3_supplier_share_value":       ("top3_supplier_share", (1,)),
    "price_stress_score_0_100":        ("price_stress", (1,)),
    "shipping_stress_score_0_100":     ("shipping_stress", (1,)),
    "import_anomaly_score_0_100":      ("import_anomaly_risk", (1,)),
}


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["product_name", "supplier_country", "date"])
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    keys = ["product_name", "supplier_country"]
    if df.duplicated(keys + ["date"]).any():
        raise ValueError("Forecast input must contain one row per product/supplier/month")
    monthly = []
    for identity, series in df.groupby(keys):
        series = series.set_index("date")
        original_dates = series.index
        series = series.reindex(pd.date_range(original_dates.min(), original_dates.max(), freq="MS"))
        series.index.name = "date"
        series["_observed"] = series.index.isin(original_dates)
        for key, value in zip(keys, identity):
            series[key] = value
        monthly.append(series.reset_index())
    if not monthly:
        return df.assign(next_month_supplyguard_risk_score_0_100=np.nan)
    df = pd.concat(monthly, ignore_index=True)
    df["month"] = df["date"].dt.month
    df["target_date"] = df["date"] + pd.offsets.MonthBegin(1)
    g = df.groupby(["product_name", "supplier_country"], group_keys=False)

    # Standard lags
    for src_col, (alias, lags) in LAG_SPEC.items():
        if src_col not in df.columns:
            continue
        for k in lags:
            df[f"{alias}_lag_{k}"] = g[src_col].shift(k)

    # Rolling risk averages (shifted by 1 so they use only past info)
    if "supplyguard_risk_score_0_100" in df.columns:
        df["risk_score_rolling_3m_mean"] = (
            g["supplyguard_risk_score_0_100"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )
        df["risk_score_rolling_6m_mean"] = (
            g["supplyguard_risk_score_0_100"]
            .transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
        )
        # Forecast target
        df["next_month_supplyguard_risk_score_0_100"] = (
            g["supplyguard_risk_score_0_100"].shift(-1)
        )

    # Static / current-period features that don't need lagging
    # logistics_risk is slowly-changing (LPI updated every few years)
    # month is a seasonality feature (current row's month is fine)
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df.loc[df["_observed"]].drop(columns="_observed").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All forecasting input columns (lags + current statics + seasonality).

    Every column included here MUST be known at or before month T when the
    target is month T+1. See audit_features() for the leakage classification.
    """
    feats = []
    for c in df.columns:
        if c.endswith(("_lag_1", "_lag_3")) or c.startswith("risk_score_rolling"):
            feats.append(c)
    for c in ("logistics_risk_score_0_100", "month_sin", "month_cos"):
        if c in df.columns:
            feats.append(c)
    return feats


# ============================================================
# Leakage audit
# ============================================================

# Columns that, if used as same-month features for a next-month target, would
# leak target information through the composite formula (composite = Σ component_i × weight_i).
_COMPONENT_COLUMNS_SAME_MONTH = {
    "supplyguard_risk_score_0_100",
    "supplier_dependency_score_0_100",
    "concentration_score_0_100",
    "import_anomaly_score_0_100",
    "price_stress_score_0_100",
    "logistics_risk_score_0_100",     # exception: see _AS_OF_OK_STATICS
    "shipping_stress_score_0_100",
    "hhi_value",
    "top3_supplier_share_value",
    "supplier_share_value",
}

# Slowly-changing / calendar features that are legitimately known as-of T.
_AS_OF_OK_STATICS = {
    "logistics_risk_score_0_100": "LPI is a slowly-changing supplier-country attribute, refreshed every few years; known as-of T.",
    "month_sin":                  "Calendar seasonality — derived from the row's month index; known before prediction.",
    "month_cos":                  "Calendar seasonality — derived from the row's month index; known before prediction.",
}


def audit_features(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Classify every column of df as OK / WARNING / REMOVE for use as a
    feature when the target is `next_month_supplyguard_risk_score_0_100`.

    Rules
    -----
    * Columns ending in `_lag_1` or `_lag_3`: OK — lagged by groupby.shift().
    * Columns starting with `risk_score_rolling`: OK — built as shift(1).rolling(...).
    * Whitelisted as-of statics (logistics_risk_score_0_100, month_sin/cos): OK.
    * Same-month component columns (price/shipping/anomaly/share/etc.) and the
      composite itself: REMOVE — would leak into the next-month target through
      the composite formula.
    * Identifier columns (product_name, supplier_country, date): WARNING (not
      features, but they must not be fed into the model).
    * Everything else: WARNING — review before use.

    The output is saved to data/outputs/forecast_leakage_audit.csv.
    """
    target = "next_month_supplyguard_risk_score_0_100"

    feats = set(feature_columns(df))
    rows = []
    for col in df.columns:
        if col == target:
            continue

        # 1. Explicitly lagged columns
        if col.endswith(("_lag_1", "_lag_3")):
            rows.append({
                "target": target, "feature": col,
                "in_model": col in feats,
                "lagged_or_as_of": "lag",
                "leakage_risk": "OK",
                "reason": "Lagged via groupby.shift(); month-T value uses month-T-1 (or T-3) data.",
            })
            continue

        # 2. Rolling features (already shift(1) before rolling)
        if col.startswith("risk_score_rolling"):
            rows.append({
                "target": target, "feature": col,
                "in_model": col in feats,
                "lagged_or_as_of": "rolling-shifted",
                "leakage_risk": "OK",
                "reason": "Built as series.shift(1).rolling(...).mean() so window only sees past values.",
            })
            continue

        # 3. Whitelisted as-of statics
        if col in _AS_OF_OK_STATICS:
            rows.append({
                "target": target, "feature": col,
                "in_model": col in feats,
                "lagged_or_as_of": "as-of-T",
                "leakage_risk": "OK",
                "reason": _AS_OF_OK_STATICS[col],
            })
            continue

        # 4. Same-month component columns and the composite itself
        if col in _COMPONENT_COLUMNS_SAME_MONTH:
            rows.append({
                "target": target, "feature": col,
                "in_model": col in feats,
                "lagged_or_as_of": "current-month",
                "leakage_risk": "REMOVE" if col in feats else "OK",
                "reason": (
                    "Same-month composite or component — directly leaks into next-month target through the composite formula."
                    if col in feats
                    else "Same-month component, but not used as a model feature (only its _lag_1 / _lag_3 is)."
                ),
            })
            continue

        # 5. Identifier / housekeeping columns
        if col in {"product_name", "supplier_country", "date", "month", "year",
                   "risk_level", "top_risk_driver", "second_risk_driver",
                   "risk_explanation_text", "iso_anomaly_score", "iso_anomaly_flag",
                   "anomaly_score", "anomaly_flag", "anomaly_reason",
                   "anomaly_z_flag", "event_window_flag", "disruption_event_name"}:
            rows.append({
                "target": target, "feature": col,
                "in_model": col in feats,
                "lagged_or_as_of": "identifier/derived",
                "leakage_risk": "OK" if col not in feats else "REMOVE",
                "reason": (
                    "Identifier or derived column — not used as a model feature."
                    if col not in feats
                    else "Identifier or post-hoc derived column should not be a model feature."
                ),
            })
            continue

        # 6. Default — review
        rows.append({
            "target": target, "feature": col,
            "in_model": col in feats,
            "lagged_or_as_of": "unknown",
            "leakage_risk": "WARNING" if col in feats else "OK",
            "reason": (
                "Not on the allow-list — review whether month-T value is known before predicting month T+1."
                if col in feats
                else "Not used as a model feature."
            ),
        })

    audit_df = pd.DataFrame(rows).sort_values(
        ["leakage_risk", "in_model", "feature"],
        key=lambda s: s.map({"REMOVE": 0, "WARNING": 1, "OK": 2}) if s.name == "leakage_risk" else s,
    ).reset_index(drop=True)

    if save:
        out = config.OUTPUTS_DIR / "forecast_leakage_audit.csv"
        audit_df.to_csv(out, index=False)
        print(f"[forecasting] leakage audit -> {out}")
        # Console summary
        counts = audit_df["leakage_risk"].value_counts().to_dict()
        print(f"  leakage_risk counts: {counts}")
        bad = audit_df[(audit_df["leakage_risk"] == "REMOVE") & (audit_df["in_model"])]
        if not bad.empty:
            print("  !! REMOVE-classified features are present in the model:")
            for _, r in bad.iterrows():
                print(f"     - {r['feature']}: {r['reason']}")
        else:
            print("  no leakage-risk features in the active model feature set.")

    return audit_df


# ============================================================
# Training / evaluation
# ============================================================

def _try_boost():
    try:
        from xgboost import XGBRegressor
        return "XGBoost", XGBRegressor(
            n_estimators=300, learning_rate=0.08, max_depth=5,
            random_state=42, n_jobs=-1, verbosity=0,
        )
    except Exception:
        return "HistGradientBoosting", HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, random_state=42,
        )


def _score(y_true, y_pred) -> dict:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "R2": round(r2_score(y_true, y_pred), 3),
    }


def _train_models(Xtr, ytr):
    """Return dict of name → fitted model. Linear is wrapped with a scaler."""
    models = {}

    # Linear (scaled)
    try:
        scaler = StandardScaler().fit(Xtr)
        lr = LinearRegression().fit(scaler.transform(Xtr), ytr)
        models["LinearRegression"] = ("linear_scaled", scaler, lr)
    except Exception:
        pass

    try:
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
        ).fit(Xtr, ytr)
        models["RandomForest"] = ("plain", None, rf)
    except Exception:
        pass

    name, boost = _try_boost()
    try:
        boost.fit(Xtr, ytr)
        models[name] = ("plain", None, boost)
    except Exception:
        pass

    return models


def _predict(model_tuple, X):
    kind, scaler, model = model_tuple
    if kind == "linear_scaled":
        return model.predict(scaler.transform(X))
    return model.predict(X)


BASELINE = "Baseline (persistence)"


def select_model(validation_errors: dict[str, float]) -> str:
    """Choose on validation only; insertion order preserves baseline on ties."""
    return min(validation_errors, key=validation_errors.get)


def _refit(model_tuple, X, y):
    kind, _, model = model_tuple
    fitted = clone(model)
    if kind == "linear_scaled":
        scaler = StandardScaler().fit(X)
        return kind, scaler, fitted.fit(scaler.transform(X), y)
    return kind, None, fitted.fit(X, y)


def _predict_frame(name, model_tuple, frame, features):
    if name == BASELINE:
        return frame["supplyguard_risk_score_0_100"].to_numpy(dtype=float)
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    return np.clip(_predict(model_tuple, matrix), 0, 100)


def evaluate_models(df: pd.DataFrame) -> dict:
    """Select using an earlier validation window; report an untouched later test.

    Splits use target months. Operational forecasts are separately refitted on
    all labelled history after the held-out evaluation has been recorded.
    """
    df = build_lag_features(df)
    audit = audit_features(df, save=True)
    removed = set(audit.loc[audit["leakage_risk"] == "REMOVE", "feature"])
    features = [f for f in feature_columns(df) if f not in removed]
    metrics, forecasts, importances, predictions, splits = [], [], [], [], []
    results = {}
    for product, series in df.groupby("product_name"):
        train_validation, test = _time_split(series, 6)
        train, validation = _time_split(train_validation, 3)
        if train.empty or validation.empty or test.empty:
            print(f"[{product}] insufficient history for train/validation/test; skipped")
            continue

        def matrix(frame):
            return frame[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()

        target = "next_month_supplyguard_risk_score_0_100"
        models = {BASELINE: None, **_train_models(matrix(train), train[target].to_numpy())}
        validation_errors = {
            name: float(mean_absolute_error(validation[target], _predict_frame(name, model, validation, features)))
            for name, model in models.items()
        }
        selected = select_model(validation_errors)
        fitted = {
            name: None if name == BASELINE else _refit(model, matrix(train_validation), train_validation[target].to_numpy())
            for name, model in models.items()
        }
        baseline_predictions = _predict_frame(BASELINE, None, test, features)
        baseline_score = _score(test[target].to_numpy(), baseline_predictions)
        selected_predictions = None
        for name, model in fitted.items():
            predicted = _predict_frame(name, model, test, features)
            score = _score(test[target].to_numpy(), predicted)
            metrics.append({
                "product": product, "model": name, **score,
                "n_test": len(test), "n_test_months": test["target_date"].nunique(),
                "validation_MAE": round(validation_errors[name], 3),
                "selected_on_validation": name == selected,
                "baseline_MAE": baseline_score["MAE"],
                "mae_improvement_vs_baseline": round(baseline_score["MAE"] - score["MAE"], 3),
            })
            if name == selected:
                selected_predictions = predicted
        held_out = pd.DataFrame({
            "product": product, "date": test["target_date"].to_numpy(),
            "as_of_date": test["date"].to_numpy(),
            "supplier_country": test["supplier_country"].to_numpy(),
            "actual_next_month_risk": test[target].to_numpy(),
            "predicted_next_month_risk": selected_predictions,
            "baseline_predicted": baseline_predictions, "model": selected,
        })
        predictions.append(held_out)
        splits.append({
            "product": product, "selected_model": selected,
            "train_target_end": train["target_date"].max(),
            "validation_target_start": validation["target_date"].min(),
            "validation_target_end": validation["target_date"].max(),
            "test_target_start": test["target_date"].min(),
            "test_target_end": test["target_date"].max(),
            "selection_rule": "lowest validation MAE, including persistence",
        })
        # Estimate the next month only for suppliers observed in the current
        # product snapshot, avoiding forecasts from stale supplier series.
        labelled = series.dropna(subset=[target])
        latest = series.loc[series["date"] == series["date"].max()].copy()
        deployment_model = None if selected == BASELINE else _refit(
            models[selected], matrix(labelled), labelled[target].to_numpy()
        )
        latest_predictions = _predict_frame(selected, deployment_model, latest, features)
        for (_, row), estimate in zip(latest.iterrows(), latest_predictions):
            current = float(row["supplyguard_risk_score_0_100"])
            forecasts.append({
                "product": product, "supplier_country": row["supplier_country"],
                "as_of_date": row["date"].strftime("%Y-%m-01"),
                "forecast_for_date": row["target_date"].strftime("%Y-%m-01"),
                "current_risk_score": current,
                "forecast_next_month_risk_score": float(estimate),
                "delta_vs_current": float(estimate - current), "model": selected,
                "supplier_share_value": float(row.get("supplier_share_value", 0) or 0),
                "import_value_usd": float(row.get("import_value_usd", 0) or 0),
            })
        importance = {"current_risk_score": 1.0} if selected == BASELINE else _feature_importance(
            selected, fitted[selected], features, matrix(validation), validation[target].to_numpy()
        )
        for feature, value in importance.items():
            importances.append({"product": product, "model": selected, "feature": feature, "importance": value})
        results[product] = {"best_name": selected, "test": held_out, "features": features}
        print(f"[{product}] selected on validation: {selected}; test rows: {len(test)}")
    for filename, rows in [
        ("forecast_model_metrics.csv", metrics), ("next_month_risk_forecast.csv", forecasts),
        ("forecast_feature_importance.csv", importances), ("forecast_split_summary.csv", splits),
    ]:
        pd.DataFrame(rows).to_csv(config.OUTPUTS_DIR / filename, index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(config.OUTPUTS_DIR / "forecast_actual_vs_predicted.csv", index=False)
    return results


def _time_split(df: pd.DataFrame, months_held_out: int):
    labelled = df.dropna(subset=["next_month_supplyguard_risk_score_0_100"]).copy()
    if labelled.empty:
        return labelled, labelled.copy()
    dates = sorted(labelled["target_date"].unique())
    if len(dates) <= months_held_out:
        return labelled, labelled.iloc[:0]
    cutoff = dates[-months_held_out]
    return labelled.loc[labelled["target_date"] < cutoff], labelled.loc[labelled["target_date"] >= cutoff]


def _feature_importance(name: str, mtup, feats: list[str], Xte, yte) -> dict[str, float]:
    """Return {feature: importance}. RF/boost use native importances;
    Linear uses |coef|; permutation is the universal fallback."""
    kind, scaler, model = mtup
    try:
        if hasattr(model, "feature_importances_") and not kind.startswith("linear"):
            imps = np.asarray(model.feature_importances_, dtype=float)
            return dict(zip(feats, imps / imps.sum() if imps.sum() > 0 else imps))
        if kind == "linear_scaled":
            coefs = np.abs(model.coef_)
            return dict(zip(feats, coefs / coefs.sum() if coefs.sum() > 0 else coefs))
    except Exception:
        pass

    # Permutation importance fallback
    try:
        if kind == "linear_scaled":
            X_use = scaler.transform(Xte)
        else:
            X_use = Xte
        r = permutation_importance(model, X_use, yte, n_repeats=5, random_state=42, n_jobs=-1)
        imps = np.maximum(r.importances_mean, 0)
        return dict(zip(feats, imps / imps.sum() if imps.sum() > 0 else imps))
    except Exception:
        return {f: 0.0 for f in feats}
