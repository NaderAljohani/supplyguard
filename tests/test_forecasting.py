import numpy as np
import pandas as pd
from src.forecasting import build_lag_features, _time_split, select_model, BASELINE
from src.evaluation import backtest_event


def test_calendar_gaps_do_not_turn_next_observation_into_next_month():
    frame = pd.DataFrame({
        "product_name": ["Rice"] * 3, "supplier_country": ["India"] * 3,
        "date": pd.to_datetime(["2023-01-01", "2023-03-01", "2023-04-01"]),
        "supplyguard_risk_score_0_100": [10.0, 30.0, 40.0],
    })
    result = build_lag_features(frame).set_index("date")
    assert len(result) == 3
    assert pd.isna(result.loc["2023-01-01", "next_month_supplyguard_risk_score_0_100"])
    assert pd.isna(result.loc["2023-03-01", "risk_score_lag_1"])
    assert result.loc["2023-03-01", "next_month_supplyguard_risk_score_0_100"] == 40.0
    assert result.loc["2023-04-01", "risk_score_lag_3"] == 10.0


def test_target_months_are_disjoint_across_train_validation_and_test():
    frame = pd.DataFrame({
        "product_name": ["Rice"] * 30, "supplier_country": ["India"] * 30,
        "date": pd.date_range("2020-01-01", periods=30, freq="MS"),
        "supplyguard_risk_score_0_100": np.arange(30, dtype=float),
    })
    featured = build_lag_features(frame)
    earlier, test = _time_split(featured, 6)
    train, validation = _time_split(earlier, 3)
    assert train.target_date.max() < validation.target_date.min()
    assert validation.target_date.max() < test.target_date.min()
    assert test.target_date.nunique() == 6
    assert validation.target_date.nunique() == 3


def test_persistence_can_win_and_is_not_excluded_from_selection():
    assert select_model({BASELINE: 1.0, "RandomForest": 2.0}) == BASELINE
    assert select_model({BASELINE: 1.0, "LinearRegression": 1.0}) == BASELINE
    assert select_model({BASELINE: 2.0, "LinearRegression": 1.0}) == "LinearRegression"


def test_multi_product_event_keeps_both_affected_products():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2022-02-01"] * 3),
        "product_name": ["Wheat", "Maize", "Rice"],
        "supplier_country": ["Russia", "Ukraine", "India"],
        "supplyguard_risk_score_0_100": [20, 40, 90],
    })
    result = backtest_event(frame, pd.Timestamp("2022-02-24"), product="Wheat; Maize", country="Russia/Ukraine")
    assert len(result) == 1
    assert result.iloc[0].supplyguard_risk_score_0_100 == 30
