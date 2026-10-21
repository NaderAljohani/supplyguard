"""End-to-end SupplyGuard Saudi multi-product pipeline.

Produces (in data/processed/ and data/outputs/):
  - supplyguard_cleaned.csv
  - supplyguard_multiproduct_modeling_dataset.csv
  - supplyguard_latest_supplier_risk.csv
  - anomaly_results.csv
  - forecast_model_metrics.csv
  - forecast_actual_vs_predicted.csv
  - next_month_risk_forecast.csv
  - forecast_feature_importance.csv
  - data_quality_summary.csv
  - missingness_by_product.csv
  - sensitivity_analysis_summary.csv
  - sensitivity_analysis_detail.csv
  - event_backtest_summary.csv
  - figures/*.png
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from src import (
    config, data_loader, preprocessing, feature_engineering,
    risk_scoring, anomaly_detection, forecasting, evaluation,
    quality, sensitivity, visualization, eda,
)


def main() -> None:
    print("=" * 80)
    print("SupplyGuard Saudi — multi-product pipeline")
    print("=" * 80)

    print("\n[1/10] loading raw data")
    main_df = data_loader.load_main()
    events = data_loader.load_events()
    data_loader.describe_frame(main_df, "main multi-product features")
    data_loader.describe_frame(events, "events")

    print("\n[2/10] preprocessing")
    cleaned = preprocessing.clean_main(main_df)
    preprocessing.save_cleaned(cleaned)

    print("\n[3/10] feature engineering (anomaly z + event proximity)")
    feats = feature_engineering.build_all_features(cleaned, events)

    print("\n[4/10] verifying composite risk score + applying current bands")
    scored = risk_scoring.compute_supplyguard_score(feats)
    print(f"  risk distribution: {scored['supplyguard_risk_score_0_100'].describe().round(2).to_dict()}")

    print("\n[5/10] anomaly detection")
    scored = anomaly_detection.run_anomaly_pipeline(scored)
    anomaly_detection.save_anomaly_results(scored)

    print("\n[6/10] saving modeling dataset")
    risk_scoring.save_modeling_dataset(scored)
    risk_scoring.save_latest_snapshot(scored)

    print("\n[7/10] per-product forecasting (4 models, best by MAE)")
    forecasting.evaluate_models(scored)

    print("\n[8/10] data quality + missingness summaries")
    quality.run(main_df, cleaned, scored)

    print("\n[9/10] sensitivity analysis (±20% one-at-a-time)")
    sensitivity.run(scored)

    print("\n[10/10] event backtesting + EDA figures")
    evaluation.run_event_backtest(scored, events)
    visualization.generate_all_figures(scored, events)
    eda.print_summary(scored, events)

    print("\nDONE. See data/processed/ and data/outputs/ for results.")


if __name__ == "__main__":
    main()
