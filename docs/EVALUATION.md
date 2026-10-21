# Forecast evaluation snapshot

Generated with the default dependency set (HistGradientBoosting, without XGBoost).
Models were selected on an earlier three-target-month validation window. The last six available target months form the later test window. Persistence can win selection.

| Product | Selected on validation | Test MAE | Baseline MAE | Test R2 | Test rows |
|---|---|---:|---:|---:|---:|
| Barley | Baseline (persistence) | 6.066 | 6.066 | -0.646 | 18 |
| Maize | Baseline (persistence) | 6.786 | 6.786 | -0.236 | 64 |
| Rice | LinearRegression | 5.629 | 6.634 | 0.039 | 96 |
| Sugar | HistGradientBoosting | 5.978 | 6.482 | 0.154 | 92 |
| Wheat | RandomForest | 7.024 | 6.804 | -3.129 | 7 |

The selected model can perform worse than persistence on the later test window. These values are retained rather than selecting again on the test results. Wheat has very few evaluation rows and is not evidence of a reliable forecast.

Calendar gaps are excluded from next-month labels. Publication lags and upstream proxy preparation are not fully simulated, so this is a retrospective evaluation, not a measured real-time warning capability.

Full tables: [model metrics](../data/outputs/forecast_model_metrics.csv), [split boundaries](../data/outputs/forecast_split_summary.csv), and [test predictions](../data/outputs/forecast_actual_vs_predicted.csv).
