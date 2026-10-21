# Publication preparation

Based on the local `ese1v` SupplyGuard version with five products and nine views.

- Excluded credentials, environments, caches, duplicate source copies, and legacy rice-only artifacts that were not used by the current application.
- Kept one input snapshot and regenerated the dashboard outputs.
- Separated model selection from final test evaluation using target-date windows.
- Included persistence in model selection and refitted the selected model separately for operational snapshot forecasts.
- Made lag and target construction respect missing calendar months.
- Fixed event filtering for semicolon-separated products and slash-separated countries.
- Matched dashboard model labels to validation selection, including baseline winners.
- Added unit tests, headless dashboard coverage, dependency constraints, and CI configuration.

Optional XGBoost and Gemini dependencies are separate from the default install.
Tests do not call a live model provider. Method changes require new output files;
the earlier local project results were not presented as results of this version.
