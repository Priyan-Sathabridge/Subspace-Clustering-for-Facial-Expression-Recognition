STANDALONE PARAMETER SEARCH

These research scripts are separate from the ten-stage root pipeline.
Run from the repository root after installing requirements.txt.

Configure FEATURE_FILE, GROUND_TRUTH_FILE, OUTPUT_FOLDER and the search
settings in the chosen script; these scripts retain local paths and do
not automatically inherit all settings from pipeline_config.py.

Robust LRSC logarithmic lambda/gamma grid:
  python -m step11_parameter_search.robust_low_rank_parameter_search

Interactive variants:
  python -m step11_parameter_search.robust_low_rank_parameter_search_interactive
  python -m step11_parameter_search.robust_ssc_parameter_search_interactive

The LRSC search writes lambda_gamma_grid_results.csv,
best_parameters_by_metric.csv, 3d_surfaces/ PNG plots, and metric_grids/
CSV tables for ARI, NMI and purity. Interactive variants also export HTML
surfaces. Inspect each script for its exact output names and grid settings.

The misspelled low_rank_paramter_search.py filename is retained for legacy
usage; prefer robust_low_rank_parameter_search.py for new runs.

Selected parameters are not automatically copied into the final pipeline.
Update pipeline_config.py deliberately after reviewing the search results.
Scores on data used to choose parameters are not held-out evaluation.
