# Run final results

> Original local experiment notes: paths, environment and sample counts below are machine-specific. See [README.md](README.md) for portable commands.

Double-click **Run Final Results.command** in the dashboard folder.

The launcher uses the existing sc_dashboard2 Python environment and resumes at stage 4 using the 132 aligned samples in `Labelled_Expression_Data`. The existing eight PCA components explain 90.11% of variance. It runs both Robust SSC and Robust LRSC with four clusters and the parameters in `pipeline_config.py`:

| Model | Lambda | Gamma |
| --- | --- | --- |
| Robust SSC | 0.025897373396156047 | 0.3986470631277377 |
| Robust LRSC | 1.2247448713915892 | 0.38729833462074176 |

Each run creates a dated folder inside `/Users/priyansathabridge/Desktop/Labelled_Expression_Data/final_results`. It copies the existing processed features, PCA files and labelled images, snapshots the dashboard code, and records package versions and input checksums. Earlier results remain intact. The launcher validates image IDs, row order, numeric values and image availability before fitting.

The experiment reuses existing preprocessing and PCA; their historical settings are not independently established by this launcher. It does not perform a new parameter search. The parameters above are taken from the current dashboard configuration, not independently reselected.

## Outputs inside each dated run folder

- `step_4`: coefficients, errors and convergence diagnostics for each model.
- `step_5`: affinity matrices.
- `step_6`: cluster assignments under each model's `k_4` folder.
- `step_7`: cluster galleries, UMAP and affinity plots.
- `step_8`: internal evaluation metrics.
- `step_9`: feature ANOVA.
- `step_10`: external labelled evaluation.
- `run.log`, `pipeline_logs`, `input_manifest.json`, `environment.txt`: execution and provenance records.

The pipeline stops on failure, including a model failing to converge. A failed run is not a complete final result; inspect its log before using its outputs. Each new launch starts a fresh run.

To validate inputs without fitting, run from the dashboard folder:

```sh
/opt/anaconda3/envs/sc_dashboard2/bin/python final_results.py
```

See [README.md](README.md) for portable setup and command-line usage. The root runner is `pipeline.py`, with shared settings in `pipeline_config.py`.
