# Facial Geometry Subspace-Clustering Pipeline

A research pipeline for grouping facial images using landmark-derived geometry. It extracts eye, mouth and eyebrow measurements, processes the features, applies PCA, and compares Robust Sparse Subspace Clustering (Robust SSC) with Robust Low-Rank Subspace Clustering (Robust LRSC). Both methods construct a graph for spectral clustering.

The supported entry point is `pipeline.py`; `main.py` forwards to the same CLI. Shared settings live in `pipeline_config.py`. Step-level configuration files adapt those settings for each script.

## Setup

Use Python 3.10 or later (required by the source syntax) in a dedicated environment. Run commands from this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. The dependency list is unpinned and is not a tested environment lock. InsightFace may need build tools and model assets on first use; the detector defaults to `CPUExecutionProvider` and uses 106 landmarks from the first detected face. Supply cropped, readable, single-face images.

## Configure inputs

Set absolute paths before running (POSIX shell examples):

```sh
export SC_DATA_ROOT="/absolute/path/to/experiment"
export SC_IMAGE_FOLDER="$SC_DATA_ROOT/all_images"
export SC_GROUND_TRUTH_CSV="$SC_DATA_ROOT/ground_truth.csv"
```

In PowerShell use `$env:SC_DATA_ROOT = "C:\path\to\experiment"` and the equivalent syntax for other variables. Without overrides, the code retains the original author's Desktop data path. The repository's `data/` and `results/` folders are not the active default output roots.

```text
experiment/
  all_images/          # images directly in this folder, not nested subfolders
    face_001.jpg
    face_002.jpg
  ground_truth.csv     # required for stage 10 and all final_results.py workflows
```

Ground truth for stage 10 has these columns:

```csv
Image,Label
face_001.jpg,happy
face_002.jpg,sad
```

`final_results.py` additionally requires a `Path` column locating each source image, either absolute or relative to its `--source` directory:

```csv
Image,Label,Path
face_001.jpg,happy,all_images/face_001.jpg
face_002.jpg,sad,all_images/face_002.jpg
```

Keep `Image` identifiers unique and unchanged throughout the pipeline. Feature/PCA tables use one row per image with numeric feature columns. Coefficient and affinity CSVs have headers but no image index; their rows and columns follow PCA sample order. Do not sort or filter one artifact independently. Stage 1 skips images with no detected face; stage 2 can drop rows with missing values. Before external evaluation, ground truth must contain exactly the retained image set. Cluster label CSVs contain `Image,Cluster`.

## Run the pipeline

```sh
python pipeline.py --list
python pipeline.py --status
# Preview all commands without needing existing intermediate files:
python pipeline.py --from 1 --to 10 --method both --dry-run --no-checks
# Full labelled workflow:
python pipeline.py --from 1 --to 10 --method both
# Without ground-truth labels, stop after ANOVA:
python pipeline.py --from 1 --to 9 --method both
# Resume with existing PCA:
python pipeline.py --from 4 --method both
# Selected stages / one model:
python pipeline.py --only 5,6 --method robust_ssc
python pipeline.py --only external --method robust_lrsc
```

The runner launches each stage in its own working directory using the same Python interpreter. Stages 1–3 run once; stage 4 fits the requested methods in one process; stages 5–10 run separately for each method. CLI aliases `ssc` and `lrsc` select the **robust** implementations. `python main.py all` and `python main.py pca` remain supported.

| Option | Behaviour |
| --- | --- |
| `--from`, `--to` | Inclusive stage range, accepting numbers or names. |
| `--only` | Comma-separated stages, executed in numerical order; overrides the range. |
| `--method` | `both` (default), `robust_ssc`, or `robust_lrsc`. |
| `--dry-run` | Prints commands; prerequisite checks still apply unless `--no-checks` is used. |
| `--skip-completed` | Skips stages based on expected output existence; does not validate settings, convergence, contents or freshness. |
| `--continue-on-error` | Continues after a subprocess failure; missing prerequisites still stop execution. |
| `--no-checks` | Bypasses the runner's path checks, not validation inside stage scripts. |

Normal runs write per-stage logs and a `manifest.json` under `SC_DATA_ROOT/pipeline_logs/<timestamp>/`. A prerequisite failure can exit before writing the manifest. Running again in the same data root can overwrite results. After changing inputs or settings, regenerate affected downstream stages instead of relying on `--skip-completed`.

## Stages and outputs

Paths below are relative to `SC_DATA_ROOT`. `<method>` is `robust_ssc` or `robust_lrsc`; `<k>` defaults to 4.

| Stage | Source folder | Main work and outputs |
| --- | --- | --- |
| 1 | `step1_feature_extraction/` | InsightFace landmarks and geometry → `step_1/geometry_features.csv`. |
| 2 | `step2_feature_processing/` | Validation, missing values, scaling and group weights → `step_2/processed_features.csv`, weights and diagnostics. |
| 3 | `step3_pca/` | PCA → `step_3/pca_features.csv`, loadings, explained variance and scree HTML. No additional scaling is applied here. |
| 4 | `step4_coefficients/` | Robust ADMM fits → `step_4/<method>/coefficients.csv`, `error_matrix.csv`, `image_order.csv`, `coefficient_info.csv`. |
| 5 | `step5_affinity/` | Symmetric graph, `W = abs(C) + abs(C).T` → `step_5/<method>/affinity.csv`. |
| 6 | `step6_clustering/` | Spectral clustering → `step_6/<method>/k_<k>/labels.csv`. |
| 7 | `step7_visualization/` | Image galleries, UMAP and affinity plots → `step_7/<method>/k_<k>/`. |
| 8 | `step8_evaluation/` | Internal metrics → `step_8/<method>/evaluation_results.csv` and plots. |
| 9 | `step9_anova/` | ANOVA on processed geometry → `step_9/<method>/anova_results.csv`. Includes eta-squared and Benjamini–Hochberg adjusted p-values. |
| 10 | `step10_external_evaluation/` | ARI, NMI, purity and contingency table → `step_10/<method>/external_metrics.csv`, `confusion_matrix.csv`. |

Stage 4 accepts samples × features, L2-normalises sample rows, and works internally with features × samples. Both methods model `X = X C + E` with zero diagonal in `C`. SSC penalises coefficient L1 norm; LRSC penalises nuclear norm; both penalise sparse error L1 norm. Diagnostic files are saved before the convergence check raises an error, so their presence alone does not establish a successful fit.

## Settings

Edit `pipeline_config.py` for shared experimental settings. Defaults are missing-value dropping, standard scaling, square-root group weighting, 90% PCA variance retention, four clusters, `kmeans` label assignment and random seed 42. The configured lambda/gamma values are existing experimental selections, not a fresh search or a guarantee of optimality for new data.

| Environment variable | Purpose |
| --- | --- |
| `SC_DATA_ROOT` | Shared input/output root. |
| `SC_IMAGE_FOLDER` | Image folder; defaults to `<data root>/all_images`. |
| `SC_GROUND_TRUTH_CSV` | Ground-truth file; defaults to `<data root>/ground_truth.csv`. |
| `SC_GEOMETRY_FEATURES_CSV` | Optional stage-2 input override; otherwise stage-1 output, then legacy `<data root>/geometry_features.csv`. |
| `SC_N_CLUSTERS` | Cluster count (at least 2; use fewer clusters than samples). |
| `SC_EVALUATION_MIN_K`, `SC_EVALUATION_MAX_K` | Inclusive stage-8 range; default to `SC_N_CLUSTERS`. All corresponding labels must exist. |
| `SC_PIPELINE_METHOD` | Branch for direct stage execution; the runner sets this automatically. |
| `SC_K_SWEEP=1` | Places external results under `step_10/<method>/k_<k>/`; set automatically by sweep workflows. |

## Isolated final runs and cluster-count sweeps

`final_results.py` reuses `step_2/processed_features.csv` and `step_3/pca_features.csv`; it does not recompute preprocessing or PCA. Its `--source` requires those files, ground truth with `Image,Label,Path`, and accessible images. All three CSVs must have exactly the same image order. Pass `--project` and `--source` explicitly because their defaults are machine-specific.

```sh
# Validate only:
python final_results.py --project . --source "$SC_DATA_ROOT"
# Copy inputs/code and execute stages 4–10 in a new timestamped folder:
python final_results.py --project . --source "$SC_DATA_ROOT" --run
# Fit coefficients/affinity once, then compare k=3..10:
python final_results.py --project . --source "$SC_DATA_ROOT" --internal-sweep --k-min 3 --k-max 10 --run
python final_results.py --project . --source "$SC_DATA_ROOT" --sweep --k-min 3 --k-max 10 --run
```

By default these produce timestamped folders under `final_results/`, `internal_evaluation_sweeps/`, or `external_evaluation_sweeps/` in the source directory. `--output-parent` overrides the parent; keep it outside the source code directory to avoid recursive code snapshots. Each run includes copied inputs, `dashboard_code/`, input checksums, package versions and logs. Internal sweeps run stages 4, 5, 6 and 8; external sweeps run 4, 5, 6 and 10. Internal sweeps still require labelled inputs because they share the final-run validator.

Regenerate comparison figures from a completed sweep:

```sh
python internal_evaluation_plots.py --root /absolute/path/to/internal/run --k-min 3 --k-max 10
python external_evaluation_plots.py --root /absolute/path/to/external/run --k-min 3 --k-max 10
```

The macOS `.command` launchers and [final-run](FINAL_RUN.md), [internal-sweep](INTERNAL_SWEEP.md), and [external-sweep](EXTERNAL_SWEEP.md) notes describe the original local experiment. Their environment paths and sample counts are not portable defaults; use the commands above on another machine.

## Parameter search and interpretation

`step11_parameter_search/` contains standalone research searches and surface-generation scripts; it is not stage 11 of the root CLI. Configure paths and search grids inside the relevant script, then run from the repository root, for example:

```sh
python -m step11_parameter_search.robust_low_rank_parameter_search
python -m step11_parameter_search.robust_ssc_parameter_search_interactive
```

See [parameter-search notes](step11_parameter_search/README.txt). Legacy coefficient implementations and older iterators remain in the tree; the root runner uses the final robust models described above.

Higher Silhouette, Affinity Silhouette, Affinity Ratio and CHI are generally preferable; lower DBI and Normalized Cut are generally preferable. Eigengaps suggest candidate cluster counts. External metrics compare clusters with expression labels; cluster numbers are arbitrary. Purity can rise with increasing k, so compare it alongside ARI and NMI. ANOVA is descriptive of the fitted clusters, and evaluation on the same labelled data used for parameter selection is not held-out validation.

## Troubleshooting and repository hygiene

- Missing prerequisites: inspect `--status`, confirm environment paths, and run earlier stages. Status checks output presence, not validity.
- Nonconvergence: inspect `coefficient_info.csv` and stage logs. Do not resume from saved coefficients until the failed fit is resolved.
- Image mismatch: check dropped samples, duplicate names, and PCA/matrix row order. Stage 8 aligns labels by image ID; it cannot recover a reordered unlabelled affinity matrix.
- Dependency or model errors: confirm the active interpreter and install dependencies there. Full image extraction requires InsightFace model assets in addition to Python packages.
- Large datasets: dense sample × sample matrices and repeated decompositions can be expensive in memory and runtime.

The `.gitignore` excludes local data/results, backups, caches and environments. Keep research images, generated results and model assets outside source control unless deliberately preparing an approved dataset release. The dependency list is based on source imports; full installation and dataset execution must be validated in the target environment. No automated end-to-end test suite is included.
