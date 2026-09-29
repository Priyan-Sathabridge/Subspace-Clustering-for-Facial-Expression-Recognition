# Internal validation for k = 3–10

> Original local experiment notes: paths, environment and sample counts below are machine-specific. See [README.md](README.md) for portable commands.

Double-click **Run Internal Evaluation.command** to compare Robust SSC and Robust LRSC on the same 132 samples and existing eight PCA components, with the existing model parameters and random seed 42.

The launcher fits coefficients and affinity once per model, generates clustering for every integer k from 3 through 10, then evaluates the entire range in Step 8. It keeps all eight rows for each model and creates comparison plots for:

- Silhouette, Affinity Silhouette, Affinity Ratio, CHI: higher is generally better.
- DBI and Normalized Cut: lower is generally better.
- Eigengap: larger gaps indicate candidate cluster counts.

Metrics retain the existing Step 8 definitions: feature metrics use the saved PCA scores; affinity metrics use the saved affinity matrix. Eigengap is computed from the normalized graph Laplacian. These measures assess different aspects of clustering and need not agree.

Each launch creates a dated folder under `/Users/priyansathabridge/Desktop/Labelled_Expression_Data/internal_evaluation_sweeps` containing:

- `step_6/<method>/k_<k>/labels.csv` for each clustering.
- `step_8/<method>/evaluation_results.csv` and per-model plots for the full range.
- `internal_evaluation_graphs/internal_metrics_by_k.csv`: all 16 model/k evaluations.
- `internal_evaluation_graphs`: seven individual comparison plots and a combined figure, each in PNG and PDF.
- Input copies, code snapshot, package versions, checksums and run logs.

Features and cluster labels are aligned by image identifier, and missing or duplicate samples stop evaluation. A missing/nonfinite metric stops comparison plotting rather than producing an incomplete graph. Earlier results and the external-evaluation launcher are preserved.

To validate inputs without running:

```sh
/opt/anaconda3/envs/sc_dashboard2/bin/python final_results.py --internal-sweep --k-min 3 --k-max 10
```

Add `--run` to execute. The default single-k workflow and `--sweep` external workflow remain available.
