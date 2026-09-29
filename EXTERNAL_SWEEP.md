# External evaluation for k = 3–10

> Original local experiment notes: paths, environment and sample counts below are machine-specific. See [README.md](README.md) for portable commands.

Double-click **Run External Evaluation.command**. This uses the sc_dashboard2 environment and runs both Robust SSC and Robust LRSC on the same 132 images and existing eight PCA components.

Coefficients and affinity are computed once per model using the existing lambda/gamma parameters. Clustering and external evaluation then run for every integer k from 3 through 10. The four ground-truth expression labels remain unchanged for every k. Random seed remains 42. This sweep does not run galleries, internal evaluation or ANOVA.

Each launch creates a new dated folder under `/Users/priyansathabridge/Desktop/Labelled_Expression_Data/external_evaluation_sweeps` with:

- `step_6/<method>/k_<k>/labels.csv`: all cluster assignments.
- `step_10/<method>/k_<k>/external_metrics.csv`: ARI, NMI and purity.
- `step_10/<method>/k_<k>/confusion_matrix.csv`: cluster-by-expression contingency table.
- `external_evaluation_graphs`: individual metric plots and a combined comparison, in PNG and PDF, plus `external_metrics_by_k.csv` containing all 16 evaluations.
- Input copies, code snapshot, package versions, checksums and run logs.

Purity can increase as the number of clusters increases; consider it alongside ARI and NMI when comparing k. These are evaluations on the same existing labelled data, not a separate held-out test set.

To check inputs without fitting:

```sh
/opt/anaconda3/envs/sc_dashboard2/bin/python final_results.py --sweep --k-min 3 --k-max 10
```

To run from Terminal, add `--run`. The existing **Run Final Results.command** continues to run the original single-k workflow. A failed fit or missing evaluation stops the sweep before comparison graphs are produced.
