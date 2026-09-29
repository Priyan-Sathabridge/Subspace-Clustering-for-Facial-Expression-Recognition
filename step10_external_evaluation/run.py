"""Evaluate saved clusters against ground truth after matching image IDs."""

from config import *

from data_io import load_data
from metrics import compute_metrics
from results import save_results


def main():

    merged = load_data(
        GROUND_TRUTH_CSV,
        CLUSTER_LABELS_CSV,
        IMAGE_COLUMN
    )

    y_true = merged[
        GROUND_TRUTH_COLUMN
    ]

    y_pred = merged[
        CLUSTER_COLUMN
    ]

    metrics, table = compute_metrics(
        y_true,
        y_pred
    )

    print("\nExternal Validation")
    print("-------------------")

    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    print("\nContingency Table")
    print(table)

    save_results(
        metrics,
        table,
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()