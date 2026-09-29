import pandas as pd


def save_results(
        metrics,
        table,
        output_dir):

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        output_dir /
        "external_metrics.csv",
        index=False
    )

    table.to_csv(
        output_dir /
        "confusion_matrix.csv"
    )