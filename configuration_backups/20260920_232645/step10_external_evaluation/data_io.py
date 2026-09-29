import pandas as pd


def load_data(gt_csv,
              cluster_csv,
              image_col):

    gt = pd.read_csv(gt_csv)
    clusters = pd.read_csv(cluster_csv)

    merged = pd.merge(
        gt,
        clusters,
        on=image_col,
        how="inner"
    )

    return merged