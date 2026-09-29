import pandas as pd

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import confusion_matrix


def adjusted_rand(labels_true,
                  labels_pred):

    return adjusted_rand_score(
        labels_true,
        labels_pred
    )


def normalized_mutual_information(
        labels_true,
        labels_pred):

    return normalized_mutual_info_score(
        labels_true,
        labels_pred
    )


def clustering_purity(
        labels_true,
        labels_pred):

    table = pd.crosstab(
        labels_pred,
        labels_true
    )

    purity = table.max(axis=1).sum() / table.values.sum()

    return purity


def contingency_table(
        labels_true,
        labels_pred):

    return pd.crosstab(
        labels_pred,
        labels_true
    )


def compute_metrics(
        labels_true,
        labels_pred):

    results = {

        "ARI":
            adjusted_rand(
                labels_true,
                labels_pred
            ),

        "NMI":
            normalized_mutual_information(
                labels_true,
                labels_pred
            ),

        "Purity":
            clustering_purity(
                labels_true,
                labels_pred
            )

    }

    table = contingency_table(
        labels_true,
        labels_pred
    )

    return results, table