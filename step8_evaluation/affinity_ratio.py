import numpy as np


def calculate_affinity_ratio(
        affinity_matrix,
        labels):
    """
    Mean within-cluster affinity divided by
    mean between-cluster affinity.
    """

    W = affinity_matrix.astype(float)

    within = []
    between = []

    n = len(labels)

    for i in range(n):

        for j in range(i + 1, n):

            if labels[i] == labels[j]:

                within.append(W[i, j])

            else:

                between.append(W[i, j])

    within = np.mean(within)

    between = np.mean(between)

    if between == 0:

        return np.inf

    return within / between