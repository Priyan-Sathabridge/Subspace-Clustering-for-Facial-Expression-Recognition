import numpy as np


def calculate_normalized_cut(
        affinity_matrix,
        labels):

    W = affinity_matrix

    unique = np.unique(labels)

    ncut = 0

    for c in unique:

        inside = np.where(labels == c)[0]

        outside = np.where(labels != c)[0]

        cut = W[np.ix_(inside, outside)].sum()

        assoc = W[inside].sum()

        ncut += cut / assoc

    return ncut