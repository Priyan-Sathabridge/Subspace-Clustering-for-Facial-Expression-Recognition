"""
iterator.py

Run clustering for multiple values of k.
"""

from pathlib import Path

import pandas as pd

from config import *

from algorithms.registry import CLUSTERING_METHODS


# ==========================================================
# Build Clustering Algorithm
# ==========================================================

def build_algorithm(k):

    method = CLUSTERING_METHOD.lower()

    if method not in CLUSTERING_METHODS:

        raise ValueError(
            f"Unknown clustering method '{CLUSTERING_METHOD}'."
        )

    return CLUSTERING_METHODS[method](

        n_clusters=k,

        assign_labels=ASSIGN_LABELS,

        random_state=RANDOM_STATE

    )


# ==========================================================
# Main Iterator
# ==========================================================

def run():

    print("=" * 60)
    print("Step 6 - Clustering Iterator")
    print("=" * 60)

    affinity = pd.read_csv(

        INPUT_CSV,
    ).to_numpy()

    # Load image names once

    features = pd.read_csv(PCA_FEATURES_CSV)

    if "Image" not in features.columns:
        raise ValueError(
            "PCA features must contain an 'Image' column."
        )

    image_names = features["Image"]

    print(f"Affinity Matrix Shape : {affinity.shape}")

    for k in range(

        ITERATOR_MIN_K,

        ITERATOR_MAX_K + 1

    ):

        print()

        print("-" * 50)
        print(f"Running k = {k}")
        print("-" * 50)

        algorithm = build_algorithm(k)

        labels = algorithm.cluster(affinity)

        output_folder = (

            OUTPUT_FOLDER /

            f"k_{k}"

        )

        output_folder.mkdir(

            parents=True,

            exist_ok=True

        )
        ##########################
        if len(image_names) != len(labels):
            raise ValueError(
                "Number of images does not match number of labels."
            )

        labels_df = pd.DataFrame({

            "Image": image_names,

            "Cluster": labels

        })
        ############################
        labels_df.to_csv(

            output_folder /

            "labels.csv",

            index=False

        )

        info = pd.DataFrame({

            "Property": [

                "Method",

                "Clusters",

                "Samples"

            ],

            "Value": [

                algorithm.name,

                k,

                len(labels)

            ]

        })

        info.to_csv(

            output_folder /

            "clustering_info.csv",

            index=False

        )

    print()
    print("=" * 60)
    print("Iterator Complete")
    print("=" * 60)


# ==========================================================

if __name__ == "__main__":

    run()