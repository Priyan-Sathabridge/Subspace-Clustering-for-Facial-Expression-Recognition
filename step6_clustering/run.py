"""
run.py

Step 6:
Clustering
"""

import pandas as pd

from config import *

from algorithms.registry import CLUSTERING_METHODS


# ==========================================================
# Build Clustering Algorithm
# ==========================================================

def build_clustering_method():

    method = CLUSTERING_METHOD.lower()

    if method not in CLUSTERING_METHODS:

        raise ValueError(

            f"Unknown clustering method '{CLUSTERING_METHOD}'."

        )

    if method == "spectral":

        return CLUSTERING_METHODS[method](

            n_clusters=N_CLUSTERS,

            assign_labels=ASSIGN_LABELS,

            random_state=RANDOM_STATE

        )

    raise ValueError(

        f"No constructor defined for '{method}'."

    )


# ==========================================================
# Main Pipeline
# ==========================================================

def run():

    print("=" * 60)
    print("Step 6 - Clustering")
    print("=" * 60)

    OUTPUT_FOLDER.mkdir(

        parents=True,
        exist_ok=True

    )

    # ------------------------------------------------------

    print("\nLoading affinity matrix...")

    affinity = pd.read_csv(

        INPUT_CSV,

    ).to_numpy()

    print(

        f"Affinity Matrix Shape : {affinity.shape}"

    )

    # ------------------------------------------------------

    clustering = build_clustering_method()

    print(

        f"\nUsing {clustering.name}"

    )

    labels = clustering.cluster(

        affinity

    )

    # ------------------------------------------------------
    # ------------------------------------------------------
    # Load image names
    # ------------------------------------------------------

    features = pd.read_csv(PCA_FEATURES_CSV)

    if "Image" not in features.columns:
        raise ValueError(
            "PCA features must contain an 'Image' column."
        )

    # Affinity indices are positional. A length check cannot detect reordered
    # PCA rows, so use the same PCA file that produced the coefficients.
    image_names = features["Image"]

    if len(image_names) != len(labels):
        raise ValueError(
            "Number of images does not match number of cluster labels."
        )

    # ------------------------------------------------------
    # Save labels
    # ------------------------------------------------------

    labels_df = pd.DataFrame({

        "Image": image_names,

        "Cluster": labels

    })

    labels_df.to_csv(

        OUTPUT_FOLDER / "labels.csv",

        index=False

    )




    # ------------------------------------------------------

    info = pd.DataFrame({

        "Property": [

            "Method",
            "Clusters",
            "Samples"

        ],

        "Value": [

            clustering.name,

            N_CLUSTERS,

            len(labels)

        ]

    })

    info.to_csv(

        OUTPUT_FOLDER / "clustering_info.csv",

        index=False

    )

    # ------------------------------------------------------

    print("\nClustering completed.")

    print(

        OUTPUT_FOLDER / "labels.csv"

    )

    print("=" * 60)


# ==========================================================

if __name__ == "__main__":

    run()