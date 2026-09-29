"""
iterator.py

Generate visualizations for every clustering result.
"""

from pathlib import Path

from config import *

from plots.cluster_gallery import ClusterGallery
from plots.umap_projection import UMAPProjection
from plots.affinity_heatmap import AffinityHeatmap


def run():

    print("=" * 60)
    print("Step 7 - Visualization Iterator")
    print("=" * 60)

    for k in range(

        ITERATOR_MIN_K,

        ITERATOR_MAX_K + 1

    ):

        print()
        print("-" * 60)
        print(f"Processing k = {k}")
        print("-" * 60)

        # ----------------------------------------------
        # Step 6 folder
        # ----------------------------------------------

        k_folder = CLUSTERING_FOLDER / f"k_{k}"

        labels_csv = k_folder / "labels.csv"

        if not labels_csv.exists():

            print(f"Skipping k={k}")

            continue

        output_folder = (

            k_folder /

            "visualizations"

        )

        output_folder.mkdir(

            parents=True,

            exist_ok=True

        )

        # ----------------------------------------------
        # Cluster Gallery
        # ----------------------------------------------

        print("Creating Cluster Galleries...")

        ClusterGallery(

            image_folder=IMAGE_FOLDER,

            labels_csv=labels_csv,

            output_folder=output_folder,

            images_per_row=IMAGES_PER_ROW,

            image_size=IMAGE_SIZE,

            show_filenames=SHOW_FILENAMES

        ).create()

        # ----------------------------------------------
        # UMAP
        # ----------------------------------------------

        print("Creating UMAP...")

        UMAPProjection(

            features_csv=PCA_FEATURES_CSV,

            labels_csv=labels_csv,

            output_folder=output_folder,

            n_neighbors=UMAP_NEIGHBORS,

            min_dist=UMAP_MIN_DIST,

            random_state=UMAP_RANDOM_STATE

        ).create()

        # ----------------------------------------------
        # Heatmap
        # ----------------------------------------------

        print("Creating Heatmap...")

        AffinityHeatmap(

            affinity_csv=AFFINITY_CSV,

            labels_csv=labels_csv,

            output_folder=output_folder,

            cmap=HEATMAP_CMAP,

            show_boundaries=SHOW_CLUSTER_BOUNDARIES

        ).create()

        print(f"Finished k={k}")

    print()
    print("=" * 60)
    print("Iterator Complete")
    print("=" * 60)


if __name__ == "__main__":

    run()