"""
run.py

Step 7:
Visualization
"""

from config import *

from plots.cluster_gallery import ClusterGallery
from plots.umap_projection import UMAPProjection
from plots.affinity_heatmap import AffinityHeatmap


def run():

    print("=" * 60)
    print("Step 7 - Visualization")
    print("=" * 60)

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Cluster Gallery
    # --------------------------------------------------

    print("\nCreating Cluster Galleries...")

    gallery = ClusterGallery(

        image_folder=IMAGE_FOLDER,

        labels_csv=LABELS_CSV,

        output_folder=OUTPUT_FOLDER,

        images_per_row=IMAGES_PER_ROW,

        image_size=IMAGE_SIZE,

        show_filenames=SHOW_FILENAMES

    )

    gallery.create()

    print("Done.")

    # --------------------------------------------------
    # UMAP
    # --------------------------------------------------

    print("\nCreating UMAP Projections...")

    # UMAP is a view of PCA features coloured by existing labels; it does not
    # refit the clustering or determine the number of clusters.
    umap_plot = UMAPProjection(

        features_csv=PCA_FEATURES_CSV,

        labels_csv=LABELS_CSV,

        output_folder=OUTPUT_FOLDER,

        n_neighbors=UMAP_NEIGHBORS,

        min_dist=UMAP_MIN_DIST,

        random_state=UMAP_RANDOM_STATE

    )

    umap_plot.create()

    print("Done.")

    # --------------------------------------------------
    # Heatmap
    # --------------------------------------------------

    print("\nCreating Affinity Heatmap...")

    heatmap = AffinityHeatmap(

        affinity_csv=AFFINITY_CSV,

        labels_csv=LABELS_CSV,

        output_folder=OUTPUT_FOLDER,

        cmap=HEATMAP_CMAP,

        show_boundaries=SHOW_CLUSTER_BOUNDARIES

    )

    heatmap.create()

    print("Done.")

    print()

    print("=" * 60)
    print("Visualization Complete")
    print("=" * 60)


if __name__ == "__main__":

    run()