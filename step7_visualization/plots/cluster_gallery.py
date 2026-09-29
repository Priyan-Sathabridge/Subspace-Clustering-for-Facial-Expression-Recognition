"""
cluster_gallery.py

Creates a gallery of images for each cluster.
"""

from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from .base import BasePlot


class ClusterGallery(BasePlot):
    """
    Create one gallery image for each cluster.
    """

    def __init__(
        self,
        image_folder,
        labels_csv,
        output_folder,
        images_per_row=5,
        image_size=(200, 200),
        show_filenames=True
    ):

        self.image_folder = Path(image_folder)
        self.labels_csv = Path(labels_csv)
        self.output_folder = Path(output_folder)

        self.images_per_row = images_per_row
        self.image_size = image_size
        self.show_filenames = show_filenames

    # -----------------------------------------------------

    def create(self):

        labels = pd.read_csv(self.labels_csv)

        required = {"Image", "Cluster"}

        if not required.issubset(labels.columns):
            raise ValueError(
                "labels.csv must contain 'Image' and 'Cluster' columns."
            )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        clusters = sorted(labels["Cluster"].unique())

        for cluster in clusters:

            cluster_images = labels.loc[
                labels["Cluster"] == cluster,
                "Image"
            ].tolist()

            self._create_cluster_gallery(
                cluster,
                cluster_images
            )

    # -----------------------------------------------------

    def _create_cluster_gallery(
        self,
        cluster,
        image_list
    ):

        n_images = len(image_list)

        n_cols = self.images_per_row

        n_rows = math.ceil(
            n_images / n_cols
        )

        fig, axes = plt.subplots(

            n_rows,
            n_cols,

            figsize=(
                3 * n_cols,
                3 * n_rows
            )

        )

        if n_rows == 1 and n_cols == 1:
            axes = [axes]

        elif n_rows == 1:
            axes = axes.flatten()

        else:
            axes = axes.ravel()

        # ----------------------------------------------

        for ax in axes:
            ax.axis("off")

        # ----------------------------------------------

        for i, image_name in enumerate(image_list):

            image_path = self.image_folder / image_name

            if not image_path.exists():

                axes[i].text(

                    0.5,
                    0.5,

                    "Missing",

                    ha="center",
                    va="center"

                )

                continue

            image = Image.open(image_path)

            image = image.resize(
                self.image_size
            )

            axes[i].imshow(image)

            axes[i].axis("off")

            if self.show_filenames:

                axes[i].set_title(

                    image_name,

                    fontsize=8

                )

        plt.suptitle(

            f"Cluster {cluster}",

            fontsize=18

        )

        plt.tight_layout()

        plt.savefig(

            self.output_folder /
            f"cluster_{cluster}.png",

            dpi=300,

            bbox_inches="tight"

        )

        plt.close(fig)