"""
affinity_heatmap.py

Generate a reordered affinity matrix heatmap.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import BasePlot


class AffinityHeatmap(BasePlot):
    """
    Create a heatmap of the affinity matrix
    reordered by cluster labels.
    """

    def __init__(
        self,
        affinity_csv,
        labels_csv,
        output_folder,
        cmap="viridis",
        show_boundaries=True
    ):

        self.affinity_csv = Path(affinity_csv)
        self.labels_csv = Path(labels_csv)
        self.output_folder = Path(output_folder)

        self.cmap = cmap
        self.show_boundaries = show_boundaries

    # -----------------------------------------------------

    def create(self):

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        affinity = pd.read_csv(
            self.affinity_csv,
        )

        labels = pd.read_csv(
            self.labels_csv
        )

        # ---------------------------------------------

        labels = labels.sort_values(
            "Cluster"
        )

        ordering = labels.index.to_numpy()

        reordered = affinity.iloc[
            ordering,
            ordering
        ]

        ordered_labels = labels["Cluster"].to_numpy()

        # ---------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )

        im = ax.imshow(

            reordered,

            cmap=self.cmap,

            interpolation="nearest",

            aspect="equal"

        )

        plt.colorbar(
            im,
            ax=ax,
            label="Affinity"
        )

        ax.set_title(
            "Affinity Matrix (Ordered by Cluster)"
        )

        ax.set_xticks([])
        ax.set_yticks([])

        # ---------------------------------------------
        # Draw cluster boundaries
        # ---------------------------------------------

        if self.show_boundaries:

            boundaries = []

            previous = ordered_labels[0]

            for i, label in enumerate(ordered_labels):

                if label != previous:

                    boundaries.append(i)

                    previous = label

            for boundary in boundaries:

                ax.axhline(

                    boundary - 0.5,

                    color="white",

                    linewidth=2

                )

                ax.axvline(

                    boundary - 0.5,

                    color="white",

                    linewidth=2

                )

        plt.tight_layout()

        plt.savefig(

            self.output_folder /
            "affinity_heatmap.png",

            dpi=300

        )

        plt.close(fig)