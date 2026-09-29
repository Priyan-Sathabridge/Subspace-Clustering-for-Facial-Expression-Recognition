"""
umap_projection.py

Generate 2D and 3D UMAP visualizations.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import umap

from .base import BasePlot


class UMAPProjection(BasePlot):
    """
    Generate 2D PNG and 3D interactive HTML UMAP projections.
    """

    def __init__(
        self,
        features_csv,
        labels_csv,
        output_folder,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    ):

        self.features_csv = Path(features_csv)
        self.labels_csv = Path(labels_csv)
        self.output_folder = Path(output_folder)

        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    # --------------------------------------------------

    def create(self):

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        features = pd.read_csv(
            self.features_csv,
            index_col=0
        )

        labels = pd.read_csv(
            self.labels_csv
        )

        X = features.values
        y = labels["Cluster"].values

        self._create_2d(X, y)
        self._create_3d(X, y)

    # --------------------------------------------------

    def _create_2d(self, X, y):

        reducer = umap.UMAP(

            n_components=2,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state

        )

        embedding = reducer.fit_transform(X)

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(

            embedding[:, 0],
            embedding[:, 1],

            c=y,

            cmap="tab10",

            s=50

        )

        plt.title("2D UMAP Projection")

        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

        plt.colorbar(
            scatter,
            label="Cluster"
        )

        plt.tight_layout()

        plt.savefig(

            self.output_folder /
            "umap_2d.png",

            dpi=300

        )

        plt.close()

    # --------------------------------------------------

    def _create_3d(self, X, y):

        reducer = umap.UMAP(

            n_components=3,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state

        )

        embedding = reducer.fit_transform(X)

        df = pd.DataFrame({

            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "UMAP3": embedding[:, 2],
            "Cluster": y.astype(str)

        })

        fig = px.scatter_3d(

            df,

            x="UMAP1",
            y="UMAP2",
            z="UMAP3",

            color="Cluster",

            title="3D UMAP Projection"

        )

        fig.update_traces(

            marker=dict(size=4)

        )

        fig.write_html(

            self.output_folder /
            "umap_3d.html"

        )