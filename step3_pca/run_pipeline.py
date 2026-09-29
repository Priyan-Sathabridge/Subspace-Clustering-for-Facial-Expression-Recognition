"""
run_pipeline.py

Runs the complete PCA pipeline.
"""

from config import (
    INPUT_CSV,
    OUTPUT_FOLDER,
    VARIANCE_THRESHOLD,
)

from normalization import (
    load_features,
)

from pca import (
    fit_pca,
    get_pca_scores,
    get_loading_matrix,
    get_variance_table,
)

from outputs import (
    save_all_outputs,
)

from scree_plots import (
    create_scree_plot,
)


def run_pipeline():

    print("=" * 60)
    print("Principal Component Analysis Pipeline")
    print("=" * 60)

    # ------------------------------------------------------
    # Load feature matrix
    # ------------------------------------------------------

    print("\nLoading feature matrix...")

    image_names, feature_names, X = load_features(INPUT_CSV)

    print(f"Samples  : {len(image_names)}")
    print(f"Features : {len(feature_names)}")

    # ------------------------------------------------------
    # Normalize
    # ------------------------------------------------------

    print("\nUsing raw features (no normalization)...")

    # Stage 2 has already scaled and weighted features. Scaling again here
    # would change those intended feature-group weights.
    X_input = X

    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    print("\nPerforming PCA...")

    pca, X_pca = fit_pca(
        X_input,
        variance_threshold=VARIANCE_THRESHOLD,
    )

    scores = get_pca_scores(
        image_names,
        X_pca,
    )

    loadings = get_loading_matrix(
        pca,
        feature_names,
    )

    variance_table = get_variance_table(
        pca,
    )

    retained_components = X_pca.shape[1]

    explained = variance_table[
        "Cumulative Variance"
    ].iloc[-1]

    print(f"Retained Components : {retained_components}")
    print(f"Variance Explained  : {explained:.2%}")

    # ------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------

    print("\nSaving outputs...")

    save_all_outputs(
        image_names=image_names,
        X_input=X_input,
        scores=scores,
        loadings=loadings,
        variance_table=variance_table,
        variance_threshold=VARIANCE_THRESHOLD,
        output_folder=OUTPUT_FOLDER,
    )

    # ------------------------------------------------------
    # Scree plot
    # ------------------------------------------------------

    print("\nGenerating scree plot...")

    create_scree_plot(
        variance_table=variance_table,
        variance_threshold=VARIANCE_THRESHOLD,
        output_html=OUTPUT_FOLDER / "scree_plot.html",
    )

    print("\nPipeline completed successfully.")

    print("=" * 60)


if __name__ == "__main__":

    run_pipeline()