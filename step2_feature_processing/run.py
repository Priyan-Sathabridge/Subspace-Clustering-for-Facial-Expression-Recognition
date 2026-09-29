"""
run.py

Step 2:
Feature Processing Pipeline
"""

import pandas as pd

from config import (
    INPUT_CSV,
    OUTPUT_FOLDER,
    REPORT_FOLDER,
    SAVE_VALIDATION_REPORT,
    MISSING_VALUE_METHOD,
    NORMALIZATION_METHOD,
    WEIGHTING_METHOD
)

from processing.validator import FeatureValidator
from processing.missing_values import MissingValueHandler
from processing.normalization import Normalizer
from processing.weighting import (
    FeatureWeighter,
    build_feature_groups
)


# ==========================================================
# Main Pipeline
# ==========================================================

def run():

    print("=" * 60)
    print("Step 2 - Feature Processing")
    print("=" * 60)

    # ------------------------------------------------------
    # Create output folders
    # ------------------------------------------------------

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Load feature matrix
    # ------------------------------------------------------

    print("\nLoading feature matrix...")

    df = pd.read_csv(INPUT_CSV)

    print(f"Images   : {len(df)}")
    print(f"Features : {len(df.columns)-1}")

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    print("\nValidating dataset...")

    validator = FeatureValidator()

    validator.validate(df)

    validator.print_report()

    if SAVE_VALIDATION_REPORT:

        validator.save_report(
            REPORT_FOLDER / "validation_report.txt"
        )

    # ------------------------------------------------------
    # Missing Values
    # ------------------------------------------------------

    print("\nHandling missing values...")

    handler = MissingValueHandler(
        method=MISSING_VALUE_METHOD
    )

    # Dropping missing rows changes the retained image set; later ground truth
    # must match this set exactly for external evaluation.
    df = handler.process(df)

    # ------------------------------------------------------
    # Normalization
    # ------------------------------------------------------

    print("\nNormalizing features...")

    normalizer = Normalizer(
        method=NORMALIZATION_METHOD
    )

    df = normalizer.process(df)

    # ------------------------------------------------------
    # Automatic Feature Grouping
    # ------------------------------------------------------

    print("\nGrouping features...")

    feature_groups = build_feature_groups(df)

    for group, columns in feature_groups.items():
        print(f"{group}: {len(columns)} features")

    # ------------------------------------------------------
    # Feature Weighting
    # ------------------------------------------------------

    print("\nApplying feature weights...")

    weighter = FeatureWeighter(
        feature_groups,
        method=WEIGHTING_METHOD
    )

    df = weighter.process(df)

    weighter.save_weights(
        OUTPUT_FOLDER / "feature_weights.csv"
    )

    # ------------------------------------------------------
    # Save processed dataset
    # ------------------------------------------------------

    output_file = OUTPUT_FOLDER / "processed_features.csv"

    df.to_csv(
        output_file,
        index=False
    )

    # ------------------------------------------------------
    # Save normalization statistics
    # ------------------------------------------------------

    stats = normalizer.get_statistics()

    for name, values in stats.items():

        stats_df = pd.DataFrame(values)

        stats_df.to_csv(
            OUTPUT_FOLDER / f"{name}.csv"
        )

    # ------------------------------------------------------

    print("\nProcessing complete.")

    print(f"\nProcessed features saved to:\n{output_file}")

    print("\nFeature weights saved to:")
    print(OUTPUT_FOLDER / "feature_weights.csv")

    print("=" * 60)


# ==========================================================

if __name__ == "__main__":
    run()