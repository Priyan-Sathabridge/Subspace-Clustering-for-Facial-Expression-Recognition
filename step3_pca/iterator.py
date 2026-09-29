"""
iterator.py

Runs the PCA pipeline for multiple explained variance thresholds.

Each run is saved into its own output folder.

Example

output/
│
├── PCA_90/
├── PCA_95/
├── PCA_97/
└── PCA_99/
"""

from pathlib import Path

from config import (
    INPUT_CSV,
    OUTPUT_FOLDER,
)

from run_pipeline import run_pipeline


# ==========================================================
# Explained variance values to evaluate
# ==========================================================

VARIANCE_LEVELS = [

    0.90,

    0.95,

    0.97,

    0.99,

]


def run_iterator():

    print("=" * 70)
    print("Running PCA Iterator")
    print("=" * 70)

    for variance in VARIANCE_LEVELS:

        folder_name = f"PCA_{int(variance * 100)}"

        output_folder = OUTPUT_FOLDER / folder_name

        print("\n" + "=" * 70)
        print(f"Variance Retained : {variance:.0%}")
        print(f"Output Folder     : {output_folder}")
        print("=" * 70)

        run_pipeline(
            input_csv=INPUT_CSV,
            output_folder=output_folder,
            variance_threshold=variance,
        )

    print("\n")
    print("=" * 70)
    print("All PCA runs completed.")
    print("=" * 70)


if __name__ == "__main__":

    run_iterator()