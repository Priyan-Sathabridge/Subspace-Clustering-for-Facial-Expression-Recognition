"""
run.py

Step 5:
Affinity Matrix Construction
"""

import pandas as pd

from config import *

from methods.registry import AFFINITY_METHODS


# ==========================================================
# Build Affinity Constructor
# ==========================================================

def build_affinity_method():

    method = AFFINITY_METHOD.lower()

    if method not in AFFINITY_METHODS:

        raise ValueError(

            f"Unknown affinity method '{AFFINITY_METHOD}'."

        )

    if method == "symmetric":

        return AFFINITY_METHODS[method](

            normalize=NORMALIZE_AFFINITY

        )

    raise ValueError(

        f"No constructor defined for '{method}'."

    )


# ==========================================================
# Main Pipeline
# ==========================================================

def run():

    print("=" * 60)
    print("Step 5 - Affinity Matrix Construction")
    print("=" * 60)

    OUTPUT_FOLDER.mkdir(

        parents=True,
        exist_ok=True

    )

    # ------------------------------------------------------

    print("\nLoading coefficient matrix...")

    # This numeric matrix carries no image IDs; retain the PCA sample order.
    C = pd.read_csv(INPUT_CSV).to_numpy()

    print(f"Matrix Shape : {C.shape}")

    # ------------------------------------------------------

    affinity_builder = build_affinity_method()

    print(

        f"\nUsing {affinity_builder.name}"

    )

    W = affinity_builder.construct_affinity(C)

    # ------------------------------------------------------

    pd.DataFrame(W).to_csv(

        OUTPUT_FOLDER / "affinity.csv",

        index=False

    )

    # ------------------------------------------------------

    info = pd.DataFrame({

        "Property": [

            "Method",
            "Rows",
            "Columns"

        ],

        "Value": [

            affinity_builder.name,
            W.shape[0],
            W.shape[1]

        ]

    })

    info.to_csv(

        OUTPUT_FOLDER / "affinity_info.csv",

        index=False

    )

    print("\nAffinity matrix saved.")

    print(

        OUTPUT_FOLDER / "affinity.csv"

    )

    print("=" * 60)


# ==========================================================

if __name__ == "__main__":

    run()