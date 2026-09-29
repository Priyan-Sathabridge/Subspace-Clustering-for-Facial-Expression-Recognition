"""Root-configured Step 9 ANOVA entry point."""
from config import *
from anova import run_anova


def run():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    return run_anova(
        feature_csv=FEATURE_CSV,
        cluster_csv=CLUSTER_CSV,
        output_csv=OUTPUT_CSV,
        image_column=IMAGE_COLUMN,
        cluster_column=CLUSTER_COLUMN,
    )


if __name__ == "__main__":
    run()
