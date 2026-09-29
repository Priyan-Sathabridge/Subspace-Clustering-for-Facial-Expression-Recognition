"""
run.py

Step 1:
Geometry Feature Extraction

This script:
    1. Loads all images.
    2. Detects facial landmarks.
    3. Executes every enabled geometry plugin.
    4. Saves geometry_features.csv
"""

from pathlib import Path

import pandas as pd

from config import (
    IMAGE_FOLDER,
    OUTPUT_FOLDER,
    ENABLED_FEATURES,
    IMAGE_EXTENSIONS
)

from detector.insightface_detector import (
    InsightFaceDetector
)

from geometry.combine_features import (
    extract_all_features
)


# ---------------------------------------------------------
# Image iterator
# ---------------------------------------------------------

def image_iterator(folder):

    folder = Path(folder)

    # Stable filename order establishes the sample order for downstream matrices.
    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() in IMAGE_EXTENSIONS:

            yield image_path


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def run():

    image_folder = Path(IMAGE_FOLDER)

    output_folder = Path(OUTPUT_FOLDER)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    detector = InsightFaceDetector()

    rows = []

    print("=" * 60)
    print("Step 1 - Geometry Feature Extraction")
    print("=" * 60)

    for image_path in image_iterator(image_folder):

        print(f"Processing {image_path.name}")

        image, landmarks = detector.detect_from_file(
            image_path
        )

        if landmarks is None:

            print(
                f"  No face detected."
            )

            continue

        try:

            features = extract_all_features(

                landmarks,

                enabled=ENABLED_FEATURES

            )

            features["Image"] = image_path.name

            rows.append(features)

        except Exception as e:

            print(

                f"  Failed: {image_path.name}"

            )

            print(e)

    if len(rows) == 0:

        raise RuntimeError(

            "No features were extracted."

        )

    df = pd.DataFrame(rows)

    columns = ["Image"] + [

        c for c in df.columns

        if c != "Image"

    ]

    df = df[columns]

    output_csv = (

        output_folder /

        "geometry_features.csv"

    )

    df.to_csv(

        output_csv,

        index=False

    )

    print()

    print("=" * 60)
    print(f"Processed {len(df)} images")
    print(f"Saved to:")
    print(output_csv)
    print("=" * 60)


# ---------------------------------------------------------

if __name__ == "__main__":

    run()