"""
Eyebrow feature extraction.
"""

import numpy as np
from .registry import register_feature


from .base import (
    distance,
    aspect_ratio,
    quadratic_curvature
)


def eyebrow_height(points):

    return (
        np.max(points[:, 1]) -
        np.min(points[:, 1])
    )


def eyebrow_slope(inner, outer):

    dx = outer[0] - inner[0]

    dy = outer[1] - inner[1]

    return np.degrees(
        np.arctan2(dy, dx)
    )


def eyebrow_features(region, eye):

    pts = region["all_points"]

    width = distance(
        region["inner"],
        region["outer"]
    )

    height = eyebrow_height(pts)

    centre = pts.mean(axis=0)

    eye_centre = eye["all_points"].mean(axis=0)

    return {

        "width": width,

        "height": height,

        "ratio": aspect_ratio(
            height,
            width
        ),

        "slope":
            eyebrow_slope(
                region["inner"],
                region["outer"]
            ),

        "curve":
            quadratic_curvature(pts),

        "distance":
            distance(
                centre,
                eye_centre
            )

    }


@register_feature("eyebrows")
def extract_features(
    landmarks,
    mapping
):

    left = eyebrow_features(
        mapping["left_brow"],
        mapping["left_eye"]
    )

    right = eyebrow_features(
        mapping["right_brow"],
        mapping["right_eye"]
    )

    return {

        "Left Brow Width":
            left["width"],

        "Right Brow Width":
            right["width"],

        "Left Brow Height":
            left["height"],

        "Right Brow Height":
            right["height"],

        "Left Brow Ratio":
            left["ratio"],

        "Right Brow Ratio":
            right["ratio"],

        "Left Brow Slope":
            left["slope"],

        "Right Brow Slope":
            right["slope"],

        "Left Brow Curvature":
            left["curve"],

        "Right Brow Curvature":
            right["curve"],

        "Inner Brow Distance":
            distance(
                mapping["left_brow"]["inner"],
                mapping["right_brow"]["inner"]
            ),

        "Brow Centre Distance":
            distance(
                mapping["left_brow"]["centre"],
                mapping["right_brow"]["centre"]
            ),

        "Eyebrow Height Symmetry":
            abs(
                left["height"] -
                right["height"]
            ),

        "Eyebrow Ratio Symmetry":
            abs(
                left["ratio"] -
                right["ratio"]
            ),

        "Eyebrow Slope Symmetry":
            abs(
                abs(left["slope"]) -
                abs(right["slope"])
            ),

        "Eyebrow Curvature Symmetry":
            abs(
                left["curve"] -
                right["curve"]
            )

    }