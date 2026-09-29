"""
Eye feature extraction.
"""

from .registry import register_feature



from .base import (
    distance,
    polygon_area,
    perimeter,
    circularity,
    solidity,
    aspect_ratio
)


def eye_measurements(region):

    pts = region["all_points"]

    width = distance(
        region["left_corner"],
        region["right_corner"]
    )

    height = distance(
        region["upper"],
        region["lower"]
    )

    area = polygon_area(pts)

    peri = perimeter(pts)

    return {

        "width": width,

        "height": height,

        "EAR": aspect_ratio(
            height,
            width
        ),

        "area": area,

        "perimeter": peri,

        "circularity":
            circularity(area, peri),

        "solidity":
            solidity(pts),

        "centre":
            pts.mean(axis=0)

    }




@register_feature("eyes")
def extract_features(
    landmarks,
    mapping
):

    left = eye_measurements(
        mapping["left_eye"]
    )

    right = eye_measurements(
        mapping["right_eye"]
    )

    return {

        "Left Eye Width":
            left["width"],

        "Right Eye Width":
            right["width"],

        "Left Eye Height":
            left["height"],

        "Right Eye Height":
            right["height"],

        "Left EAR":
            left["EAR"],

        "Right EAR":
            right["EAR"],

        "Left Eye Area":
            left["area"],

        "Right Eye Area":
            right["area"],

        "Left Eye Perimeter":
            left["perimeter"],

        "Right Eye Perimeter":
            right["perimeter"],

        "Left Eye Circularity":
            left["circularity"],

        "Right Eye Circularity":
            right["circularity"],

        "Left Eye Solidity":
            left["solidity"],

        "Right Eye Solidity":
            right["solidity"],

        "Eye Symmetry":
            abs(
                left["EAR"] -
                right["EAR"]
            ),

        "Inter-Eye Distance":
            distance(
                left["centre"],
                right["centre"]
            )

    }