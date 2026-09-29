"""
Mouth feature extraction.
"""


from .registry import register_feature


from .base import (
    distance,
    polygon_area,
    perimeter,
    circularity,
    solidity,
    aspect_ratio,
    quadratic_curvature
)


@register_feature("mouth")
def extract_features(
    landmarks,
    mapping
):

    mouth = mapping["mouth"]

    points = mouth["all_points"]

    width = distance(
        mouth["left_corner"],
        mouth["right_corner"]
    )

    opening = distance(
        mouth["upper_lip"],
        mouth["lower_lip"]
    )

    mar = aspect_ratio(
        opening,
        width
    )

    area = polygon_area(points)

    peri = perimeter(points)

    return {

        "Mouth Width": width,

        "Mouth Opening": opening,

        "MAR": mar,

        "Mouth Area": area,

        "Mouth Perimeter": peri,

        "Mouth Circularity":
            circularity(area, peri),

        "Mouth Solidity":
            solidity(points),

        "Upper Lip Curvature":
            quadratic_curvature(points)

    }