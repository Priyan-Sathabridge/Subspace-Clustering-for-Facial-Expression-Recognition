"""
Shared geometric utility functions.
"""

import numpy as np
from scipy.spatial import ConvexHull


EPS = 1e-8


def distance(p1, p2):
    return np.linalg.norm(
        np.asarray(p1) - np.asarray(p2)
    )


def polygon_area(points):

    points = np.asarray(points)

    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )


def perimeter(points):

    points = np.asarray(points)

    return np.sum(

        np.linalg.norm(

            points -
            np.roll(points, -1, axis=0),

            axis=1

        )

    )


def circularity(area, perimeter_value):

    return (

        4 * np.pi * area /

        (perimeter_value**2 + EPS)

    )


def solidity(points):

    hull = ConvexHull(points)

    return (

        polygon_area(points)

        /

        (hull.volume + EPS)

    )


def aspect_ratio(height, width):

    return height / (width + EPS)


def quadratic_curvature(points):

    points = np.asarray(points)

    points = points[
        np.argsort(points[:, 0])
    ]

    x = points[:, 0]

    y = points[:, 1]

    x = x - np.mean(x)

    coeff = np.polyfit(x, y, 2)

    return coeff[0]