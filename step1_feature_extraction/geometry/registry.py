"""
Plugin registry for geometry feature extractors.
"""

FEATURE_REGISTRY = {}


def register_feature(name):
    """
    Decorator used to register a geometry
    feature extractor.
    """

    def decorator(function):

        if name in FEATURE_REGISTRY:

            raise ValueError(
                f"Feature extractor '{name}' already exists."
            )

        FEATURE_REGISTRY[name] = function

        return function

    return decorator


def get_registered_features():

    return FEATURE_REGISTRY.copy()


def available_features():

    return sorted(FEATURE_REGISTRY.keys())