"""Outdoor world declarations and level entry point."""
from .content import BuildingSpec, WorldContent, building

__all__ = ["WorldScene", "WorldContent", "BuildingSpec", "building"]


def __getattr__(name):
    if name == "WorldScene":
        from .scene import WorldScene
        return WorldScene
    raise AttributeError(name)
