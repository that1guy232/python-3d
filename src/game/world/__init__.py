"""World declarations and the playable world scene."""

from .content import BuildingSpec, WorldContent, building

__all__ = ["WorldScene", "BuildingSpec", "WorldContent", "building"]


def __getattr__(name):
    if name == "WorldScene":
        from .scene import WorldScene
        return WorldScene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
