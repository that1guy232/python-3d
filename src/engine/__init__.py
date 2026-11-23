"""Engine-level packages with an optional, lazily imported entity export."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.core.entity import Entity


def __getattr__(name: str):
    # Data-only services such as persistence must work without Pygame. Keep
    # the historical `from engine import Entity` API for graphics callers.
    if name == "Entity":
        from engine.core.entity import Entity

        globals()[name] = Entity
        return Entity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Entity"]
