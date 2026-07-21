"""Lazy public exports for player movement and combat state."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "PlayerCameraController": (".controller", "PlayerCameraController"),
    "PlayerStats": (".stats", "PlayerStats"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
