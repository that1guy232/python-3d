"""Lazy public exports for runtime game actors and behavior contracts."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CombatCreature": (".creature", "CombatCreature"),
    "CombatIntent": (".creature", "CombatIntent"),
    "Creature": (".creature", "Creature"),
    "Goblin": (".goblin", "Goblin"),
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
