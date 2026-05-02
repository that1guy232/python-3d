"""World package public exports.

Exports are resolved lazily so low-level modules can import each other without
loading the full world scene as a side effect of importing the package.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "WorldScene": (".worldscene", "WorldScene"),
    "WorldSprite": (".sprite", "WorldSprite"),
    "draw_sprites_batched": (".sprite", "draw_sprites_batched"),
    "Decal": (".decal", "Decal"),
    "DecalBatch": (".decal_batch", "DecalBatch"),
    "GroundTile": (".ground_tile", "GroundTile"),
    "WallTile": (".objects", "WallTile"),
    "Road": (".objects", "Road"),
    "spawn_world_sprites": (".world_spawner", "spawn_world_sprites"),
    "WorldHUD": (".world_hud", "WorldHUD"),
    "movement_blocked_by_wall": (".world_collision", "movement_blocked_by_wall"),
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
