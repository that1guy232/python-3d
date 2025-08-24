"""World package: re-export common symbols for simpler imports.

Callers can import public types from `world` directly, e.g.:

    from world import WorldScene, WorldSprite, Decal

This file intentionally keeps the public surface small and stable while the
implementation files remain under `world/*.py`.
"""

from .worldscene import WorldScene
from .sprite import WorldSprite, draw_sprites_batched
from .decal import Decal
from .decal_batch import DecalBatch
from .ground_tile import GroundTile
from .objects import WallTile, Road
from .world_spawner import spawn_world_sprites
from .world_hud import WorldHUD
from .world_shade_overlay import WorldShadeOverlay
from .world_collision import movement_blocked_by_wall

__all__ = [
    "WorldScene",
    "WorldSprite",
    "draw_sprites_batched",
    "Decal",
    "DecalBatch",
    "GroundTile",
    "WallTile",
    "Road",
    "spawn_world_sprites",
    "WorldHUD",
    "WorldShadeOverlay",
    "movement_blocked_by_wall",
]
