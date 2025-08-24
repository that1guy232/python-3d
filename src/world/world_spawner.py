"""Helper to spawn world sprites moved out of WorldScene to reduce file size.

This module exposes a single function `spawn_world_sprites(scene, ...)` which uses
`scene.ground_height_at` to compute sprite heights so it stays simple to call.
"""

from __future__ import annotations

import random
from typing import List
from pygame.math import Vector3
from world.sprite import WorldSprite
from textures.texture_utils import get_texture_size


def spawn_world_sprites(
    scene,
    *,
    count: int,
    textures: list,
    px_to_world: float,
    camera,
    x_off: float,
    z_off: float,
    max_spawn_x: float,
    max_spawn_z: float,
    avoid_roads: list | None = None,
    # avoid_areas: list of objects with contains_point(x,z, margin=0.0) -> bool
    avoid_areas: list | None = None,
) -> list[WorldSprite]:
    """Create a list of billboard sprites randomly placed in the area.

    This is a near drop-in extract of WorldScene._spawn_world_sprites so call sites
    can be converted with minimal changes.
    """
    sprites: list[WorldSprite] = []
    for _ in range(count):
        # Try up to a few times to find a position not on a road
        for _tries in range(6):
            x = random.uniform(-max_spawn_x, max_spawn_x) + x_off
            z = random.uniform(-max_spawn_z, max_spawn_z) + z_off
            if not avoid_roads:
                break
            # Check roads first
            if any(
                r.contains_point(x, z, margin=2.0) for r in avoid_roads if r is not None
            ):
                # reject and retry
                continue
            # Check avoid_areas (e.g., building footprints)
            if avoid_areas:
                blocked = False
                for a in avoid_areas:
                    try:
                        if (
                            a is not None
                            and hasattr(a, "contains_point")
                            and a.contains_point(x, z, margin=2.0)
                        ):
                            blocked = True
                            break
                    except Exception:
                        pass
                if blocked:
                    continue
            break
        tex = random.choice(textures)
        # Compute world size from actual texture pixel size
        size_px = get_texture_size(tex)
        if size_px:
            w_px, h_px = size_px
            width = float(w_px) * px_to_world
            height = float(h_px) * px_to_world
        else:
            # Fallback to a small square if size unknown
            width = height = 16.0 * px_to_world
        # Set center.y so bottom sits on ground: center_y = ground_y + height/2
        y_center = scene.ground_height_at(x, z) + (height * 0.5)
        sprites.append(
            WorldSprite(
                position=Vector3(x, y_center, z),
                size=(width, height),
                texture=tex,
                camera=camera,
            )
        )
    return sprites
