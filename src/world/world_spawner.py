"""Small helper to spawn billboard sprites inside a rectangular area.

`x_off`/`z_off` are the spawn-area center. Use `avoid_padding` to keep sprites
away from roads/areas when needed.

OPTIMIZED VERSION with performance improvements:
- Batch texture size lookups
- Pre-filter collision objects by bounding box
- Use spatial indexing for sprite-sprite collision
- Vectorized position generation
- Reduced function call overhead
"""

from __future__ import annotations

import random
import numpy as np
from pygame.math import Vector3
from world.sprite import WorldSprite
from textures.texture_utils import get_texture_size
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Optional
from collections import defaultdict


class SpatialGrid:
    """Simple spatial grid for fast collision detection between sprites."""
    
    def __init__(self, cell_size: float = 50.0):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
    
    def _get_cell_key(self, x: float, z: float) -> tuple[int, int]:
        return (int(x // self.cell_size), int(z // self.cell_size))
    
    def add_sprite(self, sprite, x: float, z: float, radius: float):
        """Add sprite to all cells it might overlap with."""
        # Get range of cells this sprite could occupy
        min_x = int((x - radius) // self.cell_size)
        max_x = int((x + radius) // self.cell_size)
        min_z = int((z - radius) // self.cell_size)
        max_z = int((z + radius) // self.cell_size)
        
        for cell_x in range(min_x, max_x + 1):
            for cell_z in range(min_z, max_z + 1):
                self.grid[(cell_x, cell_z)].append((sprite, x, z, radius))
    
    def check_collision(self, x: float, z: float, radius: float, padding: float) -> bool:
        """Check if position collides with any existing sprites."""
        cell_key = self._get_cell_key(x, z)
        
        # Check current cell and adjacent cells
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                check_cell = (cell_key[0] + dx, cell_key[1] + dz)
                for sprite, sx, sz, sr in self.grid.get(check_cell, []):
                    distance_sq = (sx - x) ** 2 + (sz - z) ** 2
                    required_distance = radius + sr + padding
                    if distance_sq < required_distance ** 2:
                        return True
        return False


def _prefilter_collision_objects(avoid_objects, x_center, z_center, max_spawn_x, max_spawn_z, margin):
    """Pre-filter collision objects to only those that could intersect the spawn area."""
    if not avoid_objects:
        return []
    
    # Spawn area bounds
    min_x = x_center - max_spawn_x - margin
    max_x = x_center + max_spawn_x + margin
    min_z = z_center - max_spawn_z - margin
    max_z = z_center + max_spawn_z + margin
    
    filtered = []
    for obj in avoid_objects:
        if obj is None:
            continue
        
        # Check if object has bounding box method
        if hasattr(obj, 'get_bounds'):
            # Expect get_bounds() -> (min_x, max_x, min_z, max_z)
            obj_bounds = obj.get_bounds()
            try:
                o_min_x, o_max_x, o_min_z, o_max_z = obj_bounds
            except Exception:
                # Malformed bounds: conservatively include
                filtered.append(obj)
                continue

            # Overlap test: object's X interval intersects spawn X interval AND
            # object's Z interval intersects spawn Z interval.
            if (o_min_x <= max_x and o_max_x >= min_x and o_min_z <= max_z and o_max_z >= min_z):
                filtered.append(obj)
        else:
            # Conservative: include all objects without bounds info
            filtered.append(obj)
    
    return filtered


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
    avoid_areas: list | None = None,
    avoid_padding: float = 10.0,
    max_retries: int = 6,
    use_batch_generation: bool = True,
) -> list[WorldSprite]:
    """Create billboard sprites randomly placed inside a rectangular area.

    OPTIMIZED version with several performance improvements.

    Parameters
    ----------
    scene: WorldScene-like
        Object providing `ground_height_at(x, z)`.
    count: int
        Number of sprites to generate.
    textures: list
        Sequence of texture identifiers (passed to `get_texture_size`).
    px_to_world: float
        Conversion factor from texture pixels to world units.
    camera: Camera
        Camera instance passed to each `WorldSprite`.
    x_off, z_off: float
        Spawn-area center coordinates.
    max_spawn_x, max_spawn_z: float
        Half-size of spawn area in X and Z directions.
    avoid_roads: list | None
        Optional sequence of road objects.
    avoid_areas: list | None
        Optional sequence of area objects.
    avoid_padding: float
        Extra distance to keep sprites away from roads/areas.
    max_retries: int
        Maximum attempts to find valid position per sprite.
    use_batch_generation: bool
        Whether to use batch position generation (faster for many sprites).

    Returns
    -------
    list[WorldSprite]
        Newly created sprites positioned on the ground.
    """
    # Pre-compute texture sizes to avoid repeated lookups
    texture_sizes = {}
    for tex in set(textures):  # Remove duplicates
        size_px = get_texture_size(tex)
        if size_px:
            w_px, h_px = size_px
            width = float(w_px) * px_to_world
            height = float(h_px) * px_to_world
        else:
            width = height = 16.0 * px_to_world
        texture_sizes[tex] = (width, height)

    # Pre-filter collision objects to only those that could affect spawn area
    pad_margin = max(0.0, float(avoid_padding))
    max_sprite_radius = max(max(w, h) for w, h in texture_sizes.values()) * 0.5
    total_margin = 2.0 + pad_margin + max_sprite_radius
    
    filtered_roads = _prefilter_collision_objects(
        avoid_roads, x_off, z_off, max_spawn_x, max_spawn_z, total_margin
    )
    filtered_areas = _prefilter_collision_objects(
        avoid_areas, x_off, z_off, max_spawn_x, max_spawn_z, total_margin
    )

    # Use spatial grid for sprite-sprite collision detection
    spatial_grid = SpatialGrid(cell_size=max_sprite_radius * 4)
    sprites: list[WorldSprite] = []

    if use_batch_generation and count > 20:
        # Generate positions in batches for better performance
        return _spawn_sprites_batch(
            scene, count, textures, texture_sizes, camera, x_off, z_off,
            max_spawn_x, max_spawn_z, filtered_roads, filtered_areas,
            pad_margin, max_retries, spatial_grid
        )
    else:
        # Use original sequential approach for small counts
        return _spawn_sprites_sequential(
            scene, count, textures, texture_sizes, camera, x_off, z_off,
            max_spawn_x, max_spawn_z, filtered_roads, filtered_areas,
            pad_margin, max_retries, spatial_grid
        )


def _spawn_sprites_sequential(
    scene, count, textures, texture_sizes, camera, x_off, z_off,
    max_spawn_x, max_spawn_z, filtered_roads, filtered_areas,
    pad_margin, max_retries, spatial_grid
):
    """Sequential sprite generation (original approach, optimized)."""
    sprites = []
    
    for _ in range(count):
        tex = random.choice(textures)
        width, height = texture_sizes[tex]
        half_w = width * 0.5
        
        # Calculate margins once per sprite
        road_margin = 2.0 + pad_margin + half_w
        area_margin = 2.0 + pad_margin + half_w

        # Try to find valid position
        for _tries in range(max_retries):
            x = random.uniform(-max_spawn_x, max_spawn_x) + x_off
            z = random.uniform(-max_spawn_z, max_spawn_z) + z_off

            # Check sprite-sprite collision using spatial grid
            if spatial_grid.check_collision(x, z, half_w, pad_margin):
                continue

            # Check road collisions (pre-filtered)
            if any(r.contains_point(x, z, margin=road_margin) for r in filtered_roads):
                continue

            # Check area collisions (pre-filtered)
            if any(a.contains_point(x, z, margin=area_margin) for a in filtered_areas):
                continue

            # Valid position found
            break
        
        # Create sprite
        y_center = scene.ground_height_at(x, z) + (height * 0.5)
        sprite = WorldSprite(
            position=Vector3(x, y_center, z),
            size=(width, height),
            texture=tex,
            camera=camera,
        )
        
        sprites.append(sprite)
        spatial_grid.add_sprite(sprite, x, z, half_w)

    return sprites


def _spawn_sprites_batch(
    scene, count, textures, texture_sizes, camera, x_off, z_off,
    max_spawn_x, max_spawn_z, filtered_roads, filtered_areas,
    pad_margin, max_retries, spatial_grid
):
    """Batch sprite generation for better performance with many sprites."""
    sprites = []
    
    # Generate more candidates than needed to account for rejections
    batch_size = min(count * 3, 1000)  # Generate 3x candidates, capped at 1000
    
    while len(sprites) < count:
        remaining = count - len(sprites)
        current_batch = min(batch_size, remaining * 2)
        
        # Generate batch of candidate positions
        x_candidates = np.random.uniform(-max_spawn_x, max_spawn_x, current_batch) + x_off
        z_candidates = np.random.uniform(-max_spawn_z, max_spawn_z, current_batch) + z_off
        
        # Generate corresponding textures and sizes
        tex_candidates = [random.choice(textures) for _ in range(current_batch)]
        
        # Process each candidate
        for i in range(current_batch):
            if len(sprites) >= count:
                break
                
            x, z = x_candidates[i], z_candidates[i]
            tex = tex_candidates[i]
            width, height = texture_sizes[tex]
            half_w = width * 0.5
            
            # Check all collision conditions
            if spatial_grid.check_collision(x, z, half_w, pad_margin):
                continue
                
            road_margin = 2.0 + pad_margin + half_w
            if any(r.contains_point(x, z, margin=road_margin) for r in filtered_roads):
                continue
                
            area_margin = 2.0 + pad_margin + half_w
            if any(a.contains_point(x, z, margin=area_margin) for a in filtered_areas):
                continue
            
            # Valid position - create sprite
            y_center = scene.ground_height_at(x, z) + (height * 0.5)
            sprite = WorldSprite(
                position=Vector3(x, y_center, z),
                size=(width, height),
                texture=tex,
                camera=camera,
            )
            
            sprites.append(sprite)
            spatial_grid.add_sprite(sprite, x, z, half_w)
    
    return sprites[:count]  # Return exactly the requested count