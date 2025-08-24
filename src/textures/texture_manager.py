"""Centralized texture loading for the world scene.

Provides a single function `load_world_textures()` which loads all textures
used by the world and returns a dict of named textures. Loading is safe if
files are missing (returns None for missing textures) and keeps the scene
initializer smaller and clearer.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from textures.texture_utils import load_texture, get_texture_size
from textures.resoucepath import *


def load_world_textures() -> Dict[str, object]:
    """Load and return common world textures.

    Returns a dict with keys:
      - ground_tex, road_tex
      - tree_textures (list)
      - grasses_textures (list)
      - rock_textures (list)
      - fence_textures (list)
      - item_textures (dict)
      - wall_tex
    """
    # Core textures
    ground_tex = load_texture(GRASS_TEXTURE_PATH)
    road_tex = load_texture(ROAD_TEXTURE_PATH)

    # Flora / rocks / fences
    tree_textures = [
        load_texture(TREE1_TEXTURE_PATH),
        load_texture(TREE2_TEXTURE_PATH),
        load_texture(TREE3_TEXTURE_PATH),
        load_texture(TREE4_TEXTURE_PATH),
        load_texture(TREE5_TEXTURE_PATH),
        load_texture(TREE6_TEXTURE_PATH),
        load_texture(TREE7_TEXTURE_PATH),
    ]

    grasses_textures = [
        load_texture(GRASS1_TEXTURE_PATH),
        load_texture(GRASS2_TEXTURE_PATH),
        load_texture(GRASS3_TEXTURE_PATH),
        load_texture(GRASS4_TEXTURE_PATH),
    ]

    rock_textures = [
        load_texture(ROCK1_TEXTURE_PATH),
        load_texture(ROCK2_TEXTURE_PATH),
        load_texture(ROCK3_TEXTURE_PATH),
        load_texture(ROCK4_TEXTURE_PATH),
        load_texture(ROCK5_TEXTURE_PATH),
        load_texture(ROCK6_TEXTURE_PATH),
    ]

    fence_textures = [
        load_texture(FENCE1_TEXTURE_PATH),
        load_texture(FENCE2_TEXTURE_PATH),
        load_texture(FENCE3_TEXTURE_PATH),
        load_texture(FENCE4_TEXTURE_PATH),
        load_texture(FENCE5_TEXTURE_PATH),
        load_texture(FENCE6_TEXTURE_PATH),
    ]

    # Item textures
    item_textures = {
        "sword_texture": load_texture(SWORD_TEXTURE_PATH),
    }

    # Wall texture
    wall_tex = load_texture(WALL1_TEXTURE_PATH)

    return {
        "ground_tex": ground_tex,
        "road_tex": road_tex,
        "tree_textures": tree_textures,
        "grasses_textures": grasses_textures,
        "rock_textures": rock_textures,
        "fence_textures": fence_textures,
        "item_textures": item_textures,
        "wall_tex": wall_tex,
    }
