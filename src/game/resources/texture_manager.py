"""Centralized texture loading for the world scene.

Provides a single function `load_world_textures()` which loads all textures
used by the world and returns a dict of named textures. Loading is safe if
files are missing (returns None for missing textures) and keeps the scene
initializer smaller and clearer.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
from engine.textures.texture_utils import load_texture, load_texture_atlas
from game.resources.paths import *


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.lower())


def _png_sequence_paths(directory: str) -> list[str]:
    frame_dir = Path(directory)
    if not frame_dir.is_dir():
        return []
    return [str(path) for path in sorted(frame_dir.glob("*.png"), key=_frame_sort_key)]


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
      - torch_tex (animated frame regions when available)
      - door_tex
      - window_tex
      - goblin_tex (front/right/back animated frame regions when available)
    """
    # Core textures
    ground_tex = load_texture(GRASS_TEXTURE_PATH)
    road_tex = load_texture(ROAD_TEXTURE_PATH)

    # Flora / rocks share one atlas so depth-sorted world sprites can still
    # render as one texture run.
    tree_paths = [
        TREE1_TEXTURE_PATH,
        TREE2_TEXTURE_PATH,
        TREE3_TEXTURE_PATH,
        TREE4_TEXTURE_PATH,
        TREE5_TEXTURE_PATH,
        TREE6_TEXTURE_PATH,
        TREE7_TEXTURE_PATH,
    ]
    grass_paths = [
        GRASS1_TEXTURE_PATH,
        GRASS2_TEXTURE_PATH,
        GRASS3_TEXTURE_PATH,
        GRASS4_TEXTURE_PATH,
    ]
    rock_paths = [
        ROCK1_TEXTURE_PATH,
        ROCK2_TEXTURE_PATH,
        ROCK3_TEXTURE_PATH,
        ROCK4_TEXTURE_PATH,
        ROCK5_TEXTURE_PATH,
        ROCK6_TEXTURE_PATH,
    ]
    sprite_regions = load_texture_atlas(tree_paths + grass_paths + rock_paths)
    tree_textures = sprite_regions[: len(tree_paths)]
    grasses_textures = sprite_regions[
        len(tree_paths) : len(tree_paths) + len(grass_paths)
    ]
    rock_textures = sprite_regions[len(tree_paths) + len(grass_paths) :]

    # Fences remain separate static mesh textures.

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
    torch_frame_paths = [
        path for path in TORCH_FRAME_TEXTURE_PATHS if Path(path).is_file()
    ]
    torch_tex = (
        load_texture_atlas(torch_frame_paths)
        if torch_frame_paths
        else load_texture(TORCH_TEXTURE_PATH)
    )
    door_tex = load_texture(DOOR_TEXTURE_PATH)
    window_tex = load_texture(WINDOW_TEXTURE_PATH)
    goblin_tex = {
        "front": load_texture_atlas(_png_sequence_paths(GOBLIN_FRONT_TEXTURE_DIR_PATH)),
        "right": load_texture_atlas(_png_sequence_paths(GOBLIN_RIGHT_TEXTURE_DIR_PATH)),
        "back": load_texture_atlas(_png_sequence_paths(GOBLIN_BACK_TEXTURE_DIR_PATH)),
    }

    return {
        "ground_tex": ground_tex,
        "road_tex": road_tex,
        "tree_textures": tree_textures,
        "grasses_textures": grasses_textures,
        "rock_textures": rock_textures,
        "fence_textures": fence_textures,
        "item_textures": item_textures,
        "wall_tex": wall_tex,
        "torch_tex": torch_tex,
        "door_tex": door_tex,
        "window_tex": window_tex,
        "goblin_tex": goblin_tex,
    }

