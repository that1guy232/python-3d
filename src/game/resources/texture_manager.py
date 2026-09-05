"""Centralized texture loading for the world scene.

Provides a single function `load_world_textures()` which loads all textures
used by the world and returns a dict of named textures. Loading is safe if
files are missing (returns None for missing textures) and keeps the scene
initializer smaller and clearer.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
from engine.textures.texture_utils import (
    create_solid_color_texture,
    load_texture,
    load_texture_atlas,
)
from game.inventory import GOBLIN_SET_PLACEHOLDER_ICON
from game.resources.paths import (
    BODY_SLOT_TEXTURE_PATH,
    BOOT_SLOT_TEXTURE_PATH,
    CHEST_TEXTURE_PATH,
    DOOR_TEXTURE_PATH,
    FENCE1_TEXTURE_PATH,
    FENCE2_TEXTURE_PATH,
    FENCE3_TEXTURE_PATH,
    FENCE4_TEXTURE_PATH,
    FENCE5_TEXTURE_PATH,
    FENCE6_TEXTURE_PATH,
    GOBLIN_FIST_TEXTURE_PATH,
    GRASS_TEXTURE_PATH,
    HELMET_SLOT_TEXTURE_PATH,
    ROAD_TEXTURE_PATH,
    ROOF_TEXTURE_PATH,
    SWORD_TEXTURE_PATH,
    TORCH_FRAME_TEXTURE_PATHS,
    TORCH_TEXTURE_PATH,
    WALL1_TEXTURE_PATH,
    WEAPON_SLOT_TEXTURE_PATH,
    WINDOW_TEXTURE_PATH,
)


def load_world_textures() -> Dict[str, object]:
    """Load and return common world textures.

    Returns a dict with keys:
      - ground_tex, road_tex
      - fence_textures (list)
      - item_textures (dict)
      - equipment_slot_textures (dict)
      - wall_tex, roof_tex, chest_tex
      - torch_tex (animated frame regions when available)
      - door_tex
      - window_tex
      - goblin_material_tex (neutral material for generated goblin geometry)
    """
    # Core textures
    ground_tex = load_texture(GRASS_TEXTURE_PATH)
    road_tex = load_texture(ROAD_TEXTURE_PATH)

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
    goblin_fist_texture = load_texture(GOBLIN_FIST_TEXTURE_PATH)
    item_textures = {
        "sword_texture": load_texture(SWORD_TEXTURE_PATH),
        "goblin_fist": goblin_fist_texture,
        GOBLIN_SET_PLACEHOLDER_ICON: goblin_fist_texture,
    }
    equipment_slot_textures = {
        "body": load_texture(BODY_SLOT_TEXTURE_PATH),
        "boot": load_texture(BOOT_SLOT_TEXTURE_PATH),
        "weapon": load_texture(WEAPON_SLOT_TEXTURE_PATH),
        "helmet": load_texture(HELMET_SLOT_TEXTURE_PATH),
    }

    # Architectural and interactive-object textures
    wall_tex = load_texture(WALL1_TEXTURE_PATH)
    roof_tex = load_texture(ROOF_TEXTURE_PATH)
    building_material_tex = create_solid_color_texture()
    # Generated models carry their palette in vertex colors. Reuse the same
    # neutral material instead of loading the old directional sprite atlases.
    goblin_material_tex = building_material_tex
    chest_tex = load_texture(CHEST_TEXTURE_PATH)
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
    return {
        "ground_tex": ground_tex,
        "road_tex": road_tex,
        "fence_textures": fence_textures,
        "item_textures": item_textures,
        "equipment_slot_textures": equipment_slot_textures,
        "wall_tex": wall_tex,
        "roof_tex": roof_tex,
        "building_material_tex": building_material_tex,
        "chest_tex": chest_tex,
        "torch_tex": torch_tex,
        "door_tex": door_tex,
        "window_tex": window_tex,
        "goblin_material_tex": goblin_material_tex,
    }
