"""Central asset path constants.

Paths are resolved from the repository root so the game can be launched from
outside the project directory without losing textures or sounds.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_PATH: str = str(PROJECT_ROOT / "assets")
TEXTURES_PATH: str = str(PROJECT_ROOT / "assets" / "textures")
SOUNDS_PATH: str = str(PROJECT_ROOT / "assets" / "sounds")


def _asset(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath("assets", *parts))


STAR_TEXTURE_PATH: str = _asset("textures", "star.png")
MOON_TEXTURE_PATH: str = _asset("textures", "moon.png")
GRASS_TEXTURE_PATH: str = _asset("textures", "grass.png")
ROAD_TEXTURE_PATH: str = _asset("textures", "road.png")

# Trees
TREE1_TEXTURE_PATH: str = _asset("textures", "trees", "tree1.png")
TREE2_TEXTURE_PATH: str = _asset("textures", "trees", "tree2.png")
TREE3_TEXTURE_PATH: str = _asset("textures", "trees", "tree3.png")
TREE4_TEXTURE_PATH: str = _asset("textures", "trees", "tree4.png")
TREE5_TEXTURE_PATH: str = _asset("textures", "trees", "tree5.png")
TREE6_TEXTURE_PATH: str = _asset("textures", "trees", "tree6.png")
TREE7_TEXTURE_PATH: str = _asset("textures", "trees", "tree7.png")
TREE8_TEXTURE_PATH: str = _asset("textures", "trees", "tree8.png")

# Grasses
GRASS1_TEXTURE_PATH: str = _asset("textures", "grasses", "grass1.png")
GRASS2_TEXTURE_PATH: str = _asset("textures", "grasses", "grass2.png")
GRASS3_TEXTURE_PATH: str = _asset("textures", "grasses", "grass3.png")
GRASS4_TEXTURE_PATH: str = _asset("textures", "grasses", "grass4.png")

# Rocks
ROCK1_TEXTURE_PATH: str = _asset("textures", "rocks", "rock1.png")
ROCK2_TEXTURE_PATH: str = _asset("textures", "rocks", "rock2.png")
ROCK3_TEXTURE_PATH: str = _asset("textures", "rocks", "rock3.png")
ROCK4_TEXTURE_PATH: str = _asset("textures", "rocks", "rock4.png")
ROCK5_TEXTURE_PATH: str = _asset("textures", "rocks", "rock5.png")
ROCK6_TEXTURE_PATH: str = _asset("textures", "rocks", "rock6.png")

# Fences
FENCE1_TEXTURE_PATH: str = _asset("textures", "fences", "fence1.png")
FENCE2_TEXTURE_PATH: str = _asset("textures", "fences", "fence2.png")
FENCE3_TEXTURE_PATH: str = _asset("textures", "fences", "fence3.png")
FENCE4_TEXTURE_PATH: str = _asset("textures", "fences", "fence4.png")
FENCE5_TEXTURE_PATH: str = _asset("textures", "fences", "fence5.png")
FENCE6_TEXTURE_PATH: str = _asset("textures", "fences", "fence6.png")

# Wall textures
WALL1_TEXTURE_PATH: str = _asset("textures", "wall1.png")
TORCH_TEXTURE_PATH: str = _asset("textures", "torch.png")
TORCH_TEXTURE_DIR_PATH: str = _asset("textures", "torches")
TORCH_FRAME_TEXTURE_PATHS: tuple[str, ...] = tuple(
    _asset("textures", "torches", f"torch{i}.png") for i in range(1, 6)
)
DOOR_TEXTURE_PATH: str = _asset("textures", "door.png")
WINDOW_TEXTURE_PATH: str = _asset("textures", "window.png")
GOBLIN_FRONT_TEXTURE_DIR_PATH: str = _asset("textures", "goblin", "front")
GOBLIN_RIGHT_TEXTURE_DIR_PATH: str = _asset("textures", "goblin", "right")
GOBLIN_BACK_TEXTURE_DIR_PATH: str = _asset("textures", "goblin", "back")

# Sounds
BIRDS_SOUND_PATH: str = _asset("sounds", "birds.ogg")
LEAVES02_SOUND_PATH: str = _asset("sounds", "leaves02.ogg")
STEP1_SOUND_PATH: str = _asset("sounds", "step.ogg")

# Items
SWORD_TEXTURE_PATH: str = _asset("textures", "items", "sword.png")
COMPASS_BASE_TEXTURE_PATH: str = _asset("textures", "items", "compass_base.png")
COMPASS_NEEDLE_TEXTURE_PATH: str = _asset("textures", "items", "compass_needle.png")

SHADOW_TEXTURE_PATH: str = _asset("textures", "shadow.png")
LIGHT_TEXTURE_PATH: str = _asset("textures", "light.png")
