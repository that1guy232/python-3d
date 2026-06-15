"""Building-mounted torch object and light placement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from pygame.math import Vector3

from engine.rendering.lighting import (
    install_brightness_modifier_on_camera,
    normalize_brightness_modifier,
)
from engine.rendering.sprite import AnimatedWorldSprite
from game.resources.paths import TORCH_FRAME_TEXTURE_PATHS, TORCH_TEXTURE_PATH
from engine.textures.texture_utils import (
    get_texture_size,
    load_texture,
    load_texture_atlas,
)

TORCH_LIGHT_VALUE = 3.4
TORCH_LIGHT_FALLOFF = 2.2
TORCH_LIGHT_MIN_RADIUS = 95.0
TORCH_LIGHT_MAX_RADIUS = 180.0
TORCH_FLOOR_LIGHT_SCALE = 0.28
TORCH_WALL_INSET = 8.0
TORCH_SPRITE_HEIGHT = 16.0
TORCH_MOUNT_HEIGHT = 28.0
TORCH_COLOR = (1.0, 0.82, 0.55)
TORCH_ANIMATION_FPS = 9.0
TORCH_ANIMATION_FRAME_DURATION = 1.0 / TORCH_ANIMATION_FPS

_SIDE_NORMALS = {
    "north": (0.0, 1.0),
    "east": (1.0, 0.0),
    "south": (0.0, -1.0),
    "west": (-1.0, 0.0),
}
_OPPOSITE_SIDE = {
    "north": "south",
    "east": "west",
    "south": "north",
    "west": "east",
}


@dataclass
class Torch(AnimatedWorldSprite):
    """A textured torch billboard plus its matching brightness-area metadata."""

    brightness_modifier: dict[str, Any] | None = None

    @classmethod
    def animation_frames(cls, texture: Any | None = None) -> tuple[Any, ...]:
        if isinstance(texture, (list, tuple)):
            return tuple(frame for frame in texture if frame)
        return (texture,) if texture else ()

    @classmethod
    def texture_or_load(cls, texture: Any | None = None) -> tuple[Any, ...]:
        frames = cls.animation_frames(texture)
        if frames:
            return frames

        frame_paths = [
            path for path in TORCH_FRAME_TEXTURE_PATHS if Path(path).is_file()
        ]
        if frame_paths:
            frames = tuple(load_texture_atlas(frame_paths))
            if frames:
                return frames

        fallback = load_texture(TORCH_TEXTURE_PATH)
        return (fallback,) if fallback else ()

    @classmethod
    def side_for_building_spec(cls, spec: dict) -> str:
        doorway_side = str(spec.get("doorway_side", "south")).lower()
        return _OPPOSITE_SIDE.get(doorway_side, "north")

    @classmethod
    def light_xz_for_building_spec(cls, spec: dict) -> tuple[float, float]:
        return cls.light_xz_for_building_torch_spec(spec)

    @classmethod
    def light_xz_for_building_torch_spec(
        cls,
        spec: dict,
        torch_spec: dict | None = None,
    ) -> tuple[float, float]:
        position = spec["position"]
        torch_spec = torch_spec if isinstance(torch_spec, dict) else {}
        fallback_side = cls.side_for_building_spec(spec)
        side = str(torch_spec.get("side", fallback_side)).lower()
        if side not in _SIDE_NORMALS:
            side = fallback_side
        nx, nz = _SIDE_NORMALS[side]
        half_x = float(spec["width"]) * 0.5
        half_z = float(spec["depth"]) * 0.5
        x = float(position.x)
        z = float(position.z)
        offset = float(torch_spec.get("offset", 0.0))
        wall_inset = max(0.0, float(torch_spec.get("wall_inset", TORCH_WALL_INSET)))
        if abs(nx) > 0.0:
            x += nx * max(0.0, half_x - wall_inset)
            z += offset
        else:
            x += offset
            z += nz * max(0.0, half_z - wall_inset)
        return x, z

    @classmethod
    def light_bounds_for_building_spec(
        cls, spec: dict
    ) -> tuple[float, float, float, float]:
        position = spec["position"]
        x = float(position.x)
        z = float(position.z)
        half_x = max(0.0, float(spec["width"]) * 0.5 - 2.0)
        half_z = max(0.0, float(spec["depth"]) * 0.5 - 2.0)
        return (x - half_x, x + half_x, z - half_z, z + half_z)

    @classmethod
    def light_radius_for_building_spec(cls, spec: dict) -> float:
        short_side = min(float(spec["width"]), float(spec["depth"]))
        return max(
            TORCH_LIGHT_MIN_RADIUS,
            min(TORCH_LIGHT_MAX_RADIUS, short_side * 0.9),
        )

    @classmethod
    def brightness_modifier_for_building_spec(
        cls,
        spec: dict,
        torch_spec: dict | None = None,
    ) -> dict[str, Any]:
        x, z = cls.light_xz_for_building_torch_spec(spec, torch_spec)
        return {
            "center": Vector3(x, 0.0, z),
            "radius": cls.light_radius_for_building_spec(spec),
            "value": TORCH_LIGHT_VALUE,
            "falloff": TORCH_LIGHT_FALLOFF,
            "bounds": cls.light_bounds_for_building_spec(spec),
            "indoor_only": True,
            "floor_scale": TORCH_FLOOR_LIGHT_SCALE,
        }

    @classmethod
    def brightness_modifiers_for_building_spec(
        cls,
        spec: dict,
    ) -> list[dict[str, Any]]:
        torch_specs = spec.get("torches", None)
        if not isinstance(torch_specs, (list, tuple)):
            return [cls.brightness_modifier_for_building_spec(spec)]

        modifiers: list[dict[str, Any]] = []
        for torch_spec in torch_specs:
            if not isinstance(torch_spec, dict):
                continue
            modifiers.append(
                cls.brightness_modifier_for_building_spec(spec, torch_spec)
            )
        return modifiers

    @classmethod
    def brightness_modifiers_for_building_specs(
        cls, specs: Iterable[dict]
    ) -> list[dict[str, Any]]:
        modifiers: list[dict[str, Any]] = []
        for spec in specs or ():
            try:
                modifiers.extend(cls.brightness_modifiers_for_building_spec(spec))
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return modifiers

    @staticmethod
    def normalize_brightness_modifier(modifier: object) -> dict[str, Any]:
        return normalize_brightness_modifier(modifier)

    @staticmethod
    def install_brightness_modifier(camera: object, modifier: object) -> None:
        install_brightness_modifier_on_camera(camera, modifier)

    @staticmethod
    def sprite_size_for_texture(
        texture: Any, *, sprite_height: float = TORCH_SPRITE_HEIGHT
    ) -> tuple[float, float]:
        tex_size = get_texture_size(texture)
        aspect = (
            (tex_size[0] / tex_size[1]) if tex_size and tex_size[1] else (7.0 / 15.0)
        )
        return (sprite_height * aspect, sprite_height)

    @classmethod
    def from_brightness_modifier(
        cls,
        modifier: object,
        *,
        texture: Any,
        camera: object,
        ground_height_at: Callable[[float, float], float],
        size: tuple[float, float] | None = None,
        mount_height: float = TORCH_MOUNT_HEIGHT,
        color: tuple[float, float, float] = TORCH_COLOR,
    ) -> "Torch":
        normalized = cls.normalize_brightness_modifier(modifier)
        center = normalized["center"]
        floor_y = float(ground_height_at(center.x, center.z))
        frames = cls.animation_frames(texture)
        if not frames:
            raise ValueError("Torch requires at least one texture frame")
        frame_index = int(abs(center.x * 0.19 + center.z * 0.31)) % len(frames)
        return cls(
            position=Vector3(center.x, floor_y + mount_height, center.z),
            size=size if size is not None else cls.sprite_size_for_texture(frames[0]),
            texture=frames[0],
            camera=camera,
            color=color,
            frames=frames,
            frame_duration=TORCH_ANIMATION_FRAME_DURATION,
            frame_index=frame_index,
            brightness_modifier=normalized,
        )

    @classmethod
    def build_for_brightness_modifiers(
        cls,
        modifiers: Iterable[object],
        *,
        texture: Any | None = None,
        camera: object,
        ground_height_at: Callable[[float, float], float],
    ) -> list["Torch"]:
        frames = cls.texture_or_load(texture)
        if not frames:
            return []

        size = cls.sprite_size_for_texture(frames[0])
        torches: list[Torch] = []
        for modifier in modifiers or ():
            try:
                torches.append(
                    cls.from_brightness_modifier(
                        modifier,
                        texture=frames,
                        camera=camera,
                        ground_height_at=ground_height_at,
                        size=size,
                    )
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return torches
