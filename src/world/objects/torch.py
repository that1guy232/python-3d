"""Building-mounted torch object and light placement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from pygame.math import Vector3

from engine.rendering.sprite import WorldSprite
from textures.resource_path import TORCH_TEXTURE_PATH
from textures.texture_utils import get_texture_size, load_texture


TORCH_LIGHT_VALUE = 3.4
TORCH_LIGHT_FALLOFF = 2.2
TORCH_LIGHT_MIN_RADIUS = 95.0
TORCH_LIGHT_MAX_RADIUS = 180.0
TORCH_FLOOR_LIGHT_SCALE = 0.28
TORCH_WALL_INSET = 8.0
TORCH_SPRITE_HEIGHT = 16.0
TORCH_MOUNT_HEIGHT = 28.0
TORCH_COLOR = (1.0, 0.82, 0.55)

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


def _coerce_vector3(value: Vector3 | Sequence[float]) -> Vector3:
    if isinstance(value, Vector3):
        return value
    if len(value) == 2:
        return Vector3(float(value[0]), 0.0, float(value[1]))
    return Vector3(float(value[0]), float(value[1]), float(value[2]))


@dataclass
class Torch(WorldSprite):
    """A textured torch billboard plus its matching brightness-area metadata."""

    brightness_modifier: dict[str, Any] | None = None

    @classmethod
    def texture_or_load(cls, texture: Any | None = None) -> Any:
        if texture:
            return texture
        return load_texture(TORCH_TEXTURE_PATH)

    @classmethod
    def side_for_building_spec(cls, spec: dict) -> str:
        doorway_side = str(spec.get("doorway_side", "south")).lower()
        return _OPPOSITE_SIDE.get(doorway_side, "north")

    @classmethod
    def light_xz_for_building_spec(cls, spec: dict) -> tuple[float, float]:
        position = spec["position"]
        side = cls.side_for_building_spec(spec)
        nx, nz = _SIDE_NORMALS[side]
        half_x = float(spec["width"]) * 0.5
        half_z = float(spec["depth"]) * 0.5
        x = float(position.x)
        z = float(position.z)
        if abs(nx) > 0.0:
            x += nx * max(0.0, half_x - TORCH_WALL_INSET)
        else:
            z += nz * max(0.0, half_z - TORCH_WALL_INSET)
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
    def brightness_modifier_for_building_spec(cls, spec: dict) -> dict[str, Any]:
        x, z = cls.light_xz_for_building_spec(spec)
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
    def brightness_modifiers_for_building_specs(
        cls, specs: Iterable[dict]
    ) -> list[dict[str, Any]]:
        modifiers: list[dict[str, Any]] = []
        for spec in specs or ():
            try:
                modifiers.append(cls.brightness_modifier_for_building_spec(spec))
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return modifiers

    @staticmethod
    def normalize_brightness_modifier(modifier: object) -> dict[str, Any]:
        if isinstance(modifier, dict):
            center = _coerce_vector3(modifier["center"])
            return {
                "center": center,
                "radius": float(modifier["radius"]),
                "value": float(modifier["value"]),
                "falloff": float(modifier.get("falloff", 1.0)),
                "bounds": modifier.get("bounds"),
                "indoor_only": bool(modifier.get("indoor_only", False)),
                "floor_scale": float(modifier.get("floor_scale", 1.0)),
            }

        values = list(modifier)  # type: ignore[arg-type]
        center = _coerce_vector3(values[0])
        return {
            "center": center,
            "radius": float(values[1]),
            "value": float(values[2]),
            "falloff": float(values[3]),
            "bounds": values[4] if len(values) > 4 else None,
            "indoor_only": False,
            "floor_scale": 1.0,
        }

    @staticmethod
    def install_brightness_modifier(camera: object, modifier: object) -> None:
        add_area = getattr(camera, "add_brightness_area", None)
        if not callable(add_area):
            return

        normalized = Torch.normalize_brightness_modifier(modifier)
        add_area(
            normalized["center"],
            normalized["radius"],
            normalized["value"],
            normalized["falloff"],
            bounds=normalized["bounds"],
            indoor_only=normalized["indoor_only"],
            floor_scale=normalized["floor_scale"],
        )

    @staticmethod
    def sprite_size_for_texture(
        texture: Any, *, sprite_height: float = TORCH_SPRITE_HEIGHT
    ) -> tuple[float, float]:
        tex_size = get_texture_size(texture)
        aspect = (tex_size[0] / tex_size[1]) if tex_size and tex_size[1] else (7.0 / 15.0)
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
        return cls(
            position=Vector3(center.x, floor_y + mount_height, center.z),
            size=size if size is not None else cls.sprite_size_for_texture(texture),
            texture=texture,
            camera=camera,
            color=color,
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
        texture = cls.texture_or_load(texture)
        if not texture:
            return []

        size = cls.sprite_size_for_texture(texture)
        torches: list[Torch] = []
        for modifier in modifiers or ():
            try:
                torches.append(
                    cls.from_brightness_modifier(
                        modifier,
                        texture=texture,
                        camera=camera,
                        ground_height_at=ground_height_at,
                        size=size,
                    )
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return torches
