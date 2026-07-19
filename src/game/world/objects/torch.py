"""Building-mounted torch object and light placement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from pygame.math import Vector3

from engine.rendering.lighting import (
    install_local_light_on_camera,
    normalize_brightness_modifier,
)
from engine.rendering.lighting_state import LocalBrightnessLight, PointLight
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
TORCH_POINT_INTENSITY = 2.4
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
    emissive: bool = True

    @classmethod
    def mount_height_for_building_spec(
        cls,
        spec: dict,
        torch_spec: dict | None = None,
    ) -> float:
        """Resolve a wall-centered mount that keeps the sprite above the floor."""

        torch_spec = torch_spec if isinstance(torch_spec, dict) else {}
        if "mount_height" in torch_spec:
            return float(torch_spec["mount_height"])
        requested = float(spec.get("height", TORCH_MOUNT_HEIGHT * 2.0)) * 0.5
        wall_height = max(TORCH_SPRITE_HEIGHT, float(spec.get("height", 0.0)))
        half_sprite = TORCH_SPRITE_HEIGHT * 0.5
        clearance = 2.0
        minimum = half_sprite + clearance
        maximum = max(minimum, wall_height - half_sprite - clearance)
        return max(minimum, min(maximum, requested))

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
        *,
        light_id: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility projection of typed torch-light authoring."""

        return normalize_brightness_modifier(
            cls.local_light_for_building_spec(
                spec,
                torch_spec,
                light_id=light_id,
            )
        )

    @classmethod
    def local_light_for_building_spec(
        cls,
        spec: dict,
        torch_spec: dict | None = None,
        *,
        light_id: str | None = None,
    ) -> LocalBrightnessLight:
        """Author one typed legacy-scalar light for a building torch."""

        x, z = cls.light_xz_for_building_torch_spec(spec, torch_spec)
        return LocalBrightnessLight(
            light_id=str(light_id or f"torch:{x:.3f}:{z:.3f}"),
            center=(x, 0.0, z),
            radius=cls.light_radius_for_building_spec(spec),
            value=TORCH_LIGHT_VALUE,
            falloff=TORCH_LIGHT_FALLOFF,
            bounds=cls.light_bounds_for_building_spec(spec),
            indoor_only=True,
            floor_scale=TORCH_FLOOR_LIGHT_SCALE,
        )

    @classmethod
    def point_light_for_building_spec(
        cls,
        spec: dict,
        torch_spec: dict | None = None,
        *,
        light_id: str | None = None,
    ) -> PointLight:
        """Author one generic XYZ point light for a torch prefab."""

        torch_spec = torch_spec if isinstance(torch_spec, dict) else {}
        x, z = cls.light_xz_for_building_torch_spec(spec, torch_spec)
        base_y = float(spec.get("base_y", getattr(spec.get("position"), "y", 0.0)))
        mount_height = cls.mount_height_for_building_spec(spec, torch_spec)
        raw_color = torch_spec.get("light_color", TORCH_COLOR)
        color = tuple(float(value) for value in raw_color)
        if len(color) != 3:
            color = TORCH_COLOR
        return PointLight(
            light_id=str(light_id or f"torch:{x:.3f}:{z:.3f}"),
            position=(x, base_y + mount_height, z),
            color=(float(color[0]), float(color[1]), float(color[2])),
            intensity=max(
                0.0,
                float(torch_spec.get("light_intensity", TORCH_POINT_INTENSITY)),
            ),
            range=max(
                1.0,
                float(
                    torch_spec.get(
                        "light_range",
                        cls.light_radius_for_building_spec(spec),
                    )
                ),
            ),
            casts_shadows=bool(torch_spec.get("casts_shadows", True)),
            importance=max(0.0, float(torch_spec.get("light_importance", 1.0))),
        )

    @classmethod
    def brightness_modifiers_for_building_spec(
        cls,
        spec: dict,
        *,
        building_index: int | None = None,
    ) -> list[dict[str, Any]]:
        return [
            normalize_brightness_modifier(light)
            for light in cls.local_lights_for_building_spec(
                spec,
                building_index=building_index,
            )
        ]

    @classmethod
    def local_lights_for_building_spec(
        cls,
        spec: dict,
        *,
        building_index: int | None = None,
    ) -> list[LocalBrightnessLight]:
        torch_specs = spec.get("torches", None)
        if not isinstance(torch_specs, (list, tuple)):
            return [
                cls.local_light_for_building_spec(
                    spec,
                    light_id=(
                        f"building:{building_index}:torch:0"
                        if building_index is not None
                        else None
                    ),
                )
            ]

        lights: list[LocalBrightnessLight] = []
        for torch_index, torch_spec in enumerate(torch_specs):
            if not isinstance(torch_spec, dict):
                continue
            lights.append(
                cls.local_light_for_building_spec(
                    spec,
                    torch_spec,
                    light_id=(
                        f"building:{building_index}:torch:{torch_index}"
                        if building_index is not None
                        else None
                    ),
                )
            )
        return lights

    @classmethod
    def brightness_modifiers_for_building_specs(
        cls, specs: Iterable[dict]
    ) -> list[dict[str, Any]]:
        return [
            normalize_brightness_modifier(light)
            for light in cls.local_lights_for_building_specs(specs)
        ]

    @classmethod
    def local_lights_for_building_specs(
        cls,
        specs: Iterable[dict],
    ) -> list[LocalBrightnessLight]:
        lights: list[LocalBrightnessLight] = []
        for building_index, spec in enumerate(specs or ()):
            try:
                lights.extend(
                    cls.local_lights_for_building_spec(
                        spec,
                        building_index=building_index,
                    )
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return lights

    @classmethod
    def point_lights_for_building_spec(
        cls,
        spec: dict,
        *,
        building_index: int | None = None,
    ) -> list[PointLight]:
        torch_specs = spec.get("torches", None)
        if not isinstance(torch_specs, (list, tuple)):
            return [
                cls.point_light_for_building_spec(
                    spec,
                    light_id=(
                        f"building:{building_index}:torch:0"
                        if building_index is not None
                        else None
                    ),
                )
            ]
        lights: list[PointLight] = []
        for torch_index, torch_spec in enumerate(torch_specs):
            if not isinstance(torch_spec, dict):
                continue
            lights.append(
                cls.point_light_for_building_spec(
                    spec,
                    torch_spec,
                    light_id=(
                        f"building:{building_index}:torch:{torch_index}"
                        if building_index is not None
                        else None
                    ),
                )
            )
        return lights

    @classmethod
    def point_lights_for_building_specs(
        cls,
        specs: Iterable[dict],
    ) -> list[PointLight]:
        lights: list[PointLight] = []
        for building_index, spec in enumerate(specs or ()):
            try:
                lights.extend(
                    cls.point_lights_for_building_spec(
                        spec,
                        building_index=building_index,
                    )
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return lights

    @staticmethod
    def normalize_brightness_modifier(modifier: object) -> dict[str, Any]:
        return normalize_brightness_modifier(modifier)

    @staticmethod
    def install_local_light(camera: object, light: object) -> None:
        install_local_light_on_camera(camera, light)

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
    def from_point_light(
        cls,
        light: PointLight,
        *,
        texture: Any,
        camera: object,
        size: tuple[float, float] | None = None,
    ) -> "Torch":
        """Build the visible torch paired with a typed raster point light."""

        if not isinstance(light, PointLight):
            raise TypeError("Torch point-light visuals require PointLight records")
        frames = cls.animation_frames(texture)
        if not frames:
            raise ValueError("Torch requires at least one texture frame")
        x, y, z = (float(value) for value in light.position)
        frame_index = int(abs(x * 0.19 + z * 0.31)) % len(frames)
        return cls(
            position=Vector3(x, y, z),
            size=size if size is not None else cls.sprite_size_for_texture(frames[0]),
            texture=frames[0],
            camera=camera,
            color=tuple(float(value) for value in light.color),
            frames=frames,
            frame_duration=TORCH_ANIMATION_FRAME_DURATION,
            frame_index=frame_index,
            brightness_modifier=None,
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

    @classmethod
    def build_for_point_lights(
        cls,
        lights: Iterable[PointLight],
        *,
        texture: Any | None = None,
        camera: object,
    ) -> list["Torch"]:
        frames = cls.texture_or_load(texture)
        if not frames:
            return []
        size = cls.sprite_size_for_texture(frames[0])
        torches: list[Torch] = []
        for light in lights or ():
            try:
                torches.append(
                    cls.from_point_light(
                        light,
                        texture=frames,
                        camera=camera,
                        size=size,
                    )
                )
            except (TypeError, ValueError, AttributeError):
                continue
        return torches
