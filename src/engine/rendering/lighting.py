"""Shared lighting helpers for world rendering.

The fixed-function renderers in this project bake most lighting into vertex
colors, so keeping the sun math here helps every mesh agree on the same source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
from pygame.math import Vector3


_EPSILON = 1e-8
INDOOR_LIGHT_FACTOR = 0.34
INDOOR_NORMAL = (0.0, -1.0, 0.0)


@dataclass
class SceneLighting:
    """Shared lighting state for sun, indoor dimming, and local light areas.

    ``sun_direction`` follows the existing scene convention: it points from the
    sun toward the world. Surface lighting uses the inverse direction.
    """

    sun_position: Vector3
    sun_target: Vector3
    sky_color: tuple[float, float, float, float]
    ambient: float = 0.72
    diffuse: float = 0.48
    max_factor: float = 1.15
    sun_tint: tuple[float, float, float] = (1.0, 0.96, 0.86)
    base_brightness: float = 1.0
    brightness_modifiers: list[dict[str, Any]] = field(default_factory=list)
    covered_regions: list[object] = field(default_factory=list)

    @classmethod
    def from_world_center(
        cls,
        world_center: Vector3,
        *,
        sky_color: tuple[float, float, float, float],
        sun_offset: Vector3 | Sequence[float] = (36000.0, 22000.0, 18000.0),
        base_brightness: float = 1.0,
    ) -> "SceneLighting":
        offset = _as_vector3(sun_offset, Vector3(36000.0, 22000.0, 18000.0))
        target = Vector3(float(world_center.x), 0.0, float(world_center.z))
        position = Vector3(
            float(world_center.x) + offset.x,
            float(world_center.y) + offset.y,
            float(world_center.z) + offset.z,
        )
        return cls(
            sun_position=position,
            sun_target=target,
            sky_color=sky_color,
            base_brightness=float(base_brightness),
        )

    @property
    def sun_direction(self) -> Vector3:
        return _normalized(
            self.sun_target - self.sun_position,
            Vector3(0.0, -1.0, 0.0),
        )

    @property
    def light_direction(self) -> Vector3:
        direction = self.sun_direction
        return Vector3(-direction.x, -direction.y, -direction.z)

    def set_base_brightness(self, value: float) -> float:
        self.base_brightness = float(value)
        return self.base_brightness

    def set_covered_regions(self, regions: Sequence[object] | None) -> list[object]:
        self.covered_regions = list(regions or ())
        return self.covered_regions

    def set_brightness_modifiers(
        self,
        modifiers: Sequence[object] | None,
        *,
        camera: object | None = None,
        install_on_camera: bool = False,
    ) -> list[dict[str, Any]]:
        source = list(modifiers or ())
        self.brightness_modifiers = []
        self.extend_brightness_modifiers(
            source,
            camera=camera,
            install_on_camera=install_on_camera,
        )
        return self.brightness_modifiers

    def extend_brightness_modifiers(
        self,
        modifiers: Sequence[object],
        *,
        camera: object | None = None,
        install_on_camera: bool = True,
    ) -> list[dict[str, Any]]:
        added: list[dict[str, Any]] = []
        for modifier in modifiers or ():
            try:
                added.append(
                    self.add_brightness_modifier(
                        modifier,
                        camera=camera,
                        install_on_camera=install_on_camera,
                    )
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return added

    def add_brightness_modifier(
        self,
        modifier: object,
        *,
        camera: object | None = None,
        install_on_camera: bool = True,
    ) -> dict[str, Any]:
        normalized = normalize_brightness_modifier(modifier)
        self.brightness_modifiers.append(normalized)
        if install_on_camera:
            install_brightness_modifier_on_camera(camera, normalized)
        return normalized

    def sync_brightness_modifiers_from_camera(
        self,
        camera: object | None,
    ) -> list[dict[str, Any]]:
        areas = getattr(camera, "brightness_areas", ()) if camera is not None else ()
        return self.set_brightness_modifiers(areas, install_on_camera=False)


def _as_vector3(value, fallback: Vector3 | None = None) -> Vector3:
    if value is None:
        return Vector3(fallback or (0.0, 0.0, 0.0))

    try:
        return Vector3(float(value.x), float(value.y), float(value.z))
    except Exception:
        try:
            if len(value) == 2:
                return Vector3(float(value[0]), 0.0, float(value[1]))
            return Vector3(float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return Vector3(fallback or (0.0, 0.0, 0.0))


def normalize_brightness_modifier(modifier: object) -> dict[str, Any]:
    """Normalize tuple/dict/object brightness areas into the shared contract."""

    if isinstance(modifier, dict):
        center = _as_vector3(modifier["center"])
        return {
            "center": center,
            "radius": float(modifier["radius"]),
            "value": float(modifier["value"]),
            "falloff": float(modifier.get("falloff", 1.0)),
            "bounds": modifier.get("bounds"),
            "indoor_only": bool(modifier.get("indoor_only", False)),
            "floor_scale": float(modifier.get("floor_scale", 1.0)),
        }

    if all(hasattr(modifier, name) for name in ("center", "radius", "value")):
        return {
            "center": _as_vector3(getattr(modifier, "center")),
            "radius": float(getattr(modifier, "radius")),
            "value": float(getattr(modifier, "value")),
            "falloff": float(getattr(modifier, "falloff", 1.0)),
            "bounds": getattr(modifier, "bounds", None),
            "indoor_only": bool(getattr(modifier, "indoor_only", False)),
            "floor_scale": float(getattr(modifier, "floor_scale", 1.0)),
        }

    values = list(modifier)  # type: ignore[arg-type]
    return {
        "center": _as_vector3(values[0]),
        "radius": float(values[1]),
        "value": float(values[2]),
        "falloff": float(values[3]) if len(values) > 3 else 1.0,
        "bounds": values[4] if len(values) > 4 else None,
        "indoor_only": bool(values[5]) if len(values) > 5 else False,
        "floor_scale": float(values[6]) if len(values) > 6 else 1.0,
    }


def install_brightness_modifier_on_camera(
    camera: object | None,
    modifier: object,
) -> None:
    add_area = getattr(camera, "add_brightness_area", None)
    if not callable(add_area):
        return

    normalized = normalize_brightness_modifier(modifier)
    add_area(
        normalized["center"],
        normalized["radius"],
        normalized["value"],
        normalized["falloff"],
        bounds=normalized["bounds"],
        indoor_only=normalized["indoor_only"],
        floor_scale=normalized["floor_scale"],
    )


def _normalized(value, fallback: Vector3 | None = None) -> Vector3:
    vec = _as_vector3(value, fallback)
    try:
        if vec.length_squared() <= _EPSILON:
            return Vector3(fallback or (0.0, 1.0, 0.0))
        return vec.normalize()
    except Exception:
        return Vector3(fallback or (0.0, 1.0, 0.0))


def _light_direction(
    *,
    lighting: SceneLighting | None = None,
    sun_direction=None,
) -> Vector3:
    if lighting is not None:
        return lighting.light_direction

    if sun_direction is None:
        return Vector3(0.0, 1.0, 0.0)

    sun = _normalized(sun_direction, Vector3(0.0, -1.0, 0.0))
    return Vector3(-sun.x, -sun.y, -sun.z)


def sunlight_factor_for_normal(
    normal,
    *,
    lighting: SceneLighting | None = None,
    sun_direction=None,
    ambient: float | None = None,
    diffuse: float | None = None,
    max_factor: float | None = None,
) -> float:
    """Return a scalar light multiplier for a world-space normal."""

    n = _normalized(normal, Vector3(0.0, 1.0, 0.0))
    light = _light_direction(lighting=lighting, sun_direction=sun_direction)
    dot = max(0.0, n.x * light.x + n.y * light.y + n.z * light.z)

    ambient_value = (
        float(ambient)
        if ambient is not None
        else float(lighting.ambient if lighting is not None else 0.72)
    )
    diffuse_value = (
        float(diffuse)
        if diffuse is not None
        else float(lighting.diffuse if lighting is not None else 0.48)
    )
    max_value = (
        float(max_factor)
        if max_factor is not None
        else float(lighting.max_factor if lighting is not None else 1.15)
    )
    return max(0.0, min(max_value, ambient_value + diffuse_value * dot))


def sprite_light_factor(
    *,
    lighting: SceneLighting | None = None,
    sun_direction=None,
) -> float:
    """Subtle shared sunlight for billboard vegetation/props."""

    return sunlight_factor_for_normal(
        Vector3(0.0, 1.0, 0.0),
        lighting=lighting,
        sun_direction=sun_direction,
    )


def sky_sun_y(
    *,
    sun_direction=None,
    lighting: SceneLighting | None = None,
    xz_distance: float,
    fallback_y: float,
) -> float:
    """Project the sun's elevation onto a sky billboard position."""

    light = _light_direction(lighting=lighting, sun_direction=sun_direction)
    xz_len = math.hypot(float(light.x), float(light.z))
    if xz_len <= _EPSILON:
        return float(fallback_y)

    y = float(xz_distance) * float(light.y) / xz_len
    return max(-float(xz_distance) * 0.35, min(float(xz_distance) * 2.0, y))


def triangle_normals(
    vertex_data: np.ndarray,
    *,
    prefer_upward: bool = False,
) -> np.ndarray:
    """Return one normal per vertex, assuming rows are triangle lists."""

    count = int(len(vertex_data))
    normals = np.zeros((count, 3), dtype=np.float32)
    if count == 0:
        return normals

    tri_count = count // 3
    if tri_count == 0:
        normals[:, 1] = 1.0
        return normals

    positions = vertex_data[: tri_count * 3, 0:3].reshape(tri_count, 3, 3)
    e1 = positions[:, 1] - positions[:, 0]
    e2 = positions[:, 2] - positions[:, 0]
    face_normals = np.cross(e1, e2)

    if prefer_upward:
        flip = face_normals[:, 1] < 0.0
        face_normals[flip] *= -1.0

    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > _EPSILON
    face_normals[valid] /= lengths[valid, np.newaxis]
    face_normals[~valid] = np.array((0.0, 1.0, 0.0), dtype=np.float32)

    normals[: tri_count * 3] = np.repeat(face_normals, 3, axis=0)
    if tri_count * 3 < count:
        normals[tri_count * 3 :] = face_normals[-1]
    return normals


def apply_brightness_modifiers(
    vertex_data: np.ndarray,
    *,
    modifiers: Sequence[object] | None,
    default_brightness: float,
    receiver_mask: np.ndarray | Sequence[bool] | None = None,
    surface_floor_mask: np.ndarray | Sequence[bool] | None = None,
) -> None:
    """Apply the scene brightness-area contract to RGB vertex columns."""

    if len(vertex_data) == 0:
        return

    base = float(default_brightness)
    if not modifiers:
        vertex_data[:, 3:6] *= base
        return

    coords = vertex_data[:, [0, 2]]
    brightness = np.full(len(vertex_data), base, dtype=np.float32)
    receiver_flags = None
    if receiver_mask is not None:
        try:
            receiver_flags = np.asarray(receiver_mask, dtype=bool)
            if len(receiver_flags) != len(vertex_data):
                receiver_flags = None
        except (TypeError, ValueError):
            receiver_flags = None
    floor_flags = None
    if surface_floor_mask is not None:
        try:
            floor_flags = np.asarray(surface_floor_mask, dtype=bool)
            if len(floor_flags) != len(vertex_data):
                floor_flags = None
        except (TypeError, ValueError):
            floor_flags = None

    for modifier in modifiers:
        try:
            normalized = normalize_brightness_modifier(modifier)
            position = normalized["center"]
            radius = normalized["radius"]
            brightness_value = normalized["value"]
            fall_off = normalized["falloff"]
            bounds = normalized["bounds"]
            indoor_only = bool(normalized["indoor_only"])
            floor_scale = float(normalized["floor_scale"])
            radius = max(float(radius), _EPSILON)
            center_x = float(position.x)
            center_z = float(position.z)
            target = float(brightness_value)
            falloff = max(float(fall_off), 0.0)
            floor_scale = max(0.0, min(1.0, floor_scale))
        except (ValueError, AttributeError, IndexError, TypeError) as exc:
            print(f"Warning: Invalid modifier {modifier}, skipping. Error: {exc}")
            continue

        dx = coords[:, 0] - center_x
        dz = coords[:, 1] - center_z
        distances = np.sqrt(dx * dx + dz * dz)
        within = distances <= radius
        if bounds is not None:
            try:
                min_x, max_x, min_z, max_z = (float(value) for value in bounds)
                if max_x < min_x:
                    min_x, max_x = max_x, min_x
                if max_z < min_z:
                    min_z, max_z = max_z, min_z
                within &= (
                    (coords[:, 0] >= min_x)
                    & (coords[:, 0] <= max_x)
                    & (coords[:, 1] >= min_z)
                    & (coords[:, 1] <= max_z)
                )
            except (TypeError, ValueError):
                pass
        if indoor_only:
            if receiver_flags is not None:
                within &= receiver_flags
            elif vertex_data.shape[1] >= 6:
                within &= np.max(vertex_data[:, 3:6], axis=1) < 0.995
        if not np.any(within):
            continue

        norm = np.clip(distances[within] / radius, 0.0, 1.0)
        attenuation = (1.0 - norm) ** falloff
        targets = np.full(len(norm), target, dtype=np.float32)
        if floor_flags is not None and floor_scale < 1.0:
            on_floor = floor_flags[within]
            targets[on_floor] = base + (target - base) * floor_scale
        relative = targets if base == 0.0 else targets / base
        brightness[within] *= 1.0 + (relative - 1.0) * attenuation

    vertex_data[:, 3:6] *= brightness[:, np.newaxis]


def apply_directional_sunlight(
    vertex_data: np.ndarray,
    *,
    lighting: SceneLighting | None = None,
    sun_direction=None,
    normals: np.ndarray | None = None,
    prefer_upward_normals: bool = False,
) -> None:
    """Multiply RGB vertex columns by shared directional sunlight."""

    if len(vertex_data) == 0 or (lighting is None and sun_direction is None):
        return

    if normals is None:
        normals = triangle_normals(vertex_data, prefer_upward=prefer_upward_normals)

    light = _light_direction(lighting=lighting, sun_direction=sun_direction)
    light_arr = np.array((light.x, light.y, light.z), dtype=np.float32)
    lengths = np.linalg.norm(normals, axis=1)
    safe_normals = np.array(normals, dtype=np.float32, copy=True)
    valid = lengths > _EPSILON
    safe_normals[valid] /= lengths[valid, np.newaxis]
    safe_normals[~valid] = np.array((0.0, 1.0, 0.0), dtype=np.float32)

    dot = np.maximum(0.0, safe_normals @ light_arr)
    ambient = float(lighting.ambient if lighting is not None else 0.72)
    diffuse = float(lighting.diffuse if lighting is not None else 0.48)
    max_factor = float(lighting.max_factor if lighting is not None else 1.15)
    factors = np.clip(ambient + diffuse * dot, 0.0, max_factor)
    vertex_data[:, 3:6] *= factors[:, np.newaxis]
    np.clip(vertex_data[:, 3:6], 0.0, 1.0, out=vertex_data[:, 3:6])


def _covered_region_values(
    region,
    default_factor: float,
) -> tuple[float, float, float, float, float] | None:
    try:
        if isinstance(region, dict):
            min_x = float(region["min_x"])
            max_x = float(region["max_x"])
            min_z = float(region["min_z"])
            max_z = float(region["max_z"])
            factor = float(region.get("factor", default_factor))
        else:
            min_x = float(region[0])
            max_x = float(region[1])
            min_z = float(region[2])
            max_z = float(region[3])
            factor = float(region[4]) if len(region) > 4 else float(default_factor)
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    if max_x < min_x:
        min_x, max_x = max_x, min_x
    if max_z < min_z:
        min_z, max_z = max_z, min_z
    return min_x, max_x, min_z, max_z, max(0.0, min(1.0, factor))


def apply_covered_regions(
    vertex_data: np.ndarray,
    *,
    covered_regions: Sequence[object] | None,
    default_factor: float = INDOOR_LIGHT_FACTOR,
) -> None:
    """Dim vertex RGB for surfaces under roofs or otherwise indirectly lit."""

    if len(vertex_data) == 0 or not covered_regions:
        return

    coords = vertex_data[:, [0, 2]]
    factors = np.ones(len(vertex_data), dtype=np.float32)
    for region in covered_regions:
        values = _covered_region_values(region, default_factor)
        if values is None:
            continue
        min_x, max_x, min_z, max_z, factor = values
        inside = (
            (coords[:, 0] >= min_x)
            & (coords[:, 0] <= max_x)
            & (coords[:, 1] >= min_z)
            & (coords[:, 1] <= max_z)
        )
        factors[inside] = np.minimum(factors[inside], factor)

    vertex_data[:, 3:6] *= factors[:, np.newaxis]


def _smooth01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _doorway_region_factor(
    region,
    values: tuple[float, float, float, float, float],
    x: float,
    z: float,
) -> float:
    min_x, max_x, min_z, max_z, factor = values
    if not isinstance(region, dict):
        return factor

    doorway = region.get("doorway")
    if not isinstance(doorway, dict):
        return factor

    side = str(doorway.get("side", "")).lower()
    try:
        width = max(1.0, float(doorway.get("width", 48.0)))
        depth = max(1.0, float(doorway.get("depth", 64.0)))
        side_fade = max(1.0, float(doorway.get("side_fade", width * 0.25)))
        edge_factor = max(0.0, min(1.0, float(doorway.get("edge_factor", 1.0))))
        center_x = float(doorway.get("center_x", (min_x + max_x) * 0.5))
        center_z = float(doorway.get("center_z", (min_z + max_z) * 0.5))
    except (TypeError, ValueError):
        return factor

    if side == "north":
        inward_depth = max_z - z
        lateral = x - center_x
    elif side == "south":
        inward_depth = z - min_z
        lateral = x - center_x
    elif side == "east":
        inward_depth = max_x - x
        lateral = z - center_z
    elif side == "west":
        inward_depth = x - min_x
        lateral = z - center_z
    else:
        return factor

    if inward_depth < 0.0 or inward_depth > depth:
        return factor

    half_width = width * 0.5
    lateral_abs = abs(lateral)
    if lateral_abs >= half_width + side_fade:
        return factor

    if lateral_abs <= half_width:
        width_influence = 1.0
    else:
        width_influence = 1.0 - _smooth01((lateral_abs - half_width) / side_fade)
    depth_influence = 1.0 - _smooth01(inward_depth / depth)
    influence = max(0.0, min(1.0, width_influence * depth_influence))
    return factor + (edge_factor - factor) * influence


def covered_region_factor_at(
    x: float,
    z: float,
    *,
    covered_regions: Sequence[object] | None,
    default_factor: float = INDOOR_LIGHT_FACTOR,
) -> float:
    """Return the indirect-light factor for a single X/Z world position."""

    if not covered_regions:
        return 1.0

    px = float(x)
    pz = float(z)
    factor = 1.0
    for region in covered_regions:
        values = _covered_region_values(region, default_factor)
        if values is None:
            continue
        min_x, max_x, min_z, max_z, region_factor = values
        if min_x <= px <= max_x and min_z <= pz <= max_z:
            point_factor = _doorway_region_factor(region, values, px, pz)
            factor = min(factor, point_factor)
    return factor


def covered_region_mask(
    vertex_data: np.ndarray,
    *,
    covered_regions: Sequence[object] | None,
    default_factor: float = INDOOR_LIGHT_FACTOR,
) -> np.ndarray:
    """Return vertices that are inside a covered/indirect-light region."""

    mask = np.zeros(len(vertex_data), dtype=bool)
    if len(vertex_data) == 0 or not covered_regions:
        return mask

    coords = vertex_data[:, [0, 2]]
    for region in covered_regions:
        values = _covered_region_values(region, default_factor)
        if values is None:
            continue
        min_x, max_x, min_z, max_z, factor = values
        if factor >= 1.0:
            continue
        mask |= (
            (coords[:, 0] >= min_x)
            & (coords[:, 0] <= max_x)
            & (coords[:, 1] >= min_z)
            & (coords[:, 1] <= max_z)
        )
    return mask


def with_textured_normals(
    vertex_data: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    prefer_upward_normals: bool = False,
) -> np.ndarray:
    """Return textured vertex data as position/color/normal/uv rows."""

    if vertex_data.shape[1] >= 11:
        return np.ascontiguousarray(vertex_data, dtype=np.float32)
    if vertex_data.shape[1] < 8:
        raise ValueError("textured vertex data requires at least 8 columns")

    if normals is None:
        normals = triangle_normals(
            vertex_data,
            prefer_upward=prefer_upward_normals,
        )
    normals = np.ascontiguousarray(normals, dtype=np.float32)
    out = np.zeros((len(vertex_data), 11), dtype=np.float32)
    out[:, 0:6] = vertex_data[:, 0:6]
    out[:, 6:9] = normals[:, 0:3]
    out[:, 9:11] = vertex_data[:, 6:8]
    return out
