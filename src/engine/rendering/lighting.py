"""Shared lighting helpers for world rendering.

The fixed-function renderers in this project bake most lighting into vertex
colors, so keeping the sun math here helps every mesh agree on the same source.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from pygame.math import Vector3


_EPSILON = 1e-8
INDOOR_LIGHT_FACTOR = 0.34
INDOOR_NORMAL = (0.0, -1.0, 0.0)


@dataclass
class SceneLighting:
    """Directional sunlight settings shared by sky, meshes, sprites, and shadows.

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

    @classmethod
    def from_world_center(
        cls,
        world_center: Vector3,
        *,
        sky_color: tuple[float, float, float, float],
        sun_offset: Vector3 | Sequence[float] = (36000.0, 22000.0, 18000.0),
    ) -> "SceneLighting":
        offset = _as_vector3(sun_offset, Vector3(36000.0, 22000.0, 18000.0))
        target = Vector3(float(world_center.x), 0.0, float(world_center.z))
        position = Vector3(
            float(world_center.x) + offset.x,
            float(world_center.y) + offset.y,
            float(world_center.z) + offset.z,
        )
        return cls(sun_position=position, sun_target=target, sky_color=sky_color)

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


def _as_vector3(value, fallback: Vector3 | None = None) -> Vector3:
    if value is None:
        return Vector3(fallback or (0.0, 0.0, 0.0))

    try:
        return Vector3(float(value.x), float(value.y), float(value.z))
    except Exception:
        try:
            return Vector3(float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return Vector3(fallback or (0.0, 0.0, 0.0))


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

    for modifier in modifiers:
        try:
            if isinstance(modifier, dict):
                position = modifier["center"]
                radius = modifier["radius"]
                brightness_value = modifier["value"]
                fall_off = modifier.get("falloff", 1.0)
                bounds = modifier.get("bounds")
                indoor_only = bool(modifier.get("indoor_only", False))
            else:
                position, radius, brightness_value, fall_off = modifier[:4]
                bounds = modifier[4] if len(modifier) > 4 else None
                indoor_only = False
            radius = max(float(radius), _EPSILON)
            center_x = float(position.x)
            center_z = float(position.z)
            target = float(brightness_value)
            falloff = max(float(fall_off), 0.0)
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
        relative = target if base == 0.0 else target / base
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
