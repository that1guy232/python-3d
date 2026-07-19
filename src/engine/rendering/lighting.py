"""Shared environment lighting helpers for world rendering.

The fixed-function fallback still bakes lighting into vertex colors, while the
shader path samples this shared model at draw time so world geometry and torch
lights agree on one environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Sequence

import numpy as np
from pygame.math import Vector3

from engine.rendering.lighting_state import (
    DirectionalLightSnapshot,
    LightingSnapshot,
    LocalBrightnessLight,
    PointLight,
)

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
    ambient: float = 0.34
    diffuse: float = 1.28
    max_factor: float = 1.15
    sun_tint: tuple[float, float, float] = (1.0, 0.96, 0.86)
    base_brightness: float = 1.0
    _local_lights: list[LocalBrightnessLight] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _point_lights: list[PointLight] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _revision: int = field(default=0, init=False, repr=False)

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

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def local_lights(self) -> tuple[LocalBrightnessLight, ...]:
        """Immutable view of the authoritative typed local-light collection."""

        return tuple(self._local_lights)

    @property
    def point_lights(self) -> tuple[PointLight, ...]:
        """Immutable view of generic 3D lights used by raster shading."""

        return tuple(self._point_lights)

    def _touch(self) -> int:
        self._revision += 1
        return self._revision

    @staticmethod
    def _vector_tuple(value) -> tuple[float, float, float]:
        vector = _as_vector3(value)
        return (float(vector.x), float(vector.y), float(vector.z))

    def snapshot(self) -> LightingSnapshot:
        """Return an immutable, revisioned view for render adapters."""

        return LightingSnapshot(
            revision=self.revision,
            base_brightness=float(self.base_brightness),
            sky_color=tuple(float(value) for value in self.sky_color),
            directional=DirectionalLightSnapshot(
                sun_position=self._vector_tuple(self.sun_position),
                sun_target=self._vector_tuple(self.sun_target),
                sun_direction=self._vector_tuple(self.sun_direction),
                light_direction=self._vector_tuple(self.light_direction),
                ambient=float(self.ambient),
                diffuse=float(self.diffuse),
                max_factor=float(self.max_factor),
                tint=tuple(float(value) for value in self.sun_tint),
            ),
            local_lights=self.local_lights,
            point_lights=self.point_lights,
        )

    def replace_point_lights(
        self,
        lights: Sequence[PointLight] | None,
    ) -> tuple[PointLight, ...]:
        values = list(lights or ())
        if not all(isinstance(light, PointLight) for light in values):
            raise TypeError("SceneLighting point lights must be PointLight")
        self._point_lights[:] = values
        self._touch()
        return tuple(values)

    def extend_point_lights(
        self,
        lights: Sequence[PointLight],
    ) -> tuple[PointLight, ...]:
        values = list(lights or ())
        if not all(isinstance(light, PointLight) for light in values):
            raise TypeError("SceneLighting point lights must be PointLight")
        if values:
            self._point_lights.extend(values)
            self._touch()
        return tuple(values)

    def add_point_light(self, light: PointLight) -> PointLight:
        if not isinstance(light, PointLight):
            raise TypeError("SceneLighting point lights must be PointLight")
        self._point_lights.append(light)
        self._touch()
        return light

    def remove_point_lights(
        self,
        *,
        light_ids: Sequence[str] | None = None,
        id_prefix: str | None = None,
    ) -> int:
        selected_ids = {str(value) for value in light_ids or ()}
        prefix = str(id_prefix) if id_prefix is not None else None
        kept = [
            light
            for light in self._point_lights
            if not (
                light.light_id in selected_ids
                or (prefix is not None and light.light_id.startswith(prefix))
            )
        ]
        removed = len(self._point_lights) - len(kept)
        if removed:
            self._point_lights[:] = kept
            self._touch()
        return removed

    def set_base_brightness(self, value: float) -> float:
        value = float(value)
        if self.base_brightness != value:
            self.base_brightness = value
            self._touch()
        return self.base_brightness

    def replace_local_lights(
        self,
        lights: Sequence[LocalBrightnessLight] | None,
        *,
        camera: object | None = None,
        project_to_camera: bool = False,
    ) -> tuple[LocalBrightnessLight, ...]:
        """Replace authoritative local lights with typed records."""

        values = list(lights or ())
        if not all(isinstance(light, LocalBrightnessLight) for light in values):
            raise TypeError("SceneLighting local lights must be LocalBrightnessLight")
        self._local_lights[:] = values
        self._touch()
        if project_to_camera:
            self.project_local_lights_to_camera(camera)
        return tuple(values)

    def extend_local_lights(
        self,
        lights: Sequence[LocalBrightnessLight],
        *,
        camera: object | None = None,
        project_to_camera: bool = True,
    ) -> tuple[LocalBrightnessLight, ...]:
        """Append typed local lights and advance the authoritative revision."""

        values = list(lights or ())
        if not all(isinstance(light, LocalBrightnessLight) for light in values):
            raise TypeError("SceneLighting local lights must be LocalBrightnessLight")
        if values:
            self._local_lights.extend(values)
            self._touch()
            if project_to_camera:
                self.project_local_lights_to_camera(camera)
        return tuple(values)

    def add_local_light(
        self,
        light: LocalBrightnessLight,
        *,
        camera: object | None = None,
        project_to_camera: bool = True,
    ) -> LocalBrightnessLight:
        """Append one typed local light."""

        if not isinstance(light, LocalBrightnessLight):
            raise TypeError("SceneLighting local lights must be LocalBrightnessLight")
        self._local_lights.append(light)
        self._touch()
        if project_to_camera:
            self.project_local_lights_to_camera(camera)
        return light

    def update_local_light(
        self,
        light: LocalBrightnessLight | str,
        *,
        camera: object | None = None,
        radius: float | None = None,
        value: float | None = None,
    ) -> LocalBrightnessLight | None:
        """Replace one immutable local-light record by stable identity."""

        light_id = (
            light.light_id
            if isinstance(light, LocalBrightnessLight)
            else str(light)
        )

        index = None
        for candidate_index, candidate in enumerate(self._local_lights):
            if candidate is light or candidate.light_id == light_id:
                index = candidate_index
                break
        if index is None:
            return None

        target = self._local_lights[index]
        changes: dict[str, float] = {}
        if radius is not None:
            changes["radius"] = max(0.0, float(radius))
        if value is not None:
            changes["value"] = float(value)
        updated = replace(target, **changes)
        self._local_lights[index] = updated
        self._touch()
        self.project_local_lights_to_camera(camera)
        return updated

    def remove_local_lights(
        self,
        *,
        light_ids: Sequence[str] | None = None,
        id_prefix: str | None = None,
        camera: object | None = None,
    ) -> int:
        """Remove authoritative local lights selected by stable identity."""

        selected_ids = {str(value) for value in light_ids or ()}
        prefix = str(id_prefix) if id_prefix is not None else None

        def selected(light: LocalBrightnessLight) -> bool:
            return light.light_id in selected_ids or (
                prefix is not None and light.light_id.startswith(prefix)
            )

        kept_lights = [light for light in self._local_lights if not selected(light)]
        removed = len(self._local_lights) - len(kept_lights)
        if removed <= 0:
            return 0
        self._local_lights[:] = kept_lights
        self._touch()
        self.project_local_lights_to_camera(camera)
        return removed

    def project_local_lights_to_camera(self, camera: object | None) -> None:
        """Replace Camera's typed point-query projection from this revision."""

        if camera is None:
            return
        setter = getattr(camera, "replace_brightness_query_lights", None)
        if callable(setter):
            setter(self.local_lights, source_revision=self.revision)
            return
        clear = getattr(camera, "clear_brightness_query_lights", None)
        if callable(clear):
            clear()
        for light in self.local_lights:
            install_local_light_on_camera(camera, light)


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
        normalized = {
            "center": center,
            "radius": float(modifier["radius"]),
            "value": float(modifier["value"]),
            "falloff": float(modifier.get("falloff", 1.0)),
            "bounds": modifier.get("bounds"),
            "indoor_only": bool(modifier.get("indoor_only", False)),
            "floor_scale": float(modifier.get("floor_scale", 1.0)),
        }
        if modifier.get("light_id") is not None:
            normalized["light_id"] = str(modifier["light_id"])
        return normalized

    if all(hasattr(modifier, name) for name in ("center", "radius", "value")):
        normalized = {
            "center": _as_vector3(getattr(modifier, "center")),
            "radius": float(getattr(modifier, "radius")),
            "value": float(getattr(modifier, "value")),
            "falloff": float(getattr(modifier, "falloff", 1.0)),
            "bounds": getattr(modifier, "bounds", None),
            "indoor_only": bool(getattr(modifier, "indoor_only", False)),
            "floor_scale": float(getattr(modifier, "floor_scale", 1.0)),
        }
        light_id = getattr(modifier, "light_id", None)
        if light_id is not None:
            normalized["light_id"] = str(light_id)
        return normalized

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


def install_local_light_on_camera(
    camera: object | None,
    light: object,
) -> None:
    add_area = getattr(camera, "add_brightness_query_light", None)
    if not callable(add_area):
        return

    if isinstance(light, LocalBrightnessLight):
        typed_light = light
    else:
        normalized = normalize_brightness_modifier(light)
        typed_light = LocalBrightnessLight.from_normalized(
            normalized,
            fallback_id="legacy-camera-light",
        )
    add_area(typed_light)


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
    receiver_factors: np.ndarray | Sequence[float] | None = None,
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
    receiver_factor_values = None
    if receiver_factors is not None:
        try:
            receiver_factor_values = np.asarray(receiver_factors, dtype=np.float32)
            if len(receiver_factor_values) != len(vertex_data):
                receiver_factor_values = None
        except (TypeError, ValueError):
            receiver_factor_values = None
    if receiver_factor_values is None and vertex_data.shape[1] >= 6:
        receiver_factor_values = np.clip(
            np.max(vertex_data[:, 3:6], axis=1),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

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
        if indoor_only and receiver_factor_values is not None:
            attenuation *= _indoor_light_contribution_weights(
                receiver_factor_values[within],
            )
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


def indoor_light_contribution_weight(
    receiver_factor: float,
    *,
    indoor_factor: float = INDOOR_LIGHT_FACTOR,
) -> float:
    """Fade indoor-only local lights out on surfaces already lit by openings."""

    receiver = max(0.0, min(1.0, float(receiver_factor)))
    indoor = max(0.0, min(0.999, float(indoor_factor)))
    if receiver <= indoor:
        return 1.0
    if receiver >= 1.0:
        return 0.0
    return 1.0 - _smooth01((receiver - indoor) / (1.0 - indoor))


def _indoor_light_contribution_weights(
    receiver_factors: np.ndarray,
    *,
    indoor_factor: float = INDOOR_LIGHT_FACTOR,
) -> np.ndarray:
    receivers = np.clip(np.asarray(receiver_factors, dtype=np.float32), 0.0, 1.0)
    indoor = max(0.0, min(0.999, float(indoor_factor)))
    weights = np.ones(len(receivers), dtype=np.float32)
    fading = receivers > indoor
    if np.any(fading):
        t = np.clip((receivers[fading] - indoor) / (1.0 - indoor), 0.0, 1.0)
        weights[fading] = 1.0 - (t * t * (3.0 - 2.0 * t))
    weights[receivers >= 1.0] = 0.0
    return weights


def region_light_openings(region) -> list[dict[str, Any]]:
    """Return light openings on a covered region, preserving legacy doorway data."""

    if not isinstance(region, dict):
        return []

    openings = region.get("openings")
    if isinstance(openings, Sequence) and not isinstance(openings, (str, bytes)):
        return [opening for opening in openings if isinstance(opening, dict)]

    results: list[dict[str, Any]] = []
    doorway = region.get("doorway")
    if isinstance(doorway, dict):
        results.append(doorway)

    windows = region.get("windows")
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)):
        results.extend(window for window in windows if isinstance(window, dict))

    return results


def _opening_region_factor(
    opening: dict,
    values: tuple[float, float, float, float, float],
    x: float,
    z: float,
) -> float:
    min_x, max_x, min_z, max_z, factor = values
    side = str(opening.get("side", "")).lower()
    try:
        width = max(1.0, float(opening.get("width", 48.0)))
        depth = max(1.0, float(opening.get("depth", 64.0)))
        side_fade = max(1.0, float(opening.get("side_fade", width * 0.25)))
        edge_factor = max(0.0, min(1.0, float(opening.get("edge_factor", 1.0))))
        center_x = float(opening.get("center_x", (min_x + max_x) * 0.5))
        center_z = float(opening.get("center_z", (min_z + max_z) * 0.5))
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


def _doorway_region_factor(
    region,
    values: tuple[float, float, float, float, float],
    x: float,
    z: float,
) -> float:
    if not isinstance(region, dict):
        return values[4]

    best = values[4]
    for opening in region_light_openings(region):
        best = max(best, _opening_region_factor(opening, values, x, z))
    return best


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
