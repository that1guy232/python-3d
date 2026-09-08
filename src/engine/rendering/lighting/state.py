"""Immutable lighting records shared by scene, adapters, and render backends."""

from __future__ import annotations

from dataclasses import dataclass


Vector3Tuple = tuple[float, float, float]
BoundsXZ = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class DirectionalLightSnapshot:
    """One immutable directional-light state using the legacy sun convention."""

    sun_position: Vector3Tuple
    sun_target: Vector3Tuple
    sun_direction: Vector3Tuple
    light_direction: Vector3Tuple
    ambient: float
    diffuse: float
    max_factor: float
    tint: Vector3Tuple


@dataclass(frozen=True, slots=True)
class LocalBrightnessLight:
    """Typed form of the legacy scalar X/Z brightness-area contract."""

    light_id: str
    center: Vector3Tuple
    radius: float
    value: float
    falloff: float = 1.0
    bounds: BoundsXZ | None = None
    indoor_only: bool = False
    floor_scale: float = 1.0

    def to_legacy_dict(self) -> dict[str, object]:
        """Project this typed scalar light into the compatibility contract."""

        return {
            "light_id": self.light_id,
            "center": self.center,
            "radius": self.radius,
            "value": self.value,
            "falloff": self.falloff,
            "bounds": self.bounds,
            "indoor_only": self.indoor_only,
            "floor_scale": self.floor_scale,
        }

    @classmethod
    def from_normalized(
        cls,
        modifier: dict,
        *,
        fallback_id: str,
    ) -> "LocalBrightnessLight":
        center = modifier["center"]
        try:
            center_values = (
                float(center.x),
                float(center.y),
                float(center.z),
            )
        except AttributeError:
            values = tuple(float(value) for value in center)
            center_values = (
                values[0],
                values[1] if len(values) > 2 else 0.0,
                values[2] if len(values) > 2 else values[1],
            )

        raw_bounds = modifier.get("bounds")
        bounds = (
            tuple(float(value) for value in raw_bounds)
            if raw_bounds is not None
            else None
        )
        return cls(
            light_id=str(modifier.get("light_id") or fallback_id),
            center=center_values,
            radius=max(0.0, float(modifier["radius"])),
            value=float(modifier["value"]),
            falloff=max(0.0, float(modifier.get("falloff", 1.0))),
            bounds=bounds,
            indoor_only=bool(modifier.get("indoor_only", False)),
            floor_scale=max(
                0.0,
                min(1.0, float(modifier.get("floor_scale", 1.0))),
            ),
        )


@dataclass(frozen=True, slots=True)
class PointLight:
    """Generic three-dimensional raster light used by the material shader."""

    light_id: str
    position: Vector3Tuple
    color: Vector3Tuple
    intensity: float
    range: float
    casts_shadows: bool = True
    importance: float = 1.0


@dataclass(frozen=True, slots=True)
class LightingSnapshot:
    """Revisioned immutable input for render and compatibility adapters."""

    revision: int
    base_brightness: float
    sky_color: tuple[float, float, float, float]
    directional: DirectionalLightSnapshot
    local_lights: tuple[LocalBrightnessLight, ...] = ()
    point_lights: tuple[PointLight, ...] = ()

    @property
    def sun_position(self) -> Vector3Tuple:
        return self.directional.sun_position

    @property
    def sun_target(self) -> Vector3Tuple:
        return self.directional.sun_target

    @property
    def sun_direction(self) -> Vector3Tuple:
        return self.directional.sun_direction

    @property
    def light_direction(self) -> Vector3Tuple:
        return self.directional.light_direction

    @property
    def ambient(self) -> float:
        return self.directional.ambient

    @property
    def diffuse(self) -> float:
        return self.directional.diffuse

    @property
    def max_factor(self) -> float:
        return self.directional.max_factor

    @property
    def sun_tint(self) -> Vector3Tuple:
        return self.directional.tint
