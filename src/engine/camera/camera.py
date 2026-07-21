"""First-person camera state, brightness sampling, and frustum helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
import numpy as np
from pygame.math import Vector3
from engine.config import WIDTH, HEIGHT
import pygame

if TYPE_CHECKING:
    from engine.rendering.lighting_state import LocalBrightnessLight


@dataclass(frozen=True, slots=True)
class CameraBrightnessArea:
    """Immutable camera-owned projection for X/Z brightness point queries."""

    light_id: str
    center: tuple[float, float, float]
    radius: float
    value: float
    falloff: float = 1.0
    bounds: tuple[float, float, float, float] | None = None
    indoor_only: bool = False
    floor_scale: float = 1.0


class Camera:
    """Mutable view object shared by input, lighting, culling, and rendering."""

    def __init__(
        self,
        position=None,
        rotation=None,
        width=800,
        height=600,
        fov=75,
        default_brightness=0.1,
    ):
        # keep external API types the same (pygame.Vector3)
        self.position = position if position is not None else Vector3(0, 0, -300)
        self.rotation = rotation or Vector3(0, 0, 0)  # pitch (x), yaw (y), roll (z)
        self.speed = 5

        # Typed point-query projection of the scene's immutable local lights.
        self._brightness_lights: list[CameraBrightnessArea] = []
        self._brightness_default = float(default_brightness)
        self._brightness_source_revision: int | None = None

        # FOV scale (same formula you used)
        self._fov_scale = (height / 2) / math.tan(math.radians(fov / 2))

        # cached trig & direction vectors (kept as Vector3 where movement uses them)
        self._yaw_cos = 1.0
        self._yaw_sin = 0.0
        self._pitch_cos = 1.0
        self._pitch_sin = 0.0
        self._right = Vector3(1.0, 0.0, 0.0)
        self._up = Vector3(0.0, 1.0, 0.0)
        self._forward = Vector3(0.0, 0.0, -1.0)

        # NumPy view-basis rows used by render-time culling:
        # right, up, and positive forward depth.
        self._R = np.eye(3, dtype=np.float64)

        # Manual height offset (adjustable by user for screenshots, added on top of
        # the terrain-following camera Y). WorldScene will add this when computing
        # the target camera height.
        self.manual_height_offset = 0.0
        self.vertical_velocity = 0.0
        self.is_jumping = False

        # Speed used when adjusting manual height offset with Q/E (world units/sec).
        # Chosen to be relatively small so Q/E can be used for fine screenshot tweaks.
        self.height_adjust_speed = 50.0

        # OPTIMIZATION: Cache for brightness lookups
        self._brightness_cache = {}
        self._brightness_cache_size = 1000  # Maximum cache size
        self._brightness_cache_hits = 0
        self._brightness_cache_misses = 0

        # initialize
        self.update_rotation(0)

    @property
    def brightness_default(self):
        return self._brightness_default

    @brightness_default.setter
    def brightness_default(self, value):
        self._brightness_default = float(value)
        cache = getattr(self, "_brightness_cache", None)
        if cache is not None:
            cache.clear()

    def set_brightness_default(self, value):
        """Set the baseline world brightness and invalidate brightness samples."""
        self.brightness_default = value
        return self._brightness_default

    @staticmethod
    def _legacy_brightness_area(light: CameraBrightnessArea) -> dict:
        area = {
            "light_id": light.light_id,
            "center": Vector3(light.center),
            "radius": float(light.radius),
            "value": float(light.value),
            "falloff": float(light.falloff),
            "floor_scale": float(light.floor_scale),
        }
        if light.bounds is not None:
            area["bounds"] = tuple(float(value) for value in light.bounds)
        if light.indoor_only:
            area["indoor_only"] = True
        return area

    @classmethod
    def _legacy_brightness_contributions(cls, contributions) -> list[dict]:
        return [
            {
                **value,
                "area": cls._legacy_brightness_area(value["area"]),
            }
            for value in contributions
        ]

    @property
    def brightness_query_lights(self) -> tuple[CameraBrightnessArea, ...]:
        """Immutable typed inputs used by camera point queries."""

        return tuple(self._brightness_lights)

    @property
    def has_brightness_query_lights(self) -> bool:
        return bool(self._brightness_lights)

    @staticmethod
    def _center_tuple(center) -> tuple[float, float, float]:
        try:
            return (float(center.x), float(center.y), float(center.z))
        except AttributeError:
            values = tuple(float(value) for value in center)
            if len(values) == 2:
                return (values[0], 0.0, values[1])
            return (values[0], values[1], values[2])

    def _invalidate_brightness_projection(self, *, source_revision=None) -> None:
        self._brightness_cache.clear()
        self._brightness_source_revision = (
            int(source_revision) if source_revision is not None else None
        )

    # Typed query-light projection helpers.
    def add_brightness_query_light(
        self,
        light: CameraBrightnessArea | LocalBrightnessLight,
    ) -> CameraBrightnessArea:
        """Add one typed point-query record."""

        projected = self._project_brightness_query_light(light)
        self._brightness_lights.append(projected)
        self._invalidate_brightness_projection()
        return projected

    @classmethod
    def _project_brightness_query_light(
        cls,
        source: CameraBrightnessArea | LocalBrightnessLight,
    ) -> CameraBrightnessArea:
        from engine.rendering.lighting_state import LocalBrightnessLight

        if isinstance(source, CameraBrightnessArea):
            return source
        if not isinstance(source, LocalBrightnessLight):
            raise TypeError(
                "Camera query lights must be CameraBrightnessArea or "
                "LocalBrightnessLight"
            )
        return CameraBrightnessArea(
            light_id=source.light_id,
            center=cls._center_tuple(source.center),
            radius=max(0.0, float(source.radius)),
            value=float(source.value),
            falloff=max(0.0, float(source.falloff)),
            bounds=(
                tuple(float(item) for item in source.bounds)
                if source.bounds is not None
                else None
            ),
            indoor_only=bool(source.indoor_only),
            floor_scale=max(0.0, min(1.0, float(source.floor_scale))),
        )

    def replace_brightness_query_lights(
        self,
        lights: list[CameraBrightnessArea | LocalBrightnessLight]
        | tuple[CameraBrightnessArea | LocalBrightnessLight, ...],
        *,
        source_revision: int | None = None,
    ) -> tuple[CameraBrightnessArea, ...]:
        """Replace point-query areas from an authoritative lighting snapshot."""

        values = [
            self._project_brightness_query_light(source)
            for source in lights or ()
        ]
        self._brightness_lights[:] = values
        self._invalidate_brightness_projection(source_revision=source_revision)
        return self.brightness_query_lights

    def clear_brightness_query_lights(self) -> None:
        """Remove all point-query lights."""
        self._brightness_lights.clear()
        self._invalidate_brightness_projection()

    def _get_cache_key(self, point, surface_indoor=None):
        """Generate a cache key for a point (rounded to reduce cache size)."""
        # Round to nearest 10 units to create cache buckets
        if surface_indoor is None:
            surface_key = None
        else:
            surface_key = bool(surface_indoor)
        return (round(point.x / 10) * 10, round(point.z / 10) * 10, surface_key)

    def get_brightness_at_with_blending(self, point, surface_indoor=None):
        """
        Return the brightness at the given world-space point with proper area blending.
        This version properly combines overlapping brightness areas by adding their contributions.
        """
        if not self._brightness_lights:
            return self.brightness_default

        # Check cache first
        cache_key = self._get_cache_key(point, surface_indoor=surface_indoor)
        if cache_key in self._brightness_cache:
            self._brightness_cache_hits += 1
            cached_result = self._brightness_cache[cache_key]
            return (
                cached_result.get("brightness_blended", self.brightness_default)
                if cached_result
                else self.brightness_default
            )

        self._brightness_cache_misses += 1

        # Calculate brightness from all overlapping areas using the same
        # multiplicative target-value contract as the static mesh builders.
        point_x, point_z = point.x, point.z
        total_brightness = self.brightness_default
        contributing_areas = []

        # Find all areas that contain this point and calculate their contributions
        for area in self._brightness_lights:
            if area.indoor_only and surface_indoor is False:
                continue

            bounds = area.bounds
            if bounds is not None:
                min_x, max_x, min_z, max_z = bounds
                if not (min_x <= point_x <= max_x and min_z <= point_z <= max_z):
                    continue

            dx = point_x - area.center[0]
            dz = point_z - area.center[2]
            dist_sq = dx * dx + dz * dz

            if dist_sq <= area.radius * area.radius:
                # Calculate distance-based falloff within the area
                radius = area.radius
                falloff = area.falloff
                value = area.value

                if radius > 1e-12:
                    # Calculate normalized distance (0 at center, 1 at edge)
                    distance = math.sqrt(dist_sq)
                    norm_distance = min(1.0, distance / radius)

                    # Apply falloff (higher falloff = sharper edge)
                    attenuation = (1.0 - norm_distance) ** max(falloff, 0.0)

                    if self.brightness_default == 0.0:
                        relative = value
                    else:
                        relative = value / self.brightness_default
                    effect = 1.0 + (relative - 1.0) * attenuation
                    total_brightness *= effect
                    contributing_areas.append(
                        {
                            "area": area,
                            "contribution": abs(effect - 1.0),
                            "effect": effect,
                            "attenuation": attenuation,
                        }
                    )

        # Cache the result with all the data we computed
        best_area = (
            max(contributing_areas, key=lambda x: x["contribution"])["area"]
            if contributing_areas
            else None
        )
        result = {
            "brightness_blended": total_brightness,
            "brightness": (
                max(ca["area"].value for ca in contributing_areas)
                if contributing_areas
                else self.brightness_default
            ),
            "area": best_area,
            "contributing_areas": contributing_areas,
        }
        self._cache_result(cache_key, result)

        return total_brightness

    def get_brightness_at(self, point, surface_indoor=None):
        """
        Return the brightness at the given world-space point (pygame.Vector3).
        FIXED: Now uses proper blending for overlapping areas.
        """
        return self.get_brightness_at_with_blending(
            point,
            surface_indoor=surface_indoor,
        )

    def get_brightness_area_with_blending(self, point):
        """
        Get detailed brightness information at a point including all contributing areas.
        Returns a dict with blended brightness and list of contributing areas.
        """
        if not self._brightness_lights:
            return None

        # Check cache first
        cache_key = self._get_cache_key(point)
        if cache_key in self._brightness_cache:
            self._brightness_cache_hits += 1
            cached_result = self._brightness_cache[cache_key]
            if cached_result and "contributing_areas" in cached_result:
                return {
                    "brightness": cached_result["brightness_blended"],
                    "contributing_areas": self._legacy_brightness_contributions(
                        cached_result["contributing_areas"]
                    ),
                    "primary_area": (
                        self._legacy_brightness_area(cached_result["area"])
                        if cached_result["area"]
                        else None
                    ),
                }
            return None

        # Force calculation by calling get_brightness_at_with_blending
        brightness = self.get_brightness_at_with_blending(point)

        # Get the cached result that was just computed
        cached_result = self._brightness_cache.get(cache_key)
        if cached_result and "contributing_areas" in cached_result:
            return {
                "brightness": brightness,
                "contributing_areas": self._legacy_brightness_contributions(
                    cached_result["contributing_areas"]
                ),
                "primary_area": (
                    self._legacy_brightness_area(cached_result["area"])
                    if cached_result["area"]
                    else None
                ),
            }

        return None

    def get_brightness_area(self, point):
        """
        OPTIMIZED: Uses caching and precomputed distances.
        Returns the primary (strongest) brightness area at this point.
        """
        blend_info = self.get_brightness_area_with_blending(point)
        return blend_info["primary_area"] if blend_info else None

    def _cache_result(self, key, result):
        """Cache a result with size management."""
        # Simple LRU-like behavior: if cache is full, clear it
        if len(self._brightness_cache) >= self._brightness_cache_size:
            # Keep only the most recently accessed items (simple approach)
            self._brightness_cache.clear()

        self._brightness_cache[key] = result

    def get_cache_stats(self):
        """Return cache performance statistics for debugging."""
        total = self._brightness_cache_hits + self._brightness_cache_misses
        hit_rate = self._brightness_cache_hits / total if total > 0 else 0
        return {
            "hits": self._brightness_cache_hits,
            "misses": self._brightness_cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._brightness_cache),
        }

    def clear_cache_stats(self):
        """Reset cache statistics."""
        self._brightness_cache_hits = 0
        self._brightness_cache_misses = 0

    # BATCH PROCESSING OPTIMIZATION
    def get_brightness_batch(self, points):
        """
        Get brightness for multiple points at once with proper blending.
        More efficient than calling get_brightness_at() multiple times.

        Args:
            points: List of pygame.Vector3 points

        Returns:
            List of brightness values corresponding to each point
        """
        if not self._brightness_lights:
            return [self.brightness_default] * len(points)

        results = []
        for point in points:
            # Use the blending version for batch processing too
            brightness = self.get_brightness_at_with_blending(point)
            results.append(brightness)

        return results

    @property
    def brightness(self):
        """Convenience property: brightness at the camera's current position."""
        return self.get_brightness_at(self.position)

    def update_rotation(self, dt):
        """Precompute trig, direction vectors (Vector3) and rotation matrix (numpy)."""
        cp = math.cos(self.rotation.x)
        sp = math.sin(self.rotation.x)
        cy = math.cos(self.rotation.y)
        sy = math.sin(self.rotation.y)

        self._yaw_cos = cy
        self._yaw_sin = sy
        self._pitch_cos = cp
        self._pitch_sin = sp

        # Right vector (matches your previous expression)
        self._right = Vector3(self._yaw_cos, 0, -self._yaw_sin)
        # Full forward vector (with vertical component when pitched)
        self._forward = Vector3(
            -self._pitch_cos * self._yaw_sin,
            self._pitch_sin,
            -self._pitch_cos * self._yaw_cos,
        )
        up = self._right.cross(self._forward)
        self._up = up.normalize() if up.length_squared() > 1e-12 else Vector3(0, 1, 0)
        # Ground forward (horizontal component only — keep same formula you used)
        self._ground_forward = -Vector3(-self._yaw_sin, 0, -self._yaw_cos)

        self._R = np.array(
            [
                [self._right.x, self._right.y, self._right.z],
                [self._up.x, self._up.y, self._up.z],
                [self._forward.x, self._forward.y, self._forward.z],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _coerce_point3(point) -> tuple[float, float, float]:
        if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
            return float(point.x), float(point.y), float(point.z)
        return float(point[0]), float(point[1]), float(point[2])

    def world_delta_to_view(
        self,
        dx: float,
        dy: float,
        dz: float,
    ) -> tuple[float, float, float]:
        """Return camera-space right/up/depth for a world-space delta."""
        right = self._right
        up = self._up
        forward = self._forward
        x_cam = dx * right.x + dy * right.y + dz * right.z
        y_cam = dx * up.x + dy * up.y + dz * up.z
        depth = dx * forward.x + dy * forward.y + dz * forward.z
        return x_cam, y_cam, depth

    def world_point_to_view(self, point) -> tuple[float, float, float]:
        """Return camera-space right/up/depth for a world-space point."""
        px, py, pz = self._coerce_point3(point)
        dx = px - float(self.position.x)
        dy = py - float(self.position.y)
        dz = pz - float(self.position.z)
        return self.world_delta_to_view(dx, dy, dz)

    def tan_half_fov(self, viewport_height: float = HEIGHT) -> float:
        fov_scale = max(1e-6, float(getattr(self, "_fov_scale", HEIGHT * 0.5)))
        return (float(viewport_height) * 0.5) / fov_scale

    def sphere_in_frustum(
        self,
        center,
        radius: float = 0.0,
        *,
        far_distance: float | None = None,
        near_distance: float = 0.0,
        viewport_width: float = WIDTH,
        viewport_height: float = HEIGHT,
        extra_margin: float = 0.0,
    ) -> bool:
        """Conservative camera-frustum test for a world-space bounding sphere."""
        if center is None:
            return True

        try:
            if hasattr(center, "x") and hasattr(center, "y") and hasattr(center, "z"):
                px = float(center.x)
                py = float(center.y)
                pz = float(center.z)
            else:
                px = float(center[0])
                py = float(center[1])
                pz = float(center[2])
            position = self.position
            dx = px - float(position.x)
            dy = py - float(position.y)
            dz = pz - float(position.z)
            right = self._right
            up = self._up
            forward = self._forward
            x_cam = dx * right.x + dy * right.y + dz * right.z
            y_cam = dx * up.x + dy * up.y + dz * up.z
            depth = dx * forward.x + dy * forward.y + dz * forward.z
        except Exception:
            return True

        radius = max(0.0, float(radius) + max(0.0, float(extra_margin)))
        near = float(near_distance)
        if depth < near - radius:
            return False

        if far_distance is not None:
            far = max(near, float(far_distance))
            if depth > far + radius:
                return False

        fov_scale = max(1e-6, float(getattr(self, "_fov_scale", HEIGHT * 0.5)))
        tan_half = (float(viewport_height) * 0.5) / fov_scale
        aspect = float(viewport_width) / max(1e-6, float(viewport_height))
        depth_for_extent = max(0.0, depth)
        half_v = depth_for_extent * tan_half
        half_h = half_v * aspect
        tan_half_h = tan_half * aspect
        horizontal_margin = radius * math.sqrt(1.0 + tan_half_h * tan_half_h)
        vertical_margin = radius * math.sqrt(1.0 + tan_half * tan_half)
        return (
            abs(x_cam) <= half_h + horizontal_margin
            and abs(y_cam) <= half_v + vertical_margin
        )

    def move_camera(self, keys, speed, dt):
        """Movement using precomputed direction vectors (keeps same calls/semantics)."""
        # keys is expected to be pygame.key.get_pressed() or similar mapping
        forward_axis = int(bool(keys[pygame.K_s])) - int(bool(keys[pygame.K_w]))
        strafe_axis = int(bool(keys[pygame.K_d])) - int(bool(keys[pygame.K_a]))
        movement = (
            self._ground_forward * forward_axis + self._right * strafe_axis
        )
        if movement.length_squared() > 1.0:
            movement.normalize_ip()
        self.position += movement * speed * dt

        return self.position
