"""Directional-light shadow casters for billboard trees."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from OpenGL.GL import GL_QUADS
from pygame.math import Vector3

from engine.core.mesh import BatchedMesh
from engine.rendering.sprite import WorldSprite


_EPSILON = 1e-8


def _normalized_light_direction(direction) -> Vector3:
    try:
        value = Vector3(direction)
    except (TypeError, ValueError):
        value = Vector3(0.0, 1.0, 0.0)
    if value.length_squared() <= _EPSILON:
        return Vector3(0.0, 1.0, 0.0)
    return value.normalize()


def _tree_caster_right(light_direction: Vector3) -> Vector3:
    """Return a horizontal tree-image axis broadside to the sun."""

    horizontal = Vector3(light_direction.x, 0.0, light_direction.z)
    if horizontal.length_squared() <= _EPSILON:
        return Vector3(1.0, 0.0, 0.0)
    horizontal.normalize_ip()
    return Vector3(horizontal.z, 0.0, -horizontal.x)


def tree_shadow_vertex_groups(
    sprites: Iterable[WorldSprite],
    light_direction,
) -> dict[int, np.ndarray]:
    """Build alpha-textured vertical quads grouped by sprite atlas texture."""

    direction = _normalized_light_direction(light_direction)
    right = _tree_caster_right(direction)
    groups: dict[int, list[tuple[float, ...]]] = {}

    for sprite in sprites:
        texture = int(getattr(sprite, "texture", 0) or 0)
        if not texture:
            continue
        width, height = (float(value) for value in sprite.size)
        if width <= 0.0 or height <= 0.0:
            continue

        center = Vector3(sprite.position)
        half_right = right * (width * 0.5)
        half_up = Vector3(0.0, height * 0.5, 0.0)
        bottom_left = center - half_right - half_up
        bottom_right = center + half_right - half_up
        top_right = center + half_right + half_up
        top_left = center - half_right + half_up
        u0, v0, u1, v1 = (float(value) for value in sprite.uv_rect)

        vertices = groups.setdefault(texture, [])
        vertices.extend(
            (
                (*bottom_left, 1.0, 1.0, 1.0, u0, v0),
                (*bottom_right, 1.0, 1.0, 1.0, u1, v0),
                (*top_right, 1.0, 1.0, 1.0, u1, v1),
                (*top_left, 1.0, 1.0, 1.0, u0, v1),
            )
        )

    return {
        texture: np.asarray(vertices, dtype=np.float32)
        for texture, vertices in groups.items()
    }


class TreeSunShadowCaster:
    """Own tree quads used only by the directional shadow-map pass."""

    casts_shadows = True
    visible = True

    def __init__(self, sprites: Iterable[WorldSprite], lighting) -> None:
        self._sprites = tuple(sprites)
        self._lighting = lighting
        self._meshes: tuple[BatchedMesh, ...] = ()
        self._direction_key: tuple[float, float, float] | None = None
        self._revision = 0
        self._ensure_current()

    def _current_direction(self) -> Vector3:
        return _normalized_light_direction(
            getattr(self._lighting, "light_direction", (0.0, 1.0, 0.0))
        )

    @staticmethod
    def _key(direction: Vector3) -> tuple[float, float, float]:
        return tuple(round(float(value), 7) for value in direction)

    def _ensure_current(self) -> None:
        direction = self._current_direction()
        direction_key = self._key(direction)
        if direction_key == self._direction_key:
            return

        groups = tree_shadow_vertex_groups(self._sprites, direction)
        replacement = tuple(
            BatchedMesh.from_vertex_data(
                vertices,
                texture=texture,
                alpha_test=True,
                alpha_cutoff=0.12,
                casts_shadows=True,
                casts_sun_shadows=True,
                casts_point_shadows=False,
                keep_vertex_data=False,
                environment_lighting=False,
                shine_enabled=False,
                draw_mode=GL_QUADS,
                shader_lighting=False,
            )
            for texture, vertices in groups.items()
            if vertices.size
        )
        previous = self._meshes
        self._meshes = replacement
        self._direction_key = direction_key
        self._revision += 1
        for mesh in previous:
            mesh.dispose()

    @property
    def shadow_revision(self) -> tuple[int, tuple[float, float, float] | None]:
        self._ensure_current()
        return self._revision, self._direction_key

    def shadow_meshes(self, camera=None) -> tuple[BatchedMesh, ...]:
        self._ensure_current()
        return self._meshes

    def dispose(self) -> None:
        for mesh in self._meshes:
            mesh.dispose()
        self._meshes = ()
        self._direction_key = None
