"""Interactive opening chest entity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from pygame.math import Vector3

from engine.core.mesh import BatchedMesh
from engine.entity import Entity
from engine.rendering.lighting import sunlight_factor_for_normal
from engine.textures.texture_utils import load_texture
from game.resources.paths import WALL1_TEXTURE_PATH

from .slab import (
    SLAB_BOX_FACES,
    normal_for_side,
    signed_wall_tangent_for_normal,
    sphere_for_vertices,
    texture_id,
    texture_uv_rect,
    xz_bounds_for_vertices,
)


CHEST_INTERACTION_DISTANCE = 95.0
CHEST_OPEN_RADIANS = math.radians(82.0)
CHEST_OPEN_SPEED = 3.6
CHEST_DEFAULT_WIDTH = 52.0
CHEST_DEFAULT_DEPTH = 34.0
CHEST_DEFAULT_BODY_HEIGHT = 24.0
CHEST_DEFAULT_LID_HEIGHT = 6.0
CHEST_WALL_THICKNESS = 4.0
CHEST_LID_OVERHANG = 2.0
CHEST_FRONT_SHADE = 0.96
CHEST_BACK_SHADE = 0.82
CHEST_EDGE_SHADE = 0.74
CHEST_TOP_SHADE = 0.9
CHEST_BOTTOM_SHADE = 0.48


def _smooth01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _normalized(vector: Vector3, fallback: Vector3) -> Vector3:
    if vector.length_squared() <= 1e-8:
        return fallback.copy()
    return vector.normalize()


def _rotate_around_axis(vector: Vector3, axis: Vector3, angle: float) -> Vector3:
    axis = _normalized(axis, Vector3(1.0, 0.0, 0.0))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        vector * cos_a
        + axis.cross(vector) * sin_a
        + axis * (axis.dot(vector) * (1.0 - cos_a))
    )


def _oriented_box_vertices(
    center: Vector3,
    width_axis: Vector3,
    height_axis: Vector3,
    depth_axis: Vector3,
    *,
    width: float,
    height: float,
    depth: float,
) -> list[Vector3]:
    half_w = float(width) * 0.5
    half_h = float(height) * 0.5
    half_d = float(depth) * 0.5
    width_offset = width_axis * half_w
    height_offset = height_axis * half_h
    depth_offset = depth_axis * half_d

    return [
        center - width_offset - height_offset + depth_offset,
        center + width_offset - height_offset + depth_offset,
        center + width_offset + height_offset + depth_offset,
        center - width_offset + height_offset + depth_offset,
        center - width_offset - height_offset - depth_offset,
        center + width_offset - height_offset - depth_offset,
        center + width_offset + height_offset - depth_offset,
        center - width_offset + height_offset - depth_offset,
    ]


class Chest(Entity):
    """A textured chest with a hinged lid and door-like interaction."""

    faces = SLAB_BOX_FACES

    def __init__(
        self,
        position: Vector3,
        *,
        texture: Any,
        lighting: Any | None = None,
        sun_direction: Any | None = None,
        width: float = CHEST_DEFAULT_WIDTH,
        depth: float = CHEST_DEFAULT_DEPTH,
        body_height: float = CHEST_DEFAULT_BODY_HEIGHT,
        lid_height: float = CHEST_DEFAULT_LID_HEIGHT,
        side: str = "south",
        wall_thickness: float = CHEST_WALL_THICKNESS,
        open_radians: float = CHEST_OPEN_RADIANS,
        open_speed: float = CHEST_OPEN_SPEED,
        interaction_distance: float = CHEST_INTERACTION_DISTANCE,
    ) -> None:
        base_position = position.copy()
        center = base_position + Vector3(0.0, body_height * 0.5, 0.0)
        super().__init__(position=center)
        self.base_position = base_position
        self.side = str(side).lower()
        self.front_axis = normal_for_side(self.side)
        self.width_axis = signed_wall_tangent_for_normal(self.front_axis)
        self.up_axis = Vector3(0.0, 1.0, 0.0)
        self.width = max(4.0, float(width))
        self.depth = max(4.0, float(depth))
        self.body_height = max(4.0, float(body_height))
        self.lid_height = max(1.0, float(lid_height))
        self.wall_thickness = max(1.0, float(wall_thickness))
        self.open_radians = max(0.0, float(open_radians))
        self.open_speed = max(0.0, float(open_speed))
        self.interaction_distance = max(0.0, float(interaction_distance))
        self.open_amount = 0.0
        self.target_open = False
        self.runtime_update_enabled = False
        self.collision_enabled = True
        self.texture = texture_id(texture)
        self.uv_rect = texture_uv_rect(texture)
        self.lighting = lighting
        self.sun_direction = sun_direction
        self._mesh: BatchedMesh | None = None
        self._mesh_key = None
        self._bounds_cache: tuple[float, float, float, float] | None = None

    @classmethod
    def texture_or_load(cls, texture: Any | None = None) -> Any:
        if texture:
            return texture
        return load_texture(WALL1_TEXTURE_PATH)

    @property
    def is_open(self) -> bool:
        return self.open_amount >= 1.0 - 1e-4 and self.target_open

    def get_interaction_position(self) -> Vector3:
        return self.base_position + Vector3(0.0, self.body_height * 0.6, 0.0)

    def open(self) -> None:
        self.target_open = True
        self.runtime_update_enabled = self.open_amount < 1.0 - 1e-4

    def close(self) -> None:
        self.target_open = False
        self.runtime_update_enabled = self.open_amount > 1e-4

    def toggle(self) -> None:
        if self.target_open:
            self.close()
        else:
            self.open()

    def interact(self, actor=None, scene=None) -> bool:
        self.toggle()
        return True

    def update(self, dt: float) -> None:
        target = 1.0 if self.target_open else 0.0
        if self.open_amount == target:
            self.runtime_update_enabled = False
            return

        step = self.open_speed * max(0.0, float(dt))
        if step <= 0.0:
            self.open_amount = target
        elif self.open_amount < target:
            self.open_amount = min(target, self.open_amount + step)
        else:
            self.open_amount = max(target, self.open_amount - step)

        if abs(self.open_amount - target) <= 1e-4:
            self.open_amount = target
            self.runtime_update_enabled = False

        self._mark_mesh_dirty()

    def dispose(self) -> None:
        self._dispose_mesh()

    def _dispose_mesh(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.dispose()
            except Exception:
                pass
        self._mesh = None
        self._mesh_key = None

    def _mark_mesh_dirty(self) -> None:
        self._mesh_key = None

    def _sunlight_factor(self, normal: Vector3) -> float:
        lighting = getattr(self, "lighting", None)
        sun_direction = getattr(
            lighting,
            "sun_direction",
            getattr(self, "sun_direction", None),
        )
        if lighting is None and sun_direction is None:
            return 1.0
        return sunlight_factor_for_normal(
            normal,
            lighting=lighting,
            sun_direction=sun_direction,
        )

    def _light_cache_key(self):
        lighting = getattr(self, "lighting", None)
        sun_direction = getattr(
            lighting,
            "sun_direction",
            getattr(self, "sun_direction", None),
        )
        try:
            sun_key = (
                round(float(sun_direction.x), 6),
                round(float(sun_direction.y), 6),
                round(float(sun_direction.z), 6),
            )
        except Exception:
            sun_key = None

        if lighting is None:
            return sun_key

        return (
            sun_key,
            round(float(getattr(lighting, "ambient", 0.72)), 6),
            round(float(getattr(lighting, "diffuse", 0.48)), 6),
            round(float(getattr(lighting, "max_factor", 1.15)), 6),
        )

    def _face_normal(
        self,
        face_idx: int,
        width_axis: Vector3,
        height_axis: Vector3,
        depth_axis: Vector3,
    ) -> Vector3:
        if face_idx == 0:
            return depth_axis
        if face_idx == 1:
            return -depth_axis
        if face_idx == 2:
            return -width_axis
        if face_idx == 3:
            return width_axis
        if face_idx == 4:
            return height_axis
        return -height_axis

    def _face_shade(
        self,
        face_idx: int,
        width_axis: Vector3,
        height_axis: Vector3,
        depth_axis: Vector3,
    ) -> float:
        if face_idx == 0:
            base = CHEST_FRONT_SHADE
        elif face_idx == 1:
            base = CHEST_BACK_SHADE
        elif face_idx == 4:
            base = CHEST_TOP_SHADE
        elif face_idx == 5:
            base = CHEST_BOTTOM_SHADE
        else:
            base = CHEST_EDGE_SHADE
        normal = self._face_normal(face_idx, width_axis, height_axis, depth_axis)
        return max(0.0, min(1.0, base * self._sunlight_factor(normal)))

    def _face_uvs(self) -> tuple[tuple[float, float], ...]:
        u0, v0, u1, v1 = self.uv_rect
        return ((u0, v1), (u1, v1), (u1, v0), (u0, v0))

    def _append_box_vertex_data(
        self,
        rows: list[tuple[float, float, float, float, float, float, float, float]],
        verts: Sequence[Vector3],
        width_axis: Vector3,
        height_axis: Vector3,
        depth_axis: Vector3,
    ) -> None:
        uv_values = self._face_uvs()
        for face_idx, face in enumerate(self.faces):
            shade = self._face_shade(face_idx, width_axis, height_axis, depth_axis)
            for vertex_idx, uv in (
                (face[0], uv_values[0]),
                (face[1], uv_values[1]),
                (face[2], uv_values[2]),
                (face[0], uv_values[0]),
                (face[2], uv_values[2]),
                (face[3], uv_values[3]),
            ):
                vertex = verts[vertex_idx]
                rows.append(
                    (
                        vertex.x,
                        vertex.y,
                        vertex.z,
                        shade,
                        shade,
                        shade,
                        uv[0],
                        uv[1],
                    )
                )

    def _body_boxes(self):
        thickness = min(
            self.wall_thickness,
            self.width * 0.45,
            self.depth * 0.45,
            self.body_height * 0.45,
        )
        body_center_y = self.body_height * 0.5
        bottom_center_y = thickness * 0.5
        side_depth = max(1.0, self.depth - thickness * 2.0)

        base = self.base_position
        front = self.front_axis
        width_axis = self.width_axis
        up = self.up_axis

        return (
            (
                base
                + up * body_center_y
                + front * (self.depth * 0.5 - thickness * 0.5),
                width_axis,
                up,
                front,
                self.width,
                self.body_height,
                thickness,
            ),
            (
                base
                + up * body_center_y
                - front * (self.depth * 0.5 - thickness * 0.5),
                width_axis,
                up,
                front,
                self.width,
                self.body_height,
                thickness,
            ),
            (
                base
                + up * body_center_y
                - width_axis * (self.width * 0.5 - thickness * 0.5),
                front,
                up,
                width_axis,
                side_depth,
                self.body_height,
                thickness,
            ),
            (
                base
                + up * body_center_y
                + width_axis * (self.width * 0.5 - thickness * 0.5),
                front,
                up,
                width_axis,
                side_depth,
                self.body_height,
                thickness,
            ),
            (
                base + up * bottom_center_y,
                width_axis,
                up,
                front,
                self.width,
                thickness,
                self.depth,
            ),
        )

    def _lid_box(self):
        amount = _smooth01(self.open_amount)
        angle = self.open_radians * amount
        lid_width = self.width + CHEST_LID_OVERHANG * 2.0
        lid_depth = self.depth + CHEST_LID_OVERHANG * 2.0
        hinge = (
            self.base_position
            + self.up_axis * self.body_height
            - self.front_axis * (lid_depth * 0.5)
        )
        depth_axis = _rotate_around_axis(self.front_axis, self.width_axis, angle)
        height_axis = _rotate_around_axis(self.up_axis, self.width_axis, angle)
        center = hinge + depth_axis * (lid_depth * 0.5) + height_axis * (
            self.lid_height * 0.5
        )

        return (
            center,
            self.width_axis,
            height_axis,
            depth_axis,
            lid_width,
            self.lid_height,
            lid_depth,
        )

    def _visual_box_specs(self):
        return (*self._body_boxes(), self._lid_box())

    def _visual_vertices(self) -> list[Vector3]:
        verts: list[Vector3] = []
        for center, width_axis, height_axis, depth_axis, width, height, depth in (
            self._visual_box_specs()
        ):
            verts.extend(
                _oriented_box_vertices(
                    center,
                    width_axis,
                    height_axis,
                    depth_axis,
                    width=width,
                    height=height,
                    depth=depth,
                )
            )
        return verts

    def _vertex_data(self) -> np.ndarray:
        rows: list[tuple[float, float, float, float, float, float, float, float]] = []
        for center, width_axis, height_axis, depth_axis, width, height, depth in (
            self._visual_box_specs()
        ):
            verts = _oriented_box_vertices(
                center,
                width_axis,
                height_axis,
                depth_axis,
                width=width,
                height=height,
                depth=depth,
            )
            self._append_box_vertex_data(
                rows,
                verts,
                width_axis,
                height_axis,
                depth_axis,
            )

        if not rows:
            return np.zeros((0, 8), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def _mesh_cache_key(self):
        return (
            int(self.texture or 0),
            round(float(self.open_amount), 5),
            self._light_cache_key(),
        )

    def get_render_bounding_sphere(self):
        return sphere_for_vertices(self._visual_vertices())

    def _collision_vertices(self) -> list[Vector3]:
        center = self.base_position + self.up_axis * (self.body_height * 0.5)
        return _oriented_box_vertices(
            center,
            self.width_axis,
            self.up_axis,
            self.front_axis,
            width=self.width,
            height=self.body_height,
            depth=self.depth,
        )

    def get_world_vertices(self):
        if not self.collision_enabled:
            return []
        return self._collision_vertices()

    def get_bounding_box(self):
        if not self.collision_enabled:
            return None
        if self._bounds_cache is None:
            self._bounds_cache = xz_bounds_for_vertices(self._collision_vertices())
        return self._bounds_cache

    def get_collision_meshes(self):
        return (self,)

    def draw(self, camera=None) -> None:  # pragma: no cover - visual
        if not self.visible or not self.texture:
            return

        mesh_key = self._mesh_cache_key()
        if self._mesh is None or mesh_key != self._mesh_key:
            self._dispose_mesh()
            vertex_data = self._vertex_data()
            if vertex_data.size == 0:
                return
            self._mesh = BatchedMesh.from_vertex_data(
                vertex_data,
                texture=self.texture,
                alpha_test=True,
                exposure_baseline=1.0,
                environment_lighting=False,
            )
            self._mesh_key = mesh_key

        self._mesh.draw(camera=camera)
