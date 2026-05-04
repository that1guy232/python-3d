"""Shared helpers for textured rectangular slab objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from OpenGL.GL import (
    GL_ALPHA_TEST,
    GL_BLEND,
    GL_GREATER,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    glAlphaFunc,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glTexCoord2f,
    glVertex3f,
)
from pygame.math import Vector3

from core.mesh import BatchedMesh
from engine.rendering.lighting import sunlight_factor_for_normal


SLAB_BOX_FACES = [
    (0, 1, 2, 3),  # front
    (5, 4, 7, 6),  # back
    (4, 0, 3, 7),  # left/hinge edge
    (1, 5, 6, 2),  # right/latch edge
    (3, 2, 6, 7),  # top
    (4, 5, 1, 0),  # bottom
]

_SIDE_NORMALS = {
    "north": Vector3(0.0, 0.0, 1.0),
    "east": Vector3(1.0, 0.0, 0.0),
    "south": Vector3(0.0, 0.0, -1.0),
    "west": Vector3(-1.0, 0.0, 0.0),
}


def normal_for_side(side: str, *, default: str = "south") -> Vector3:
    """Return the outward XZ normal for a cardinal side name."""
    fallback = _SIDE_NORMALS.get(default, _SIDE_NORMALS["south"])
    return _SIDE_NORMALS.get(str(side).lower(), fallback).copy()


def signed_wall_tangent_for_normal(normal: Vector3) -> Vector3:
    """Return a handed wall tangent that follows the normal orientation."""
    tangent = Vector3(-normal.z, 0.0, normal.x)
    if tangent.length_squared() <= 1e-8:
        return Vector3(1.0, 0.0, 0.0)
    return tangent.normalize()


def axis_wall_tangent_for_normal(normal: Vector3) -> Vector3:
    """Return the unsigned wall axis tangent for side-mounted static slabs."""
    if abs(float(normal.z)) > 0.0:
        return Vector3(1.0, 0.0, 0.0)
    return Vector3(0.0, 0.0, 1.0)


def texture_id(texture: Any) -> int:
    return int(getattr(texture, "texture", texture) or 0)


def texture_uv_rect(texture: Any) -> tuple[float, float, float, float]:
    return getattr(texture, "uv_rect", (0.0, 0.0, 1.0, 1.0))


def slab_box_vertices(
    center: Vector3,
    width_axis: Vector3,
    depth_axis: Vector3,
    *,
    width: float,
    height: float,
    thickness: float,
) -> list[Vector3]:
    half_w = float(width) * 0.5
    half_h = float(height) * 0.5
    half_t = float(thickness) * 0.5
    up = Vector3(0.0, 1.0, 0.0)
    width_offset = width_axis * half_w
    height_offset = up * half_h
    depth_offset = depth_axis * half_t

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


def slab_face_normal(
    face_idx: int,
    panel_axis: Vector3,
    depth_axis: Vector3,
) -> Vector3:
    if face_idx == 0:
        return depth_axis
    if face_idx == 1:
        return -depth_axis
    if face_idx == 2:
        return -panel_axis
    if face_idx == 3:
        return panel_axis
    if face_idx == 4:
        return Vector3(0.0, 1.0, 0.0)
    return Vector3(0.0, -1.0, 0.0)


def xz_bounds_for_vertices(
    vertices: Sequence[Vector3],
) -> tuple[float, float, float, float] | None:
    if not vertices:
        return None
    min_x = min(v.x for v in vertices)
    max_x = max(v.x for v in vertices)
    min_z = min(v.z for v in vertices)
    max_z = max(v.z for v in vertices)
    return (min_x, max_x, min_z, max_z)


class TexturedSlabMixin:
    """Shared behavior for entity slabs with textured box visuals."""

    faces = SLAB_BOX_FACES

    def _dispose_slab_mesh(self) -> None:
        mesh = getattr(self, "_slab_mesh", None)
        if mesh is not None:
            try:
                mesh.dispose()
            except Exception:
                pass
        self._slab_mesh = None

    def _mark_slab_mesh_dirty(self) -> None:
        self._slab_mesh_dirty = True

    def dispose(self) -> None:
        self._dispose_slab_mesh()

    def _slab_light_cache_key(self):
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

    def get_collision_meshes(self):
        return (self,)

    def _face_normal(self, face_idx: int) -> Vector3:
        return slab_face_normal(face_idx, self.panel_axis, self.depth_axis)

    def _sunlight_factor(self, normal) -> float:
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

    def _visual_vertices(self) -> list[Vector3]:
        return self._box_vertices(
            self.position,
            self.panel_axis,
            self.depth_axis,
            width=self.width,
            height=self.height,
            thickness=self.thickness,
        )

    @staticmethod
    def _box_vertices(
        center: Vector3,
        width_axis: Vector3,
        depth_axis: Vector3,
        *,
        width: float,
        height: float,
        thickness: float,
    ) -> list[Vector3]:
        return slab_box_vertices(
            center,
            width_axis,
            depth_axis,
            width=width,
            height=height,
            thickness=thickness,
        )

    def _collision_center(self) -> Vector3:
        return self.position

    def _collision_panel_axis(self) -> Vector3:
        return self.panel_axis

    def _collision_depth_axis(self) -> Vector3:
        return self.depth_axis

    def get_world_vertices(self):
        if not self.collision_enabled:
            return []

        return self._box_vertices(
            self._collision_center(),
            self._collision_panel_axis(),
            self._collision_depth_axis(),
            width=self.collision_width,
            height=self.collision_height,
            thickness=self.collision_thickness,
        )

    def get_bounding_box(self):
        if not self.collision_enabled:
            return None
        if self._bounds_cache is not None:
            return self._bounds_cache

        self._bounds_cache = xz_bounds_for_vertices(self.get_world_vertices())
        return self._bounds_cache

    def _draw_textured_slab_faces(self, verts: Sequence[Vector3]) -> None:
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glEnable(GL_ALPHA_TEST)
        glAlphaFunc(GL_GREATER, 0.01)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        glBegin(GL_QUADS)
        try:
            for face_idx, face in enumerate(self.faces):
                shade = self._face_shade(face_idx)
                glColor4f(shade, shade, shade, 1.0)
                for vertex_idx, uv in zip(face, self._face_uvs(face_idx)):
                    vertex = verts[vertex_idx]
                    glTexCoord2f(uv[0], uv[1])
                    glVertex3f(vertex.x, vertex.y, vertex.z)
        finally:
            glEnd()
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glDisable(GL_ALPHA_TEST)
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)

    def _slab_vertex_data(self, verts: Sequence[Vector3]) -> np.ndarray:
        rows = []
        for face_idx, face in enumerate(self.faces):
            if len(face) != 4:
                continue
            shade = self._face_shade(face_idx)
            uvs = self._face_uvs(face_idx)
            for vertex_idx, uv in (
                (face[0], uvs[0]),
                (face[1], uvs[1]),
                (face[2], uvs[2]),
                (face[0], uvs[0]),
                (face[2], uvs[2]),
                (face[3], uvs[3]),
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

        if not rows:
            return np.zeros((0, 8), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def _draw_cached_textured_slab_faces(self, verts: Sequence[Vector3]) -> None:
        if not self.texture:
            return

        mesh = getattr(self, "_slab_mesh", None)
        light_key = self._slab_light_cache_key()
        dirty = bool(getattr(self, "_slab_mesh_dirty", True))
        if light_key != getattr(self, "_slab_mesh_light_key", None):
            dirty = True

        if mesh is None or dirty:
            self._dispose_slab_mesh()
            vertex_data = self._slab_vertex_data(verts)
            if vertex_data.size == 0:
                return
            self._slab_mesh = BatchedMesh.from_vertex_data(
                vertex_data,
                texture=self.texture,
                alpha_test=True,
                exposure_baseline=1.0,
                environment_lighting=False,
            )
            self._slab_mesh_dirty = False
            self._slab_mesh_light_key = light_key
            mesh = self._slab_mesh

        mesh.draw()
