"""Fixed window entity backed by a world-space textured slab."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from pygame.math import Vector3
from OpenGL.GL import (
    GL_ALPHA_TEST,
    GL_BLEND,
    GL_QUADS,
    GL_TEXTURE_2D,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_REPEAT,
    glBegin,
    glBindTexture,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glTexCoord2f,
    glTexParameteri,
    glVertex3f,
)

from core.mesh import BatchedMesh
from engine.entity import Entity
from engine.rendering.lighting import INDOOR_NORMAL
from .slab import (
    TexturedSlabMixin,
    axis_wall_tangent_for_normal,
    normal_for_side,
    texture_id,
    texture_uv_rect,
)
from textures.resource_path import WINDOW_TEXTURE_PATH
from textures.texture_utils import get_texture_size, load_texture


WINDOW_THICKNESS = 3.0
WINDOW_EDGE_UV_FRACTION = 0.08
WINDOW_FRAME_OVERLAP = 0.5
WINDOW_TEXTURE_WIDTH = 22.0
WINDOW_TEXTURE_HEIGHT = 38.0
WINDOW_TEXTURE_ASPECT = WINDOW_TEXTURE_WIDTH / WINDOW_TEXTURE_HEIGHT
WINDOW_DEFAULT_HEIGHT = 38.0
WINDOW_DEFAULT_WIDTH = WINDOW_DEFAULT_HEIGHT * WINDOW_TEXTURE_ASPECT
WINDOW_DEFAULT_SILL_HEIGHT = 18.0
WINDOW_EDGE_SHADE = 0.82
WINDOW_TOP_SHADE = 0.9
WINDOW_BOTTOM_SHADE = 0.62
WINDOW_INTERIOR_SHADE = 0.92
WINDOW_CORNER_BACKING_COLOR = (0.16, 0.15, 0.16)
WINDOW_CORNER_BACKING_BANDS = (
    (0.55, 0.65, 1.0),
    (0.65, 0.72, 0.96),
    (0.72, 0.79, 0.88),
    (0.79, 0.86, 0.76),
    (0.86, 0.93, 0.58),
    (0.93, 0.98, 0.38),
    (0.98, 1.0, 0.24),
)


class Window(TexturedSlabMixin, Entity):
    """A fixed window that blocks movement but always lets light through."""

    TEXTURE_ASPECT = WINDOW_TEXTURE_ASPECT
    DEFAULT_WIDTH = WINDOW_DEFAULT_WIDTH
    DEFAULT_HEIGHT = WINDOW_DEFAULT_HEIGHT
    DEFAULT_SILL_HEIGHT = WINDOW_DEFAULT_SILL_HEIGHT

    def __init__(
        self,
        position: Vector3,
        *,
        camera: object,
        texture: Any,
        backing_texture: Any | None = None,
        lighting: Any | None = None,
        sun_direction: Any | None = None,
        width: float = WINDOW_DEFAULT_WIDTH,
        height: float = WINDOW_DEFAULT_HEIGHT,
        side: str = "south",
        thickness: float = WINDOW_THICKNESS,
        collision_width: float | None = None,
        collision_height: float | None = None,
        collision_thickness: float | None = None,
    ) -> None:
        super().__init__(position=position.copy())
        self.camera = camera
        self.center = position.copy()
        self.side = str(side).lower()
        self.normal = normal_for_side(self.side)
        self.tangent = axis_wall_tangent_for_normal(self.normal)
        self.width = max(1.0, float(width))
        self.height = max(1.0, float(height))
        self.thickness = max(0.1, float(thickness))
        self.collision_width = max(1.0, float(collision_width or self.width))
        self.collision_height = max(1.0, float(collision_height or self.height))
        self.collision_thickness = max(
            0.1,
            float(collision_thickness or self.thickness),
        )
        self.collision_enabled = True
        self.texture = texture_id(texture)
        self.uv_rect = texture_uv_rect(texture)
        self.backing_texture = texture_id(backing_texture)
        self.backing_uv_rect = texture_uv_rect(backing_texture)
        self.backing_texture_size = get_texture_size(backing_texture)
        self.lighting = lighting
        self.sun_direction = sun_direction
        self.panel_axis = self.tangent.copy()
        self.depth_axis = self.normal.copy()
        self._bounds_cache: tuple[float, float, float, float] | None = None

    def dispose(self) -> None:
        super().dispose()
        mesh = getattr(self, "_corner_backing_mesh", None)
        if mesh is not None:
            try:
                mesh.dispose()
            except Exception:
                pass
        self._corner_backing_mesh = None

    @classmethod
    def texture_or_load(cls, texture: Any | None = None) -> Any:
        if texture:
            return texture
        return load_texture(WINDOW_TEXTURE_PATH)

    @classmethod
    def from_building_spec(
        cls,
        spec: dict,
        *,
        window_spec: dict | None = None,
        texture: Any,
        backing_texture: Any | None = None,
        camera: object,
        ground_height_at: Callable[[float, float], float],
        lighting: Any | None = None,
        sun_direction: Any | None = None,
        wall_thickness: float = 2.5,
    ) -> "Window":
        window = window_spec or {}
        side = str(window.get("side", spec.get("window_side", "north"))).lower()
        wall_thickness = float(window.get("wall_thickness", spec.get("wall_thickness", wall_thickness)))
        normal = normal_for_side(side)
        tangent = axis_wall_tangent_for_normal(normal)
        center = spec["position"]
        half_x = float(spec["width"]) * 0.5
        half_z = float(spec["depth"]) * 0.5
        wall_half = max(0.0, float(wall_thickness)) * 0.5

        x = float(center.x)
        z = float(center.z)
        if abs(normal.x) > 0.0:
            x += normal.x * max(0.0, half_x - wall_half)
        else:
            z += normal.z * max(0.0, half_z - wall_half)

        offset = float(window.get("offset", spec.get("window_offset", 0.0)))
        x += tangent.x * offset
        z += tangent.z * offset

        window_height_value = window.get("height", spec.get("window_height", None))
        window_width_value = window.get("width", spec.get("window_width", None))
        if window_height_value is not None:
            opening_height = max(4.0, float(window_height_value))
            opening_width = opening_height * WINDOW_TEXTURE_ASPECT
        elif window_width_value is not None:
            opening_width = max(4.0, float(window_width_value))
            opening_height = opening_width / WINDOW_TEXTURE_ASPECT
        else:
            opening_width = WINDOW_DEFAULT_WIDTH
            opening_height = WINDOW_DEFAULT_HEIGHT

        visual_width = max(4.0, opening_width + WINDOW_FRAME_OVERLAP * 2.0)
        visual_height = max(4.0, visual_width / WINDOW_TEXTURE_ASPECT)
        base_y = spec.get("base_y", None)
        if base_y is None:
            base_y = ground_height_at(float(center.x), float(center.z))
        base_y = float(base_y)
        sill_height = max(
            0.0,
            float(window.get("sill_height", spec.get("window_sill_height", WINDOW_DEFAULT_SILL_HEIGHT))),
        )
        position = Vector3(x, base_y + sill_height + visual_height * 0.5, z)

        return cls(
            position,
            camera=camera,
            texture=texture,
            backing_texture=backing_texture,
            lighting=lighting,
            sun_direction=sun_direction,
            width=visual_width,
            height=visual_height,
            side=side,
            thickness=max(WINDOW_THICKNESS, wall_thickness),
            collision_width=visual_width,
            collision_height=visual_height,
            collision_thickness=max(WINDOW_THICKNESS, wall_thickness),
        )

    def _face_shade(self, face_idx: int) -> float:
        if face_idx == 0:
            base = 1.0
            normal = self._face_normal(face_idx)
        elif face_idx == 1:
            base = WINDOW_INTERIOR_SHADE
            normal = INDOOR_NORMAL
        elif face_idx == 4:
            base = WINDOW_TOP_SHADE
            normal = self._face_normal(face_idx)
        elif face_idx == 5:
            base = WINDOW_BOTTOM_SHADE
            normal = self._face_normal(face_idx)
        else:
            base = WINDOW_EDGE_SHADE
            normal = self._face_normal(face_idx)
        return max(0.0, min(1.0, base * self._sunlight_factor(normal)))

    def _face_uvs(self, face_idx: int) -> list[tuple[float, float]]:
        u0, v0, u1, v1 = self.uv_rect
        u_span = u1 - u0
        v_span = v1 - v0
        strip_u = u_span * WINDOW_EDGE_UV_FRACTION
        strip_v = v_span * WINDOW_EDGE_UV_FRACTION

        if face_idx == 0:
            return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        if face_idx == 1:
            return [(u1, v0), (u0, v0), (u0, v1), (u1, v1)]
        if face_idx == 2:
            return [(u0, v0), (u0 + strip_u, v0), (u0 + strip_u, v1), (u0, v1)]
        if face_idx == 3:
            return [(u1 - strip_u, v0), (u1, v0), (u1, v1), (u1 - strip_u, v1)]
        if face_idx == 4:
            return [(u0, v1 - strip_v), (u1, v1 - strip_v), (u1, v1), (u0, v1)]
        return [(u0, v0), (u1, v0), (u1, v0 + strip_v), (u0, v0 + strip_v)]

    def _corner_backing_quads(self, face_sign: float) -> list[tuple[Vector3, ...]]:
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        half_t = self.thickness * 0.5
        inset = min(0.05, half_t * 0.25)
        center = self.position + self.depth_axis * (face_sign * max(0.0, half_t - inset))
        up = Vector3(0.0, 1.0, 0.0)
        quads: list[tuple[Vector3, ...]] = []

        def point(x: float, y: float) -> Vector3:
            return center + self.panel_axis * x + up * y

        for y0_t, y1_t, clear_width_factor in WINDOW_CORNER_BACKING_BANDS:
            y0 = -half_h + self.height * y0_t
            y1 = -half_h + self.height * y1_t
            if y1 <= y0:
                continue

            clear_half_w = half_w * max(0.0, min(1.0, float(clear_width_factor)))
            if clear_half_w >= half_w - 1e-6:
                continue

            quads.append(
                (
                    point(-half_w, y0),
                    point(-clear_half_w, y0),
                    point(-clear_half_w, y1),
                    point(-half_w, y1),
                )
            )
            quads.append(
                (
                    point(clear_half_w, y0),
                    point(half_w, y0),
                    point(half_w, y1),
                    point(clear_half_w, y1),
                )
            )

        return quads

    def _wall_backing_uv(self, vertex: Vector3) -> tuple[float, float]:
        tex_size = self.backing_texture_size or (32, 32)
        tex_w = max(1.0, float(tex_size[0]))
        tex_h = max(1.0, float(tex_size[1]))
        along = float(vertex.z if abs(self.normal.x) > 0.0 else vertex.x)
        u = along / tex_w
        v = float(vertex.y) / tex_h

        u0, v0, u1, v1 = self.backing_uv_rect
        if (u0, v0, u1, v1) != (0.0, 0.0, 1.0, 1.0):
            u = u0 + (u % 1.0) * (u1 - u0)
            v = v0 + (v % 1.0) * (v1 - v0)
        return u, v

    def _draw_corner_backing(self) -> None:
        textured = bool(self.backing_texture)
        if textured:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.backing_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        else:
            glDisable(GL_TEXTURE_2D)
        glDisable(GL_ALPHA_TEST)
        glDisable(GL_BLEND)

        glBegin(GL_QUADS)
        try:
            for face_idx, face_sign in ((0, 1.0), (1, -1.0)):
                shade = self._face_shade(face_idx)
                if textured:
                    glColor4f(shade, shade, shade, 1.0)
                else:
                    r, g, b = WINDOW_CORNER_BACKING_COLOR
                    glColor4f(r * shade, g * shade, b * shade, 1.0)
                for quad in self._corner_backing_quads(face_sign):
                    for vertex in quad:
                        if textured:
                            u, v = self._wall_backing_uv(vertex)
                            glTexCoord2f(u, v)
                        glVertex3f(vertex.x, vertex.y, vertex.z)
        finally:
            glEnd()
            glColor4f(1.0, 1.0, 1.0, 1.0)
            if textured:
                glDisable(GL_TEXTURE_2D)

    def _corner_backing_vertex_data(self) -> np.ndarray:
        textured = bool(self.backing_texture)
        rows = []
        for face_idx, face_sign in ((0, 1.0), (1, -1.0)):
            shade = self._face_shade(face_idx)
            if textured:
                color = (shade, shade, shade)
            else:
                r, g, b = WINDOW_CORNER_BACKING_COLOR
                color = (r * shade, g * shade, b * shade)

            for quad in self._corner_backing_quads(face_sign):
                tri_vertices = (
                    quad[0],
                    quad[1],
                    quad[2],
                    quad[0],
                    quad[2],
                    quad[3],
                )
                for vertex in tri_vertices:
                    if textured:
                        u, v = self._wall_backing_uv(vertex)
                        rows.append(
                            (
                                vertex.x,
                                vertex.y,
                                vertex.z,
                                color[0],
                                color[1],
                                color[2],
                                u,
                                v,
                            )
                        )
                    else:
                        rows.append(
                            (
                                vertex.x,
                                vertex.y,
                                vertex.z,
                                color[0],
                                color[1],
                                color[2],
                            )
                        )

        columns = 8 if textured else 6
        if not rows:
            return np.zeros((0, columns), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def _draw_cached_corner_backing(self) -> None:
        mesh = getattr(self, "_corner_backing_mesh", None)
        light_key = self._slab_light_cache_key()
        dirty = light_key != getattr(self, "_corner_backing_light_key", None)

        if mesh is None or dirty:
            if mesh is not None:
                try:
                    mesh.dispose()
                except Exception:
                    pass
            vertex_data = self._corner_backing_vertex_data()
            if vertex_data.size == 0:
                self._corner_backing_mesh = None
                return

            if self.backing_texture:
                glBindTexture(GL_TEXTURE_2D, self.backing_texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

            self._corner_backing_mesh = BatchedMesh.from_vertex_data(
                vertex_data,
                texture=self.backing_texture if self.backing_texture else None,
                alpha_test=False,
                exposure_baseline=1.0,
                environment_lighting=False,
            )
            self._corner_backing_light_key = light_key
            mesh = self._corner_backing_mesh

        mesh.draw()

    def draw(self, camera=None) -> None:  # pragma: no cover - visual
        if not self.visible or not self.texture:
            return

        verts = self._visual_vertices()
        if not verts:
            return

        self._draw_cached_corner_backing()
        self._draw_cached_textured_slab_faces(verts)
