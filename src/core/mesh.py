"""Mesh and batching utilities extracted from renderer.py.

Provides BatchedMesh (VBO-backed mesh container) and GroundHeightSampler
which samples interpolated heights for the ground grid.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import ctypes
import math
import numpy as np
from OpenGL.GL import (
    glGenBuffers,
    glBindBuffer,
    glBufferData,
    glEnableClientState,
    glVertexPointer,
    glColorPointer,
    glTexCoordPointer,
    glDrawArrays,
    glDisableClientState,
    glBindTexture,
    glEnable,
    glDisable,
    GL_ARRAY_BUFFER,
    GL_STATIC_DRAW,
    GL_FLOAT,
    GL_TRIANGLES,
    GL_VERTEX_ARRAY,
    GL_COLOR_ARRAY,
    GL_TEXTURE_COORD_ARRAY,
    GL_TEXTURE_2D,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
)


@dataclass
class BatchedMesh:
    vbo_vertices: int
    vertex_count: int
    texture: int | None = None
    height_sampler: Optional[object] = None

    # Optional runtime data populated by builders that allows regenerating
    # the vertex buffer when the underlying height grid changes. These
    # attributes are set by the ground grid builder when creating a ground
    # mesh.
    _tile_vertex_array: Optional[np.ndarray] = None
    _count: Optional[int] = None
    _spacing: Optional[float] = None
    _half: Optional[float] = None

    def draw_untextured(self):
        if self.vertex_count == 0:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        stride = 6 * 4
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, stride, None)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(12))
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def draw(self):
        if self.vertex_count == 0:
            return

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)

        if self.texture is not None:
            stride = 8 * 4
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, None)
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(12))
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(24))

            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            from OpenGL.GL import glBlendFunc

            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBindTexture(GL_TEXTURE_2D, self.texture)

            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        else:
            self.draw_untextured()

    # --- Runtime editing helpers ---------------------------------

    def update_vertex_buffer_from_heights(self):
        """Rebuild the VBO from the stored tile template and the mesh's
        height sampler. This lets callers modify the sampler._heights array
        (or the sampler itself) and then refresh the GPU vertex buffer.

        If the builder did not attach the required metadata this becomes
        a no-op.
        """
        if self._tile_vertex_array is None or self._count is None:
            return

        tile_vertex_array = self._tile_vertex_array
        count = int(self._count)
        spacing = float(self._spacing)
        half = float(self._half)
        hs = self.height_sampler

        vertices = []
        for gx in range(count):
            for gz in range(count):
                tx = gx * spacing
                tz = gz * spacing
                translated = tile_vertex_array.copy()
                translated[:, 0] += tx
                translated[:, 2] += tz

                # Fill Y from height sampler for each vertex
                for i in range(len(translated)):
                    x = float(translated[i, 0])
                    z = float(translated[i, 2])
                    try:
                        translated[i, 1] = float(hs.height_at(x, z))
                    except Exception:
                        # Fallback to current value if sampler fails
                        pass

                vertices.append(translated)

        if not vertices:
            empty = np.zeros((0, 8), dtype=np.float32)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
            glBufferData(GL_ARRAY_BUFFER, empty.nbytes, empty, GL_STATIC_DRAW)
            self.vertex_count = 0
            return

        vertex_data = np.vstack(vertices)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        self.vertex_count = int(vertex_data.shape[0])

    def raise_area(
        self,
        center_x: float,
        center_z: float,
        radius: float,
        delta: float,
        *,
        smooth: bool = True,
    ):
        """Raise (or lower, if delta<0) corner heights inside a circular area.

        The corner influence falls off linearly with distance when smooth=True.
        After mutating the internal height array this calls
        `update_vertex_buffer_from_heights()`.
        """
        hs = self.height_sampler
        if not hasattr(hs, "_heights") or self._count is None:
            return
        heights = hs._heights
        count = int(self._count)
        spacing = float(self._spacing)
        half = float(self._half)

        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        for gx in range(count):
            for gz in range(count):
                base_x = gx * spacing
                base_z = gz * spacing
                for ci, (cx_off, cz_off) in enumerate(corners):
                    wx = base_x + cx_off
                    wz = base_z + cz_off
                    dist = math.hypot(wx - center_x, wz - center_z)
                    if dist > radius:
                        continue
                    if smooth and radius > 0:
                        t = max(0.0, 1.0 - (dist / radius))
                    else:
                        t = 1.0
                    heights[gx, gz, ci] = float(heights[gx, gz, ci] + delta * t)

        # Inform sampler and rebuild VBO
        self.height_sampler._heights = heights
        self.update_vertex_buffer_from_heights()

    def level_area(
        self,
        center_x: float,
        center_z: float,
        radius: float,
        target: float,
        *,
        smooth: bool = True,
    ):
        """Move corner heights toward `target` inside a circular area.

        Uses linear falloff when smooth=True. Fully sets corners at center to
        `target` and blends toward their original values near the radius.
        """
        hs = self.height_sampler
        if not hasattr(hs, "_heights") or self._count is None:
            return
        heights = hs._heights
        count = int(self._count)
        spacing = float(self._spacing)
        half = float(self._half)

        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        for gx in range(count):
            for gz in range(count):
                base_x = gx * spacing
                base_z = gz * spacing
                for ci, (cx_off, cz_off) in enumerate(corners):
                    wx = base_x + cx_off
                    wz = base_z + cz_off
                    dist = math.hypot(wx - center_x, wz - center_z)
                    if dist > radius:
                        continue
                    if smooth and radius > 0:
                        t = max(0.0, 1.0 - (dist / radius))
                    else:
                        t = 1.0
                    heights[gx, gz, ci] = float(
                        (1.0 - t) * heights[gx, gz, ci] + t * target
                    )

        self.height_sampler._heights = heights
        self.update_vertex_buffer_from_heights()

    def smooth_heights(self, iterations: int = 1):
        """Apply a simple smoothing pass over the corner heights.

        This performs a neighbor average (4-neighbor) smoothing `iterations`
        times and rebuilds the VBO.
        """
        hs = self.height_sampler
        if not hasattr(hs, "_heights") or self._count is None:
            return
        heights = hs._heights
        count = int(self._count)

        for _ in range(max(0, int(iterations))):
            new_h = heights.copy()
            for gx in range(count):
                for gz in range(count):
                    for ci in range(4):
                        acc = heights[gx, gz, ci]
                        n = 1
                        # check neighbors in grid (clamp at edges)
                        for ngx, ngz in (
                            (gx - 1, gz),
                            (gx + 1, gz),
                            (gx, gz - 1),
                            (gx, gz + 1),
                        ):
                            if 0 <= ngx < count and 0 <= ngz < count:
                                acc += heights[ngx, ngz, ci]
                                n += 1
                        new_h[gx, gz, ci] = acc / float(n)
            heights = new_h

        self.height_sampler._heights = heights
        self.update_vertex_buffer_from_heights()


class GroundHeightSampler:
    __slots__ = ("_count", "_spacing", "_w", "_heights")

    def __init__(self, count: int, spacing: float, half: float, heights: np.ndarray):
        self._count = count
        self._spacing = spacing
        self._w = half
        self._heights = heights

    @staticmethod
    def _barycentric_y(
        px: float,
        pz: float,
        x0: float,
        z0: float,
        y0: float,
        x1: float,
        z1: float,
        y1: float,
        x2: float,
        z2: float,
        y2: float,
    ) -> float:
        v0x, v0z = x1 - x0, z1 - z0
        v1x, v1z = x2 - x0, z2 - z0
        v2x, v2z = px - x0, pz - z0
        denom = v0x * v1z - v1x * v0z
        if abs(denom) < 1e-8:
            return (y0 + y1 + y2) / 3.0
        inv = 1.0 / denom
        u = (v2x * v1z - v1x * v2z) * inv
        v = (v0x * v2z - v2x * v0z) * inv
        w = 1.0 - u - v
        return u * y1 + v * y2 + w * y0

    def height_at(self, x: float, z: float) -> float:
        s = self._spacing
        half = self._w
        gx = int(math.floor((x + half) / s))
        gz = int(math.floor((z + half) / s))
        gx = max(0, min(self._count - 1, gx))
        gz = max(0, min(self._count - 1, gz))
        tx = gx * s
        tz = gz * s
        lx = x - tx
        lz = z - tz

        x_a, z_a = -half, -half
        x_b, z_b = +half, -half
        x_c, z_c = +half, +half
        x_d, z_d = -half, +half
        a_y, b_y, c_y, d_y = self._heights[gx, gz]

        ax, az = tx + x_a, tz + z_a
        bx, bz = tx + x_b, tz + z_b
        cx, cz = tx + x_c, tz + z_c
        dx, dz = tx + x_d, tz + z_d

        def barycentric_weights(px, pz, x0, z0, x1, z1, x2, z2):
            v0x, v0z = x1 - x0, z1 - z0
            v1x, v1z = x2 - x0, z2 - z0
            v2x, v2z = px - x0, pz - z0
            denom = v0x * v1z - v1x * v0z
            if abs(denom) < 1e-12:
                return None
            inv = 1.0 / denom
            u = (v2x * v1z - v1x * v2z) * inv
            v = (v0x * v2z - v2x * v0z) * inv
            w = 1.0 - u - v
            return (u, v, w)

        w_abc = barycentric_weights(x, z, ax, az, bx, bz, cx, cz)
        eps = -1e-6
        if w_abc is not None:
            u, v, w = w_abc
            if u >= eps and v >= eps and w >= eps:
                return u * float(b_y) + v * float(c_y) + w * float(a_y)

        w_acd = barycentric_weights(x, z, ax, az, cx, cz, dx, dz)
        if w_acd is not None:
            u, v, w = w_acd
            if u >= eps and v >= eps and w >= eps:
                return u * float(c_y) + v * float(d_y) + w * float(a_y)

        u_lin = (lx + half) / (2.0 * half)
        v_lin = (lz + half) / (2.0 * half)
        a = float(a_y)
        b = float(b_y)
        c = float(c_y)
        d = float(d_y)
        return (
            (1 - u_lin) * (1 - v_lin) * a
            + u_lin * (1 - v_lin) * b
            + u_lin * v_lin * c
            + (1 - u_lin) * v_lin * d
        )
