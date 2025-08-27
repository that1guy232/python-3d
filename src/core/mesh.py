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
    glBindBuffer,
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
    GL_FLOAT,
    GL_TRIANGLES,
    GL_VERTEX_ARRAY,
    GL_COLOR_ARRAY,
    GL_TEXTURE_COORD_ARRAY,
    GL_TEXTURE_2D,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    glBlendFunc,
    glTexEnvi,
    GL_TEXTURE_ENV_MODE,
    GL_MODULATE,
    GL_TEXTURE_ENV
    )



@dataclass
class BatchedMesh:
    vbo_vertices: int
    vertex_count: int
    texture: int | None = None
    height_sampler: Optional[object] = None


    def draw(self):
        if self.vertex_count == 0:
            return
            
        import time
        start_draw = time.perf_counter()
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
    
        if self.texture is not None:
            # Vertex format: [x, y, z, r, g, b, u, v] = 8 floats per vertex
            stride = 8 * 4  # 8 floats * 4 bytes per float = 32 bytes per vertex
            
            # Enable vertex arrays
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, None)  # Position at offset 0
            
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(3 * 4))  # Color at offset 3 floats (12 bytes)
            
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(6 * 4))  # UV at offset 6 floats (24 bytes)

            # Enable texturing and blending
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            glBindTexture(GL_TEXTURE_2D, self.texture)

            # Draw the mesh
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

            # Clean up
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        else:
            # Handle non-textured case (if needed)
            stride = 6 * 4  # Position (3) + Color (3) = 6 floats
            
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, None)
            
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(3 * 4))
            
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            
            glDisable(GL_BLEND)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

        end_draw = time.perf_counter()
        draw_duration = end_draw - start_draw
        print(f"Mesh draw time: {draw_duration:.6f} seconds")

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