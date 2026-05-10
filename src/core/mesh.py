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
    glDeleteBuffers,
    glEnableClientState,
    glVertexPointer,
    glColorPointer,
    glTexCoordPointer,
    glNormalPointer,
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
    GL_NORMAL_ARRAY,
    GL_TEXTURE_COORD_ARRAY,
    GL_TEXTURE_2D,
    GL_BLEND,
    GL_ALPHA_TEST,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_GREATER,
    glBlendFunc,
    glAlphaFunc,
    glTexEnvi,
    GL_TEXTURE_ENV_MODE,
    GL_MODULATE,
    GL_TEXTURE_ENV,
    GL_STATIC_DRAW,
    )

from core.compat_shader import get_texture_color_exposure_shader, use_fixed_pipeline



@dataclass
class BatchedMesh:
    vbo_vertices: int
    vertex_count: int
    texture: int | None = None
    height_sampler: Optional[object] = None
    alpha_test: bool = False
    environment_lighting: bool = True
    owns_vbo: bool = True
    exposure_baseline: float = 1.0
    vertex_width: int = 0
    shader_lighting: bool = False
    shine_enabled: bool = True
    _base_vertex_data: Optional[np.ndarray] = None
    _current_exposure: float = 1.0
    _vbo_exposure: float = 1.0

    @classmethod
    def from_vertex_data(
        cls,
        vertex_data: np.ndarray,
        *,
        texture: int | None = None,
        alpha_test: bool = False,
        height_sampler: Optional[object] = None,
        exposure_baseline: float = 1.0,
        keep_vertex_data: bool = True,
        environment_lighting: bool = True,
        shine_enabled: bool = True,
    ) -> "BatchedMesh":
        upload_data = np.ascontiguousarray(vertex_data, dtype=np.float32)
        source_width = int(upload_data.shape[1]) if upload_data.ndim == 2 else 0
        shader_lighting = bool(texture is not None and source_width >= 11)
        if texture is not None and shine_enabled and 8 <= source_width < 11:
            try:
                if get_texture_color_exposure_shader() is not None:
                    from engine.rendering.lighting import with_textured_normals

                    upload_data = with_textured_normals(
                        upload_data,
                        prefer_upward_normals=True,
                    )
            except Exception:
                upload_data = np.ascontiguousarray(vertex_data, dtype=np.float32)
        vbo = glGenBuffers(1)
        vbo_id = int(np.asarray(vbo).reshape(-1)[0])
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        glBufferData(GL_ARRAY_BUFFER, upload_data.nbytes, upload_data, GL_STATIC_DRAW)
        baseline = float(exposure_baseline)
        return cls(
            vbo_vertices=vbo_id,
            vertex_count=int(upload_data.shape[0]),
            texture=texture,
            height_sampler=height_sampler,
            alpha_test=alpha_test,
            environment_lighting=bool(environment_lighting),
            exposure_baseline=baseline,
            vertex_width=int(upload_data.shape[1]) if upload_data.ndim == 2 else 0,
            shader_lighting=shader_lighting,
            shine_enabled=bool(shine_enabled),
            _base_vertex_data=upload_data.copy() if keep_vertex_data else None,
            _current_exposure=baseline,
            _vbo_exposure=baseline,
        )

    def _exposure_scale(self, exposure: float | None = None) -> float:
        exposure_value = self._current_exposure if exposure is None else float(exposure)
        base = float(self.exposure_baseline)
        if abs(base) <= 1e-8:
            return exposure_value
        return exposure_value / base

    def _upload_vertex_data_for_exposure(self, exposure: float) -> None:
        if self.vertex_count == 0 or not self.vbo_vertices:
            return
        if self._base_vertex_data is None:
            return

        exposure_value = float(exposure)
        if abs(self._vbo_exposure - exposure_value) <= 1e-6:
            return

        vertex_data = self._base_vertex_data.copy()
        if vertex_data.shape[1] >= 6:
            vertex_data[:, 3:6] = self._base_vertex_data[:, 3:6] * self._exposure_scale(
                exposure_value
            )

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        self._vbo_exposure = exposure_value

    def _restore_baseline_vertex_data(self) -> None:
        if self.vertex_count == 0 or not self.vbo_vertices:
            return
        if self._base_vertex_data is None:
            return

        baseline = float(self.exposure_baseline)
        if abs(self._vbo_exposure - baseline) <= 1e-6:
            return

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(
            GL_ARRAY_BUFFER,
            self._base_vertex_data.nbytes,
            self._base_vertex_data,
            GL_STATIC_DRAW,
        )
        self._vbo_exposure = baseline

    def set_exposure(self, exposure: float, *, baseline: float | None = None) -> None:
        """Apply a global exposure scale to stored vertex colors.

        Textured meshes drawn through ``draw()`` use a small compatibility shader
        for this scale. Other paths keep the old CPU-side color upload.
        """
        if baseline is not None:
            self.exposure_baseline = float(baseline)

        exposure_value = float(exposure)
        if abs(self._current_exposure - exposure_value) <= 1e-6 and baseline is None:
            return

        self._current_exposure = exposure_value
        if self.texture is not None:
            return

        self._upload_vertex_data_for_exposure(exposure_value)

    def dispose(self) -> None:
        """Release the owned OpenGL vertex buffer, if it still exists."""
        vbo = int(self.vbo_vertices or 0)
        if not self.owns_vbo or vbo == 0:
            self.vbo_vertices = 0
            self.vertex_count = 0
            self.vertex_width = 0
            return

        try:
            glDeleteBuffers(1, [vbo])
        except TypeError:
            try:
                glDeleteBuffers([vbo])
            except Exception:
                pass
        except Exception:
            pass
        finally:
            self.vbo_vertices = 0
            self.vertex_count = 0
            self.vertex_width = 0
            self.owns_vbo = False
            self._base_vertex_data = None

    def draw(self):
        if self.vertex_count == 0 or not self.vbo_vertices:
            return

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
    
        if self.texture is not None:
            shader = get_texture_color_exposure_shader()
            if shader is not None:
                self._restore_baseline_vertex_data()
            else:
                self._upload_vertex_data_for_exposure(self._current_exposure)

            # Supported textured formats:
            # [x, y, z, r, g, b, u, v]
            # [x, y, z, r, g, b, nx, ny, nz, u, v]
            vertex_width = self.vertex_width or 8
            has_normals = vertex_width >= 11
            shader_lighting_enabled = has_normals and self.shader_lighting
            stride = vertex_width * 4
            
            # Enable vertex arrays
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, None)  # Position at offset 0
            
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(3 * 4))  # Color at offset 3 floats (12 bytes)

            if has_normals:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(6 * 4))
            
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            uv_offset = 9 if has_normals else 6
            glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(uv_offset * 4))

            # Enable texturing and blending
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            if self.alpha_test:
                glEnable(GL_ALPHA_TEST)
                glAlphaFunc(GL_GREATER, 0.01)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            glBindTexture(GL_TEXTURE_2D, self.texture)

            # Draw the mesh
            if shader is not None:
                shader.bind(
                    scene_lighting_enabled=shader_lighting_enabled,
                    directional_enabled=shader_lighting_enabled,
                    environment_enabled=(
                        shader_lighting_enabled and self.environment_lighting
                    ),
                    shine_enabled=has_normals and self.shine_enabled,
                )
            try:
                glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            finally:
                if shader is not None:
                    use_fixed_pipeline()

            # Clean up
            if self.alpha_test:
                glDisable(GL_ALPHA_TEST)
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            if has_normals:
                glDisableClientState(GL_NORMAL_ARRAY)
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

    def draw_textured_prepared(self, *, bind_texture: bool = True):
        """Draw a textured VBO while caller owns shared GL state.

        Decal batches can draw many small meshes in a row; setting texture,
        blend, and client-array state once around the loop avoids a surprising
        amount of driver churn.
        """
        if self.vertex_count == 0 or not self.vbo_vertices or self.texture is None:
            return

        vertex_width = self.vertex_width or 8
        has_normals = vertex_width >= 11
        stride = vertex_width * 4
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glVertexPointer(3, GL_FLOAT, stride, None)
        glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(3 * 4))
        if has_normals:
            glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(6 * 4))
        uv_offset = 9 if has_normals else 6
        glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(uv_offset * 4))
        if bind_texture:
            glBindTexture(GL_TEXTURE_2D, self.texture)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

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
