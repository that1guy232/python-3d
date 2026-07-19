"""Mesh and batching utilities extracted from renderer.py.

Provides BatchedMesh (VBO-backed mesh container) and GroundHeightSampler
which samples interpolated heights for the ground grid.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
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
    glClientActiveTexture,
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
    GL_TEXTURE0,
    GL_TEXTURE1,
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

from engine.lighting_receiver import LightingReceiver, ReceiverShaderFlags
from engine.core.gl_state import use_fixed_pipeline
from engine.core.legacy_shader_adapter import get_legacy_texture_shader

if TYPE_CHECKING:
    from engine.rendering.lighting_adapter import ReceiverLightingPacket
    from engine.rendering.packet_shader import PacketTextureLightingShader
    from engine.rendering.directional_shadow import ShadowCasterShader
    from engine.rendering.point_shadow import PointShadowCasterShader


@dataclass
class BatchedMesh:
    vbo_vertices: int
    vertex_count: int
    texture: int | None = None
    height_sampler: Optional[object] = None
    alpha_test: bool = False
    alpha_cutoff: float = 0.01
    casts_shadows: bool = True
    casts_sun_shadows: bool = True
    environment_lighting: bool = True
    owns_vbo: bool = True
    exposure_baseline: float = 1.0
    vertex_width: int = 0
    shader_lighting: bool = False
    shine_enabled: bool = True
    lighting_receiver: LightingReceiver | None = None
    draw_mode: int = GL_TRIANGLES
    bounds_center: tuple[float, float, float] | None = None
    bounds_radius: float = 0.0
    _base_vertex_data: Optional[np.ndarray] = None
    _current_exposure: float = 1.0
    _vbo_exposure: float = 1.0
    _vertex_stride: int = field(init=False, repr=False)
    _has_normals: bool = field(init=False, repr=False)
    _has_directional_normals: bool = field(init=False, repr=False)
    _color_offset_ptr: ctypes.c_void_p = field(init=False, repr=False)
    _normal_offset_ptr: ctypes.c_void_p = field(init=False, repr=False)
    _directional_normal_offset_ptr: ctypes.c_void_p = field(init=False, repr=False)
    _uv_offset_ptr: ctypes.c_void_p = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._refresh_vertex_layout()

    def _refresh_vertex_layout(self) -> None:
        vertex_width = int(self.vertex_width or 8)
        has_directional_normals = vertex_width >= 14
        has_normals = vertex_width >= 11
        uv_offset = 12 if has_directional_normals else 9 if has_normals else 6
        self._vertex_stride = vertex_width * 4
        self._has_normals = has_normals
        self._has_directional_normals = has_directional_normals
        self._color_offset_ptr = ctypes.c_void_p(3 * 4)
        self._normal_offset_ptr = ctypes.c_void_p(6 * 4)
        self._directional_normal_offset_ptr = ctypes.c_void_p(9 * 4)
        self._uv_offset_ptr = ctypes.c_void_p(uv_offset * 4)

    @staticmethod
    def _bounds_from_vertex_data(
        vertex_data: np.ndarray,
    ) -> tuple[tuple[float, float, float] | None, float]:
        if (
            vertex_data.ndim != 2
            or vertex_data.shape[0] == 0
            or vertex_data.shape[1] < 3
        ):
            return None, 0.0

        positions = np.asarray(vertex_data[:, 0:3], dtype=np.float32)
        if positions.size == 0:
            return None, 0.0

        finite = np.isfinite(positions).all(axis=1)
        if not np.any(finite):
            return None, 0.0
        positions = positions[finite]
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        center = (mins + maxs) * 0.5
        radius = float(np.linalg.norm((maxs - mins) * 0.5))
        return (float(center[0]), float(center[1]), float(center[2])), radius

    @classmethod
    def from_vertex_data(
        cls,
        vertex_data: np.ndarray,
        *,
        texture: int | None = None,
        alpha_test: bool = False,
        alpha_cutoff: float = 0.01,
        casts_shadows: bool = True,
        casts_sun_shadows: bool = True,
        height_sampler: Optional[object] = None,
        exposure_baseline: float = 1.0,
        keep_vertex_data: bool = True,
        environment_lighting: bool = True,
        shine_enabled: bool = True,
        draw_mode: int = GL_TRIANGLES,
        shader_lighting: bool | None = None,
        lighting_receiver: LightingReceiver | None = None,
    ) -> "BatchedMesh":
        upload_data = np.ascontiguousarray(vertex_data, dtype=np.float32)
        source_width = int(upload_data.shape[1]) if upload_data.ndim == 2 else 0
        computed_shader_lighting = bool(texture is not None and source_width >= 11)
        if shader_lighting is not None:
            computed_shader_lighting = bool(shader_lighting)
        if texture is not None and shine_enabled and 8 <= source_width < 11:
            try:
                if get_legacy_texture_shader() is not None:
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
        bounds_center, bounds_radius = cls._bounds_from_vertex_data(upload_data)
        return cls(
            vbo_vertices=vbo_id,
            vertex_count=int(upload_data.shape[0]),
            texture=texture,
            height_sampler=height_sampler,
            alpha_test=alpha_test,
            alpha_cutoff=max(0.0, min(1.0, float(alpha_cutoff))),
            casts_shadows=bool(casts_shadows),
            casts_sun_shadows=bool(casts_sun_shadows),
            environment_lighting=bool(environment_lighting),
            exposure_baseline=baseline,
            vertex_width=int(upload_data.shape[1]) if upload_data.ndim == 2 else 0,
            shader_lighting=computed_shader_lighting,
            shine_enabled=bool(shine_enabled),
            lighting_receiver=lighting_receiver,
            draw_mode=int(draw_mode),
            bounds_center=bounds_center,
            bounds_radius=bounds_radius,
            _base_vertex_data=upload_data.copy() if keep_vertex_data else None,
            _current_exposure=baseline,
            _vbo_exposure=baseline,
        )

    def shadow_meshes(self, camera=None) -> tuple["BatchedMesh", ...]:
        """Expose this mesh through the shared shadow-submission protocol."""

        return (self,) if self.casts_shadows and self.vertex_count > 0 else ()

    def draw_shadow(
        self,
        caster_shader: "ShadowCasterShader",
        light_matrix: tuple[float, ...],
    ) -> None:
        """Draw geometry into a light-space depth map."""

        if not self.casts_shadows or self.vertex_count <= 0:
            return
        cutout = bool(self.alpha_test and self.texture)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, self._vertex_stride, ctypes.c_void_p(0))
        if cutout:
            glClientActiveTexture(GL_TEXTURE0)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(2, GL_FLOAT, self._vertex_stride, self._uv_offset_ptr)
        caster_shader.bind(
            light_matrix,
            texture=int(self.texture or 0),
            alpha_cutout=cutout,
            alpha_cutoff=self.alpha_cutoff,
        )
        try:
            glDrawArrays(self.draw_mode, 0, self.vertex_count)
        finally:
            use_fixed_pipeline()
            if cutout:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw_point_shadow(
        self,
        caster_shader: "PointShadowCasterShader",
        light_matrix: tuple[float, ...],
        light_position,
        light_range: float,
    ) -> None:
        """Draw geometry into one face of a radial point-light depth cube."""

        if not self.casts_shadows or self.vertex_count <= 0:
            return
        cutout = bool(self.alpha_test and self.texture)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, self._vertex_stride, ctypes.c_void_p(0))
        if cutout:
            glClientActiveTexture(GL_TEXTURE0)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(2, GL_FLOAT, self._vertex_stride, self._uv_offset_ptr)
        caster_shader.bind(
            light_matrix,
            light_position,
            light_range,
            texture=int(self.texture or 0),
            alpha_cutout=cutout,
            alpha_cutoff=self.alpha_cutoff,
        )
        try:
            glDrawArrays(self.draw_mode, 0, self.vertex_count)
        finally:
            use_fixed_pipeline()
            if cutout:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def receiver_shader_flags(self) -> ReceiverShaderFlags:
        """Return explicit flags, or project the legacy implicit mesh policy."""

        if self.lighting_receiver is not None:
            return self.lighting_receiver.compatibility_shader_flags(
                has_normals=self._has_normals,
            )
        shader_lighting = self._has_normals and self.shader_lighting
        return ReceiverShaderFlags(
            scene_lighting=shader_lighting,
            directional=shader_lighting,
            environment=shader_lighting and self.environment_lighting,
            fog=True,
            shine=self._has_normals and self.shine_enabled,
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

    def is_visible(self, camera=None, *, view_distance: float | None = None) -> bool:
        if camera is None or self.bounds_center is None:
            return True

        tester = getattr(camera, "sphere_in_frustum", None)
        if callable(tester):
            return bool(
                tester(
                    self.bounds_center,
                    self.bounds_radius,
                    far_distance=view_distance,
                )
            )

        cam_pos = getattr(camera, "position", None)
        if cam_pos is None or view_distance is None:
            return True

        dx = self.bounds_center[0] - float(cam_pos.x)
        dy = self.bounds_center[1] - float(cam_pos.y)
        dz = self.bounds_center[2] - float(cam_pos.z)
        max_dist = float(view_distance) + float(self.bounds_radius)
        return (dx * dx + dy * dy + dz * dz) <= max_dist * max_dist

    def dispose(self) -> None:
        """Release the owned OpenGL vertex buffer, if it still exists."""
        vbo = int(self.vbo_vertices or 0)
        if not self.owns_vbo or vbo == 0:
            self.vbo_vertices = 0
            self.vertex_count = 0
            self.vertex_width = 0
            self.bounds_center = None
            self.bounds_radius = 0.0
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
            self.bounds_center = None
            self.bounds_radius = 0.0

    def draw(
        self,
        camera=None,
        *,
        view_distance: float | None = None,
        lighting_packet: "ReceiverLightingPacket | None" = None,
        packet_shader: "PacketTextureLightingShader | None" = None,
    ):
        if self.vertex_count == 0 or not self.vbo_vertices:
            return
        if not self.is_visible(camera, view_distance=view_distance):
            return

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)

        if self.texture is not None:
            use_packet_shader = packet_shader is not None and lighting_packet is not None
            shader = (
                None if use_packet_shader else get_legacy_texture_shader()
            )
            if use_packet_shader or shader is not None:
                self._restore_baseline_vertex_data()
            else:
                self._upload_vertex_data_for_exposure(self._current_exposure)

            # Supported textured formats:
            # [x, y, z, r, g, b, u, v]
            # [x, y, z, r, g, b, nx, ny, nz, u, v]
            has_normals = self._has_normals
            receiver_flags = (
                None if use_packet_shader else self.receiver_shader_flags()
            )
            stride = self._vertex_stride

            # Enable vertex arrays
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, None)  # Position at offset 0

            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(
                3, GL_FLOAT, stride, self._color_offset_ptr
            )  # Color at offset 3 floats (12 bytes)

            if has_normals:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, stride, self._normal_offset_ptr)

            glClientActiveTexture(GL_TEXTURE0)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(2, GL_FLOAT, stride, self._uv_offset_ptr)
            if self._has_directional_normals:
                glClientActiveTexture(GL_TEXTURE1)
                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glTexCoordPointer(
                    3,
                    GL_FLOAT,
                    stride,
                    self._directional_normal_offset_ptr,
                )
                glClientActiveTexture(GL_TEXTURE0)

            # Enable texturing and blending
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            if self.alpha_test:
                glEnable(GL_ALPHA_TEST)
                glAlphaFunc(GL_GREATER, self.alpha_cutoff)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            glBindTexture(GL_TEXTURE_2D, self.texture)

            # Draw the mesh
            if use_packet_shader:
                packet_shader.bind(
                    lighting_packet,
                    directional_normal_stream=self._has_directional_normals,
                )
            elif shader is not None and receiver_flags is not None:
                shader.bind(
                    scene_lighting_enabled=receiver_flags.scene_lighting,
                    directional_enabled=receiver_flags.directional,
                    environment_enabled=receiver_flags.environment,
                    fog_enabled=receiver_flags.fog,
                    shine_enabled=receiver_flags.shine,
                )
            try:
                glDrawArrays(self.draw_mode, 0, self.vertex_count)
            finally:
                if use_packet_shader or shader is not None:
                    use_fixed_pipeline()

            # Clean up
            if self.alpha_test:
                glDisable(GL_ALPHA_TEST)
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)
            if self._has_directional_normals:
                glClientActiveTexture(GL_TEXTURE1)
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
                glClientActiveTexture(GL_TEXTURE0)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            if has_normals:
                glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        else:
            # Handle non-textured case (if needed)
            stride = self._vertex_stride  # Position (3) + Color (3) = 6 floats

            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, None)

            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, stride, self._color_offset_ptr)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            glDrawArrays(self.draw_mode, 0, self.vertex_count)

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

        has_normals = self._has_normals
        stride = self._vertex_stride
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glVertexPointer(3, GL_FLOAT, stride, None)
        glColorPointer(3, GL_FLOAT, stride, self._color_offset_ptr)
        if has_normals:
            glNormalPointer(GL_FLOAT, stride, self._normal_offset_ptr)
        glClientActiveTexture(GL_TEXTURE0)
        glTexCoordPointer(2, GL_FLOAT, stride, self._uv_offset_ptr)
        if self._has_directional_normals:
            glClientActiveTexture(GL_TEXTURE1)
            glTexCoordPointer(
                3,
                GL_FLOAT,
                stride,
                self._directional_normal_offset_ptr,
            )
            glClientActiveTexture(GL_TEXTURE0)
        if bind_texture:
            glBindTexture(GL_TEXTURE_2D, self.texture)
        glDrawArrays(self.draw_mode, 0, self.vertex_count)

    @staticmethod
    def _prepared_draw_key(mesh: "BatchedMesh", shader) -> tuple:
        receiver_flags = mesh.receiver_shader_flags()
        if shader is None:
            return (
                bool(mesh.alpha_test),
                mesh._has_normals,
                mesh._has_directional_normals,
                False,
                False,
                False,
                False,
                False,
            )
        return (
            bool(mesh.alpha_test),
            mesh._has_normals,
            mesh._has_directional_normals,
            receiver_flags.scene_lighting,
            receiver_flags.directional,
            receiver_flags.environment,
            receiver_flags.fog,
            receiver_flags.shine,
        )

    @staticmethod
    def _draw_textured_prepared_run(
        meshes: list["BatchedMesh"],
        key: tuple,
        shader,
        *,
        lighting_packet: "ReceiverLightingPacket | None" = None,
        packet_shader: "PacketTextureLightingShader | None" = None,
    ) -> None:
        if not meshes:
            return

        (
            alpha_test,
            has_normals,
            has_directional_normals,
            scene_lighting,
            directional,
            environment_lighting,
            fog_enabled,
            shine_enabled,
        ) = key
        alpha_cutoff = float(meshes[0].alpha_cutoff)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glClientActiveTexture(GL_TEXTURE0)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        if has_directional_normals:
            glClientActiveTexture(GL_TEXTURE1)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glClientActiveTexture(GL_TEXTURE0)
        if has_normals:
            glEnableClientState(GL_NORMAL_ARRAY)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        if alpha_test:
            glEnable(GL_ALPHA_TEST)
            glAlphaFunc(GL_GREATER, alpha_cutoff)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        use_packet_shader = packet_shader is not None and lighting_packet is not None
        if use_packet_shader:
            packet_shader.bind(
                lighting_packet,
                directional_normal_stream=has_directional_normals,
            )
        elif shader is not None:
            shader.bind(
                scene_lighting_enabled=scene_lighting,
                directional_enabled=directional,
                environment_enabled=environment_lighting,
                fog_enabled=fog_enabled,
                shine_enabled=shine_enabled,
            )

        bound_texture = None
        try:
            for mesh in meshes:
                if use_packet_shader or shader is not None:
                    mesh._restore_baseline_vertex_data()
                else:
                    mesh._upload_vertex_data_for_exposure(mesh._current_exposure)

                texture = int(mesh.texture or 0)
                if texture != bound_texture:
                    glBindTexture(GL_TEXTURE_2D, texture)
                    bound_texture = texture
                mesh.draw_textured_prepared(bind_texture=False)
        finally:
            if alpha_test:
                glDisable(GL_ALPHA_TEST)
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)
            if has_normals:
                glDisableClientState(GL_NORMAL_ARRAY)
            if has_directional_normals:
                glClientActiveTexture(GL_TEXTURE1)
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
                glClientActiveTexture(GL_TEXTURE0)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

    @staticmethod
    def draw_many(
        meshes,
        *,
        camera=None,
        view_distance: float | None = None,
        lighting_packets: dict[str, "ReceiverLightingPacket"] | None = None,
        packet_shader: "PacketTextureLightingShader | None" = None,
        require_lighting_packets: bool = False,
    ) -> None:
        """Draw a sequence of meshes while sharing GL state across adjacent VBOs."""
        shader = None
        legacy_shader_loaded = False
        run: list[BatchedMesh] = []
        run_key = None
        run_packet = None

        def flush_run() -> None:
            nonlocal run, run_key, run_packet
            if not run:
                return
            try:
                BatchedMesh._draw_textured_prepared_run(
                    run,
                    run_key,
                    shader,
                    lighting_packet=run_packet,
                    packet_shader=packet_shader if run_packet is not None else None,
                )
            finally:
                if run_packet is not None or shader is not None:
                    use_fixed_pipeline()
                run = []
                run_key = None
                run_packet = None

        for mesh in meshes or ():
            if mesh is None:
                continue
            if (
                mesh.vertex_count == 0
                or not mesh.vbo_vertices
                or not mesh.is_visible(camera, view_distance=view_distance)
            ):
                continue
            if mesh.texture is None:
                flush_run()
                mesh.draw(camera=camera, view_distance=view_distance)
                continue

            packet = None
            receiver = mesh.lighting_receiver
            if packet_shader is not None and receiver is not None:
                packet = (lighting_packets or {}).get(receiver.receiver_id)
                if packet is None and require_lighting_packets:
                    raise RuntimeError(
                        "packet lighting backend has no packet for receiver "
                        f"{receiver.receiver_id!r}"
                    )
            if packet is not None:
                key = (
                    bool(mesh.alpha_test),
                    mesh._has_normals,
                    mesh._has_directional_normals,
                    bool(receiver.local),
                    bool(receiver.directional),
                    bool(receiver.environment),
                    bool(receiver.fog),
                    bool(receiver.shine),
                )
            else:
                if not legacy_shader_loaded:
                    shader = get_legacy_texture_shader()
                    legacy_shader_loaded = True
                key = BatchedMesh._prepared_draw_key(mesh, shader)
            if run and (key != run_key or packet != run_packet):
                flush_run()
            run.append(mesh)
            run_key = key
            run_packet = packet

        flush_run()


class GroundHeightSampler:
    __slots__ = (
        "_count",
        "_spacing",
        "_w",
        "_heights",
        "_height_adjustments",
        "_surface_triangles",
        "_surface_cell_offsets",
    )

    def __init__(
        self,
        count: int,
        spacing: float,
        half: float,
        heights: np.ndarray,
        height_adjustments=None,
        surface_vertices: np.ndarray | None = None,
    ):
        self._count = count
        self._spacing = spacing
        self._w = half
        self._heights = heights
        adjustments = []
        for adjustment in height_adjustments or ():
            normalized = self._normalize_height_adjustment(adjustment)
            if normalized is not None:
                adjustments.append(normalized)
        self._height_adjustments = tuple(adjustments)
        (
            self._surface_triangles,
            self._surface_cell_offsets,
        ) = self._build_surface_index(surface_vertices)

    def _build_surface_index(
        self,
        surface_vertices: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if surface_vertices is None:
            return None, None

        positions = np.asarray(surface_vertices, dtype=np.float32)
        triangle_vertex_count = (len(positions) // 3) * 3
        if triangle_vertex_count <= 0 or positions.ndim != 2 or positions.shape[1] < 3:
            return None, None

        triangles = positions[:triangle_vertex_count, :3].reshape(-1, 3, 3)
        centers = triangles[:, :, (0, 2)].mean(axis=1)
        cell_x = np.floor((centers[:, 0] + self._w) / self._spacing).astype(np.int64)
        cell_z = np.floor((centers[:, 1] + self._w) / self._spacing).astype(np.int64)
        cell_x = np.clip(cell_x, 0, self._count - 1)
        cell_z = np.clip(cell_z, 0, self._count - 1)
        cell_ids = cell_x * self._count + cell_z

        order = np.argsort(cell_ids, kind="stable")
        sorted_triangles = np.ascontiguousarray(triangles[order], dtype=np.float32)
        cell_count = self._count * self._count
        counts = np.bincount(cell_ids, minlength=cell_count)
        offsets = np.zeros(cell_count + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return sorted_triangles, offsets

    def _surface_height_at(
        self,
        x: float,
        z: float,
        gx: int,
        gz: int,
    ) -> float | None:
        triangles = self._surface_triangles
        offsets = self._surface_cell_offsets
        if triangles is None or offsets is None:
            return None

        cell_id = gx * self._count + gz
        start = int(offsets[cell_id])
        end = int(offsets[cell_id + 1])
        if start >= end:
            return None

        candidates = triangles[start:end]
        x0 = candidates[:, 0, 0]
        z0 = candidates[:, 0, 2]
        x1 = candidates[:, 1, 0]
        z1 = candidates[:, 1, 2]
        x2 = candidates[:, 2, 0]
        z2 = candidates[:, 2, 2]

        v0x = x1 - x0
        v0z = z1 - z0
        v1x = x2 - x0
        v1z = z2 - z0
        v2x = float(x) - x0
        v2z = float(z) - z0
        denom = v0x * v1z - v1x * v0z
        valid = np.abs(denom) > 1e-8
        safe_denom = np.where(valid, denom, 1.0)
        first_weight = (v2x * v1z - v1x * v2z) / safe_denom
        second_weight = (v0x * v2z - v2x * v0z) / safe_denom
        third_weight = 1.0 - first_weight - second_weight
        epsilon = -1e-5
        inside = (
            valid
            & (first_weight >= epsilon)
            & (second_weight >= epsilon)
            & (third_weight >= epsilon)
        )
        matches = np.flatnonzero(inside)
        if len(matches) == 0:
            return None

        index = int(matches[0])
        return float(
            first_weight[index] * candidates[index, 1, 1]
            + second_weight[index] * candidates[index, 2, 1]
            + third_weight[index] * candidates[index, 0, 1]
        )

    @staticmethod
    def _normalize_height_adjustment(adjustment):
        try:
            min_x, max_x, min_z, max_z, height, blend_margin = adjustment
            min_x = float(min_x)
            max_x = float(max_x)
            min_z = float(min_z)
            max_z = float(max_z)
            if max_x < min_x:
                min_x, max_x = max_x, min_x
            if max_z < min_z:
                min_z, max_z = max_z, min_z
            return (
                min_x,
                max_x,
                min_z,
                max_z,
                float(height),
                max(0.0, float(blend_margin)),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _smooth01(value: float) -> float:
        value = max(0.0, min(1.0, float(value)))
        return value * value * (3.0 - 2.0 * value)

    @classmethod
    def _rect_blend_influence(
        cls,
        x: float,
        z: float,
        *,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        blend_margin: float,
    ) -> float:
        if min_x <= x <= max_x and min_z <= z <= max_z:
            return 1.0
        if blend_margin <= 1e-6:
            return 0.0

        dx = max(min_x - x, 0.0, x - max_x)
        dz = max(min_z - z, 0.0, z - max_z)
        distance = math.hypot(dx, dz)
        if distance >= blend_margin:
            return 0.0
        return 1.0 - cls._smooth01(distance / blend_margin)

    def _apply_height_adjustments(self, x: float, z: float, height: float) -> float:
        if not self._height_adjustments:
            return float(height)

        adjusted = float(height)
        px = float(x)
        pz = float(z)
        for (
            min_x,
            max_x,
            min_z,
            max_z,
            target_y,
            blend_margin,
        ) in self._height_adjustments:
            influence = self._rect_blend_influence(
                px,
                pz,
                min_x=min_x,
                max_x=max_x,
                min_z=min_z,
                max_z=max_z,
                blend_margin=blend_margin,
            )
            if influence <= 0.0:
                continue
            adjusted += (target_y - adjusted) * influence
        return adjusted

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

        surface_height = self._surface_height_at(x, z, gx, gz)
        if surface_height is not None:
            return surface_height

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
        base_height = None
        if w_abc is not None:
            u, v, w = w_abc
            if u >= eps and v >= eps and w >= eps:
                base_height = u * float(b_y) + v * float(c_y) + w * float(a_y)

        if base_height is None:
            w_acd = barycentric_weights(x, z, ax, az, cx, cz, dx, dz)
            if w_acd is not None:
                u, v, w = w_acd
                if u >= eps and v >= eps and w >= eps:
                    base_height = u * float(c_y) + v * float(d_y) + w * float(a_y)

        if base_height is None:
            u_lin = (lx + half) / (2.0 * half)
            v_lin = (lz + half) / (2.0 * half)
            a = float(a_y)
            b = float(b_y)
            c = float(c_y)
            d = float(d_y)
            base_height = (
                (1 - u_lin) * (1 - v_lin) * a
                + u_lin * (1 - v_lin) * b
                + u_lin * v_lin * c
                + (1 - u_lin) * v_lin * d
            )

        return self._apply_height_adjustments(x, z, base_height)
