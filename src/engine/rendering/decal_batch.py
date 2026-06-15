"""Decal batching: merge many terrain-conforming decals into a single VBO per texture.

This builds one BatchedMesh per texture from many per-sprite decals by
concatenating their vertex data. Use for static decals that don't move each frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Iterable
import ctypes
import math
import numpy as np
from OpenGL.GL import (
    glBindTexture,
    glBindBuffer,
    glEnable,
    glDisable,
    glEnableClientState,
    glDisableClientState,
    glDepthMask,
    glBlendFunc,
    glTexEnvi,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_ENV,
    GL_TEXTURE_ENV_MODE,
    GL_MODULATE,
    GL_ARRAY_BUFFER,
    GL_FLOAT,
    GL_QUADS,
    GL_TRIANGLES,
    GL_VERTEX_ARRAY,
    GL_COLOR_ARRAY,
    GL_TEXTURE_COORD_ARRAY,
    glColorPointer,
    glDrawArrays,
    glMultiDrawArrays,
    glTexCoordPointer,
    glVertexPointer,
)

from engine.core.mesh import BatchedMesh
from engine.rendering.decal import Decal
from engine.config import HEIGHT, VIEWDISTANCE, WIDTH


@dataclass
class DecalBatch:
    meshes_by_tex: Dict[tuple, BatchedMesh] = field(default_factory=dict)
    # Precomputed center per texture (cx, cy, cz) used for cheap distance culling
    mesh_centers: Dict[tuple, tuple[float, float, float]] = field(default_factory=dict)
    mesh_radii: Dict[tuple, float] = field(default_factory=dict)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    draw_items: tuple = field(default_factory=tuple)
    draw_groups: tuple = field(default_factory=tuple)
    draw_mode: int = GL_TRIANGLES
    _use_multi_draw: bool = True

    def __post_init__(self) -> None:
        self._refresh_draw_items()

    def _refresh_draw_items(self) -> None:
        color_offset_ptr = ctypes.c_void_p(3 * 4)
        uv_offset_ptr = ctypes.c_void_p(6 * 4)
        self.draw_items = tuple(
            (
                self.mesh_centers.get(key),
                float(self.mesh_radii.get(key, 0.0)),
                int(getattr(mesh, "texture", 0) or 0),
                int(getattr(mesh, "vbo_vertices", 0) or 0),
                int(getattr(mesh, "vertex_count", 0) or 0),
                int(getattr(mesh, "_vertex_stride", 8 * 4) or 8 * 4),
                getattr(mesh, "_color_offset_ptr", color_offset_ptr),
                getattr(mesh, "_uv_offset_ptr", uv_offset_ptr),
            )
            for key, mesh in self.meshes_by_tex.items()
        )
        self.draw_groups = tuple(
            (
                int(getattr(mesh, "texture", 0) or 0),
                int(getattr(mesh, "vbo_vertices", 0) or 0),
                int(getattr(mesh, "_vertex_stride", 8 * 4) or 8 * 4),
                getattr(mesh, "_color_offset_ptr", color_offset_ptr),
                getattr(mesh, "_uv_offset_ptr", uv_offset_ptr),
                (
                    (
                        self.mesh_centers.get(key),
                        float(self.mesh_radii.get(key, 0.0)),
                        0,
                        int(getattr(mesh, "vertex_count", 0) or 0),
                    ),
                ),
                np.empty(1, dtype=np.int32),
                np.empty(1, dtype=np.int32),
            )
            for key, mesh in self.meshes_by_tex.items()
        )

    @staticmethod
    def build(
        decals: Iterable[Decal], *, tile_size: float | None = None
    ) -> "DecalBatch":
        """Build decal batches grouped by texture and spatial tile.

        Grouping by tile ensures each VBO covers a small area so distance
        culling per-batch is effective. By default tile_size uses VIEWDISTANCE
        so a small view distance yields small batches.
        """
        if tile_size is None:
            # Smaller buckets let frustum culling skip shadow VBOs behind or
            # beside the camera while keeping draw-call count bounded.
            tile_size = max(256.0, float(VIEWDISTANCE) * 0.5)

        # buckets keyed by (tex, tx, tz)
        buckets: Dict[tuple, List[np.ndarray]] = {}
        for d in decals:
            # force generate vertex data without building per-decal VBOs
            get_quad_data = getattr(d, "get_quad_vertex_data", None)
            data = get_quad_data() if callable(get_quad_data) else d.get_vertex_data()
            if data.size == 0:
                continue
            tex = int(d.texture)
            # Use decal center only for bucketing; final culling bounds are
            # computed from the combined VBO vertices below.
            try:
                c = d.center
                cx, cz = float(c.x), float(c.z)
            except Exception:
                # compute centroid from vertex positions (x,z in cols 0,2)
                try:
                    verts = data[:, [0, 2]]
                    cx = float(verts[:, 0].mean())
                    cz = float(verts[:, 1].mean())
                except Exception:
                    cx = cz = 0.0

            tx = int(math.floor(cx / tile_size))
            tz = int(math.floor(cz / tile_size))
            key = (tex, tx, tz)
            buckets.setdefault(key, []).append(data)

        meshes: Dict[tuple, BatchedMesh] = {}
        centers_out: Dict[tuple, tuple[float, float, float]] = {}
        radii_out: Dict[tuple, float] = {}
        tiles_by_tex: Dict[
            int, list[tuple[tuple[float, float, float], float, np.ndarray]]
        ] = {}

        for key, chunks in buckets.items():
            vertex_data = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
            mins = vertex_data[:, 0:3].min(axis=0)
            maxs = vertex_data[:, 0:3].max(axis=0)
            center = (mins + maxs) * 0.5
            half_extents = (maxs - mins) * 0.5
            center_out = (
                float(center[0]),
                float(center[1]),
                float(center[2]),
            )
            radius_out = float(np.linalg.norm(half_extents))
            centers_out[key] = center_out
            radii_out[key] = radius_out
            tiles_by_tex.setdefault(int(key[0]), []).append(
                (center_out, radius_out, vertex_data)
            )

        draw_groups = []
        color_offset_ptr = ctypes.c_void_p(3 * 4)
        uv_offset_ptr = ctypes.c_void_p(6 * 4)
        for texture, tiles in tiles_by_tex.items():
            if not tiles:
                continue
            combined = np.ascontiguousarray(
                np.concatenate([tile[2] for tile in tiles], axis=0),
                dtype=np.float32,
            )
            if combined.size == 0:
                continue
            mesh = BatchedMesh.from_vertex_data(
                combined,
                texture=texture,
                keep_vertex_data=False,
                shine_enabled=False,
            )
            meshes[(texture,)] = mesh

            ranges = []
            start = 0
            for center, radius, vertex_data in tiles:
                count = int(vertex_data.shape[0])
                ranges.append((center, radius, start, count))
                start += count
            ranges_tuple = tuple(ranges)
            draw_groups.append(
                (
                    texture,
                    int(mesh.vbo_vertices or 0),
                    int(getattr(mesh, "_vertex_stride", 8 * 4) or 8 * 4),
                    getattr(mesh, "_color_offset_ptr", color_offset_ptr),
                    getattr(mesh, "_uv_offset_ptr", uv_offset_ptr),
                    ranges_tuple,
                    np.empty(len(ranges_tuple), dtype=np.int32),
                    np.empty(len(ranges_tuple), dtype=np.int32),
                )
            )

        db = DecalBatch(meshes_by_tex=meshes)
        db.mesh_centers = centers_out
        db.mesh_radii = radii_out
        db.draw_mode = GL_QUADS
        db.draw_groups = tuple(draw_groups)
        db._refresh_draw_items()
        db.draw_groups = tuple(draw_groups)
        return db

    def dispose(self) -> None:
        for mesh in self.meshes_by_tex.values():
            mesh.dispose()
        self.meshes_by_tex.clear()
        self.mesh_centers.clear()
        self.mesh_radii.clear()
        self.draw_items = ()
        self.draw_groups = ()

    @staticmethod
    def _cull_context(camera, cam_pos):
        try:
            forward = camera._forward
            right = camera._right
            up = camera._up
            fov_scale = max(1e-6, float(getattr(camera, "_fov_scale", HEIGHT * 0.5)))
            tan_half = (HEIGHT * 0.5) / fov_scale
            aspect = WIDTH / HEIGHT
            tan_half_h = tan_half * aspect
            return (
                float(cam_pos.x),
                float(cam_pos.y),
                float(cam_pos.z),
                float(right.x),
                float(right.y),
                float(right.z),
                float(up.x),
                float(up.y),
                float(up.z),
                float(forward.x),
                float(forward.y),
                float(forward.z),
                tan_half,
                aspect,
                math.sqrt(1.0 + tan_half_h * tan_half_h),
                math.sqrt(1.0 + tan_half * tan_half),
            )
        except Exception:
            return None

    @staticmethod
    def _mesh_is_visible(center, radius: float, cull_context) -> bool:
        if center is None:
            return True
        if cull_context is None:
            return True
        (
            cx,
            cy,
            cz,
            rx,
            ry,
            rz,
            ux,
            uy,
            uz,
            fx,
            fy,
            fz,
            tan_half,
            aspect,
            horizontal_radius_scale,
            vertical_radius_scale,
        ) = cull_context
        dx = float(center[0]) - cx
        dy = float(center[1]) - cy
        dz = float(center[2]) - cz
        radius = max(0.0, float(radius))
        depth = dx * fx + dy * fy + dz * fz
        if depth < -radius or depth > VIEWDISTANCE + radius:
            return False

        x_cam = dx * rx + dy * ry + dz * rz
        y_cam = dx * ux + dy * uy + dz * uz
        depth_for_extent = max(0.0, depth)
        half_v = depth_for_extent * tan_half
        half_h = half_v * aspect
        return (
            abs(x_cam) <= half_h + radius * horizontal_radius_scale
            and abs(y_cam) <= half_v + radius * vertical_radius_scale
        )

    @staticmethod
    def _count(profiler, name: str, amount: float = 1.0) -> None:
        if profiler is not None and getattr(profiler, "enabled", False):
            profiler.count(name, amount)

    def draw(self, camera=None, profiler=None) -> None:
        """Draw decal batches. If `camera` is provided, perform a cheap
        distance cull per-texture using precomputed centers.
        """
        # Quick culling: if camera provided and we have a center for this tex,
        # skip drawing that texture's mesh if it's farther than VIEWDISTANCE.
        groups = self.draw_groups
        if not groups:
            return

        tile_count = sum(len(group[5]) for group in groups)
        self._count(profiler, "decals.tiles", tile_count)
        self._count(profiler, "decals.groups", len(groups))
        cam_pos = getattr(camera, "position", None) if camera is not None else None
        cull_context = (
            self._cull_context(camera, cam_pos) if cam_pos is not None else None
        )
        glDepthMask_local = glDepthMask
        glEnable_local = glEnable
        glDisable_local = glDisable
        glBlendFunc_local = glBlendFunc
        glTexEnvi_local = glTexEnvi
        glEnableClientState_local = glEnableClientState
        glDisableClientState_local = glDisableClientState
        glBindTexture_local = glBindTexture
        glBindBuffer_local = glBindBuffer
        glVertexPointer_local = glVertexPointer
        glColorPointer_local = glColorPointer
        glTexCoordPointer_local = glTexCoordPointer
        glDrawArrays_local = glDrawArrays
        glMultiDrawArrays_local = glMultiDrawArrays
        draw_texture_2d = GL_TEXTURE_2D
        draw_mode = self.draw_mode

        if cull_context is not None:
            (
                cx,
                cy,
                cz,
                rx,
                ry,
                rz,
                ux,
                uy,
                uz,
                fx,
                fy,
                fz,
                tan_half,
                aspect,
                horizontal_radius_scale,
                vertical_radius_scale,
            ) = cull_context
        else:
            cx = cy = cz = 0.0
            rx = ry = rz = ux = uy = uz = fx = fy = fz = 0.0
            tan_half = aspect = horizontal_radius_scale = vertical_radius_scale = 0.0

        glDepthMask_local(False)
        glEnable_local(GL_BLEND)
        glBlendFunc_local(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable_local(draw_texture_2d)
        glTexEnvi_local(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glEnableClientState_local(GL_VERTEX_ARRAY)
        glEnableClientState_local(GL_COLOR_ARRAY)
        glEnableClientState_local(GL_TEXTURE_COORD_ARRAY)
        drawn_tiles = 0
        drawn_vertices = 0
        draw_calls = 0
        try:
            for (
                texture,
                vbo,
                stride,
                color_offset_ptr,
                uv_offset_ptr,
                ranges,
                firsts,
                counts,
            ) in groups:
                if not vbo or not ranges:
                    continue

                visible_count = 0
                visible_vertices = 0
                for center, radius, start, vertex_count in ranges:
                    if vertex_count <= 0:
                        continue
                    if cam_pos is not None:
                        if center is not None and cull_context is not None:
                            dx = float(center[0]) - cx
                            dy = float(center[1]) - cy
                            dz = float(center[2]) - cz
                            radius = max(0.0, float(radius))
                            depth = dx * fx + dy * fy + dz * fz
                            if depth < -radius or depth > VIEWDISTANCE + radius:
                                continue

                            x_cam = dx * rx + dy * ry + dz * rz
                            y_cam = dx * ux + dy * uy + dz * uz
                            half_v = max(0.0, depth) * tan_half
                            half_h = half_v * aspect
                            if (
                                abs(x_cam) > half_h + radius * horizontal_radius_scale
                                or abs(y_cam) > half_v + radius * vertical_radius_scale
                            ):
                                continue

                    firsts[visible_count] = int(start)
                    counts[visible_count] = int(vertex_count)
                    visible_count += 1
                    visible_vertices += int(vertex_count)

                if visible_count <= 0:
                    continue

                glBindTexture_local(draw_texture_2d, texture)
                glBindBuffer_local(GL_ARRAY_BUFFER, vbo)
                glVertexPointer_local(3, GL_FLOAT, stride, None)
                glColorPointer_local(3, GL_FLOAT, stride, color_offset_ptr)
                glTexCoordPointer_local(2, GL_FLOAT, stride, uv_offset_ptr)

                if self._use_multi_draw and visible_count > 1:
                    try:
                        glMultiDrawArrays_local(
                            draw_mode,
                            firsts[:visible_count],
                            counts[:visible_count],
                            visible_count,
                        )
                        draw_calls += 1
                    except Exception:
                        self._use_multi_draw = False
                        for draw_index in range(visible_count):
                            glDrawArrays_local(
                                draw_mode,
                                int(firsts[draw_index]),
                                int(counts[draw_index]),
                            )
                        draw_calls += visible_count
                else:
                    for draw_index in range(visible_count):
                        glDrawArrays_local(
                            draw_mode,
                            int(firsts[draw_index]),
                            int(counts[draw_index]),
                        )
                    draw_calls += visible_count

                drawn_tiles += visible_count
                drawn_vertices += visible_vertices
        finally:
            if drawn_tiles:
                self._count(profiler, "decals.drawn_tiles", drawn_tiles)
                self._count(profiler, "decals.vertices", drawn_vertices)
                self._count(profiler, "decals.draw_calls", draw_calls)
            glDisableClientState_local(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState_local(GL_COLOR_ARRAY)
            glDisableClientState_local(GL_VERTEX_ARRAY)
            glDisable_local(draw_texture_2d)
            glDisable_local(GL_BLEND)
            glDepthMask_local(True)
