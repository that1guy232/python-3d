"""Decal batching: merge many terrain-conforming decals into a single VBO per texture.

This builds one BatchedMesh per texture from many per-sprite decals by
concatenating their vertex data. Use for static decals that don't move each frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Iterable
import math
import numpy as np
from OpenGL.GL import (
    glBindTexture,
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
    GL_VERTEX_ARRAY,
    GL_COLOR_ARRAY,
    GL_TEXTURE_COORD_ARRAY,
)

from core.mesh import BatchedMesh
from engine.rendering.decal import Decal
from config import HEIGHT, VIEWDISTANCE, WIDTH


@dataclass
class DecalBatch:
    meshes_by_tex: Dict[tuple, BatchedMesh] = field(default_factory=dict)
    # Precomputed center per texture (cx, cy, cz) used for cheap distance culling
    mesh_centers: Dict[tuple, tuple[float, float, float]] = field(default_factory=dict)
    mesh_radii: Dict[tuple, float] = field(default_factory=dict)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @staticmethod
    def build(decals: Iterable[Decal], *, tile_size: float | None = None) -> "DecalBatch":
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
            data = d.get_vertex_data()
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

        for key, chunks in buckets.items():
            vertex_data = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
            meshes[key] = BatchedMesh.from_vertex_data(
                vertex_data,
                texture=key[0],
                keep_vertex_data=False,
                shine_enabled=False,
            )
            mins = vertex_data[:, 0:3].min(axis=0)
            maxs = vertex_data[:, 0:3].max(axis=0)
            center = (mins + maxs) * 0.5
            half_extents = (maxs - mins) * 0.5
            centers_out[key] = (
                float(center[0]),
                float(center[1]),
                float(center[2]),
            )
            radii_out[key] = float(np.linalg.norm(half_extents))

        db = DecalBatch(meshes_by_tex=meshes)
        db.mesh_centers = centers_out
        db.mesh_radii = radii_out
        return db

    def dispose(self) -> None:
        for mesh in self.meshes_by_tex.values():
            mesh.dispose()
        self.meshes_by_tex.clear()
        self.mesh_centers.clear()
        self.mesh_radii.clear()

    def _mesh_is_visible(self, key, camera, cam_pos, vd_sq: float) -> bool:
        center = self.mesh_centers.get(key)
        if center is None:
            return True

        radius = self.mesh_radii.get(key, 0.0)
        dx = center[0] - cam_pos.x
        dy = center[1] - cam_pos.y
        dz = center[2] - cam_pos.z
        max_dist = VIEWDISTANCE + radius
        if (dx * dx + dy * dy + dz * dz) > max(vd_sq, max_dist * max_dist):
            return False

        forward = getattr(camera, "_forward", None)
        if forward is None:
            return True

        depth = dx * forward.x + dy * forward.y + dz * forward.z
        if depth < -radius or depth > VIEWDISTANCE + radius:
            return False

        matrix = getattr(camera, "_R", None)
        try:
            x_cam = dx * float(matrix[0][0]) + dy * float(matrix[0][1]) + dz * float(matrix[0][2])
            y_cam = dx * float(matrix[1][0]) + dy * float(matrix[1][1]) + dz * float(matrix[1][2])
        except Exception:
            right = getattr(camera, "_right", None)
            if right is None:
                return True
            x_cam = dx * right.x + dy * right.y + dz * right.z
            y_cam = dy

        fov_scale = getattr(camera, "_fov_scale", HEIGHT * 0.5)
        tan_half_fov = (HEIGHT * 0.5) / max(1e-6, float(fov_scale))
        depth_for_extent = max(1.0, depth)
        half_v = depth_for_extent * tan_half_fov
        half_h = half_v * (WIDTH / HEIGHT)
        return abs(x_cam) <= half_h + radius and abs(y_cam) <= half_v + radius

    def draw(self, camera=None) -> None:
        """Draw decal batches. If `camera` is provided, perform a cheap
        distance cull per-texture using precomputed centers.
        """
        # Quick culling: if camera provided and we have a center for this tex,
        # skip drawing that texture's mesh if it's farther than VIEWDISTANCE.
        cam_pos = getattr(camera, "position", None) if camera is not None else None
        vd_sq = VIEWDISTANCE * VIEWDISTANCE
        glDepthMask(False)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        bound_texture = None
        try:
            for key, mesh in self.meshes_by_tex.items():
                if cam_pos is not None and not self._mesh_is_visible(
                    key, camera, cam_pos, vd_sq
                ):
                    continue
                if mesh.texture != bound_texture:
                    glBindTexture(GL_TEXTURE_2D, mesh.texture)
                    bound_texture = mesh.texture
                mesh.draw_textured_prepared(bind_texture=False)
        finally:
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_BLEND)
            glDepthMask(True)
