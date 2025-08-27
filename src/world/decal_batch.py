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
    glGenBuffers,
    glBindBuffer,
    glBufferData,
    glEnable,
    glDisable,
    glDepthMask,
    glBlendFunc,
    GL_ARRAY_BUFFER,
    GL_STATIC_DRAW,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
)

from core.mesh import BatchedMesh
from world.decal import Decal
from config import VIEWDISTANCE


@dataclass
class DecalBatch:
    meshes_by_tex: Dict[int, BatchedMesh] = field(default_factory=dict)
    # Precomputed center per texture (cx, cy, cz) used for cheap distance culling
    mesh_centers: Dict[int, tuple[float, float, float]] = field(default_factory=dict)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @staticmethod
    def build(decals: Iterable[Decal], *, tile_size: float | None = None) -> "DecalBatch":
        """Build decal batches grouped by texture and spatial tile.

        Grouping by tile ensures each VBO covers a small area so distance
        culling per-batch is effective. By default tile_size uses VIEWDISTANCE
        so a small view distance yields small batches.
        """
        if tile_size is None:
            # default tile size: use view distance as a heuristic (fallback 256)
            tile_size = float(VIEWDISTANCE) if VIEWDISTANCE > 0 else 256.0

        # buckets keyed by (tex, tx, tz)
        buckets: Dict[tuple, List[np.ndarray]] = {}
        centers: Dict[tuple, List[tuple[float, float, float]]] = {}

        for d in decals:
            # force generate vertex data without building per-decal VBOs
            data = d.get_vertex_data()
            if data.size == 0:
                continue
            tex = int(d.texture)
            # try to get decal center; fall back to vertex centroid if missing
            try:
                c = d.center
                cx, cy, cz = float(c.x), float(c.y), float(c.z)
            except Exception:
                # compute centroid from vertex positions (x,z in cols 0,2)
                try:
                    verts = data[:, [0, 2]]
                    cx = float(verts[:, 0].mean())
                    cz = float(verts[:, 1].mean())
                    cy = 0.0
                except Exception:
                    cx = cz = cy = 0.0

            tx = int(math.floor(cx / tile_size))
            tz = int(math.floor(cz / tile_size))
            key = (tex, tx, tz)
            buckets.setdefault(key, []).append(data)
            centers.setdefault(key, []).append((cx, cy, cz))

        meshes: Dict[tuple, BatchedMesh] = {}
        centers_out: Dict[tuple, tuple[float, float, float]] = {}

        for key, chunks in buckets.items():
            vertex_data = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW
            )
            meshes[key] = BatchedMesh(
                vbo_vertices=vbo, vertex_count=vertex_data.shape[0], texture=key[0]
            )
            pts = centers.get(key)
            if pts:
                sx = sum(p[0] for p in pts) / len(pts)
                sy = sum(p[1] for p in pts) / len(pts)
                sz = sum(p[2] for p in pts) / len(pts)
                centers_out[key] = (sx, sy, sz)

        db = DecalBatch(meshes_by_tex=meshes)
        db.mesh_centers = centers_out
        return db

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
        try:
            for key, mesh in self.meshes_by_tex.items():
                if cam_pos is not None:
                    center = self.mesh_centers.get(key)
                    if center is not None:
                        dx = center[0] - cam_pos.x
                        dy = center[1] - cam_pos.y
                        dz = center[2] - cam_pos.z
                        dist_sq = (dx * dx + dy * dy + dz * dz)
                        if dist_sq > vd_sq:
                            continue
                mesh.draw()
        finally:
            glDisable(GL_BLEND)
            glDepthMask(True)
