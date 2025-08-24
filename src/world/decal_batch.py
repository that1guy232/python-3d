"""Decal batching: merge many terrain-conforming decals into a single VBO per texture.

This builds one BatchedMesh per texture from many per-sprite decals by
concatenating their vertex data. Use for static decals that don't move each frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Iterable
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


@dataclass
class DecalBatch:
    meshes_by_tex: Dict[int, BatchedMesh] = field(default_factory=dict)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @staticmethod
    def build(decals: Iterable[Decal]) -> "DecalBatch":
        buckets: Dict[int, List[np.ndarray]] = {}
        for d in decals:
            # force generate vertex data without building per-decal VBOs
            data = d.get_vertex_data()
            if data.size == 0:
                continue
            tex = int(d.texture)
            buckets.setdefault(tex, []).append(data)

        meshes: Dict[int, BatchedMesh] = {}
        for tex, chunks in buckets.items():
            vertex_data = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(
                GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW
            )
            meshes[tex] = BatchedMesh(
                vbo_vertices=vbo, vertex_count=vertex_data.shape[0], texture=tex
            )

        return DecalBatch(meshes_by_tex=meshes)

    def draw(self) -> None:
        # Render with alpha blending and disable depth writes so decals overlap well
        glDepthMask(False)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        try:
            # If we wanted color tinting, we'd need shader or per-vertex color; meshes already carry color
            for mesh in self.meshes_by_tex.values():
                mesh.draw()
        finally:
            glDisable(GL_BLEND)
            glDepthMask(True)
