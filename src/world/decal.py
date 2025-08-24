"""Terrain-conforming textured decal.

Builds a small tessellated quad in world space and samples a provided
terrain height function so the decal hugs the surface (e.g., ground).

Uses the fixed-function BatchedMesh pipeline for rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import math

import numpy as np
from pygame.math import Vector3
from OpenGL.GL import (
    glGenBuffers,
    glBindBuffer,
    glBufferData,
    GL_ARRAY_BUFFER,
    GL_STATIC_DRAW,
    glEnable,
    glDisable,
    glPolygonOffset,
    GL_POLYGON_OFFSET_FILL,
    glDepthMask,
    glBlendFunc,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
)

from core.mesh import BatchedMesh


HeightFn = Callable[[float, float], float]


@dataclass
class Decal:
    """A textured mesh that conforms to terrain.

    Parameters
    ----------
    center: Vector3
        World-space center of the decal (Y is ignored; terrain height is sampled).
    size: tuple[float, float]
        (width, height) in world units.
    texture: int
        OpenGL texture ID.
    rotation_deg: float
        Rotation around +Y in degrees (0 = aligned with +X width, +Z height).
    subdiv_u, subdiv_v: int
        Grid resolution for tessellation; higher = smoother conformance.
    height_fn: callable(x, z) -> y
        Terrain height sampler. If not provided, height_sampler.height_at is used.
    height_sampler: object
        Optional sampler with method height_at(x, z) -> y.
    elevation: float
        Small lift above surface to avoid z-fighting.
    uv_repeat: tuple[float, float]
        How many times to repeat the texture across (u, v).
    color: tuple[float, float, float]
        Per-vertex color multiplier (RGB), alpha comes from the texture.
    """

    center: Vector3
    size: Tuple[float, float]
    texture: int
    rotation_deg: float = 0.0
    subdiv_u: int = 8
    subdiv_v: int = 8
    height_fn: Optional[HeightFn] = None
    height_sampler: Optional[object] = None
    elevation: float = 0.05
    uv_repeat: Tuple[float, float] = (1.0, 1.0)
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Depth bias to reduce z-fighting with the surface
    use_depth_offset: bool = True
    depth_offset_factor: float = -1.0
    depth_offset_units: float = -1.0

    # Internal
    _mesh: Optional[BatchedMesh] = None
    _vertex_data: Optional[np.ndarray] = None
    build_vbo: bool = True

    def __post_init__(self) -> None:
        # Prefer explicit height_fn, else sampler.height_at
        if (
            self.height_fn is None
            and self.height_sampler is not None
            and hasattr(self.height_sampler, "height_at")
        ):
            self.height_fn = lambda x, z: float(self.height_sampler.height_at(x, z))
        if self.build_vbo:
            self._rebuild()
        else:
            # Generate and keep vertex data only
            self._vertex_data = self._generate_vertex_data()

    # --- Public API -----------------------------------------------------
    def set_center(self, center: Vector3 | Tuple[float, float, float]) -> None:
        self.center = center if isinstance(center, Vector3) else Vector3(*center)
        if self.build_vbo:
            self._rebuild()
        else:
            self._vertex_data = self._generate_vertex_data()

    def set_rotation(self, degrees: float) -> None:
        self.rotation_deg = float(degrees)
        if self.build_vbo:
            self._rebuild()
        else:
            self._vertex_data = self._generate_vertex_data()

    def set_size(self, width: float, height: float) -> None:
        self.size = (float(width), float(height))
        if self.build_vbo:
            self._rebuild()
        else:
            self._vertex_data = self._generate_vertex_data()

    def set_uv_repeat(self, u_rep: float, v_rep: float) -> None:
        self.uv_repeat = (float(u_rep), float(v_rep))
        self._rebuild()

    def draw_untextured(self) -> None:  # parity with Drawable
        self.draw()

    def draw(self) -> None:
        if self._mesh is None:
            return
        if self.use_depth_offset:
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(self.depth_offset_factor, self.depth_offset_units)
        # Enable blending and disable depth writes so overlapping decals blend nicely
        glDepthMask(False)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        try:
            self._mesh.draw()
        finally:
            glDisable(GL_BLEND)
            glDepthMask(True)
            if self.use_depth_offset:
                glDisable(GL_POLYGON_OFFSET_FILL)

    # --- Internal mesh builder -----------------------------------------
    def _rebuild(self) -> None:
        vertex_data = self._generate_vertex_data()
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        self._mesh = BatchedMesh(
            vbo_vertices=vbo, vertex_count=vertex_data.shape[0], texture=self.texture
        )

    def _generate_vertex_data(self) -> np.ndarray:
        w, h = self.size
        if w <= 1e-6 or h <= 1e-6 or self.subdiv_u <= 0 or self.subdiv_v <= 0:
            # Build empty mesh
            empty = np.zeros((0, 8), dtype=np.float32)
            return empty

        # Axes in world space for local (u,v) directions
        theta = math.radians(self.rotation_deg)
        c, s = math.cos(theta), math.sin(theta)
        axis_u = Vector3(c, 0.0, -s)  # local +u (width) rotated around +Y
        axis_v = Vector3(s, 0.0, c)  # local +v (height) rotated around +Y

        # Precompute grid of vertices (positions with height sampling) and UVs
        # Grid has (subdiv_u+1) x (subdiv_v+1) vertices
        nu = int(self.subdiv_u)
        nv = int(self.subdiv_v)
        cx, cz = float(self.center.x), float(self.center.z)
        half_w = w * 0.5
        half_h = h * 0.5
        u_rep, v_rep = self.uv_repeat
        r, g, b = self.color

        # Helper to sample height; fallback flat at y=5.0 if none provided
        def sample_y(x: float, z: float) -> float:
            if self.height_fn is not None:
                try:
                    return float(self.height_fn(x, z)) + self.elevation
                except Exception:
                    pass
            return 5.0 + self.elevation

        # Build grid of positions and UVs
        verts_grid = [[None for _ in range(nu + 1)] for _ in range(nv + 1)]
        for iv in range(nv + 1):
            v = iv / nv
            # Offset along axis_v from center
            off_v = (v - 0.5) * h
            dvx = axis_v.x * off_v
            dvz = axis_v.z * off_v
            for iu in range(nu + 1):
                u = iu / nu
                off_u = (u - 0.5) * w
                dux = axis_u.x * off_u
                duz = axis_u.z * off_u

                x = cx + dux + dvx
                z = cz + duz + dvz
                y = sample_y(x, z)
                verts_grid[iv][iu] = (x, y, z, r, g, b, u * u_rep, v * v_rep)

        # Triangulate cells into a flat vertex array (two tris per cell)
        tris: list[tuple[float, ...]] = []
        for iv in range(nv):
            for iu in range(nu):
                v00 = verts_grid[iv][iu]
                v10 = verts_grid[iv][iu + 1]
                v01 = verts_grid[iv + 1][iu]
                v11 = verts_grid[iv + 1][iu + 1]
                # Keep diagonal consistent to avoid cracks with terrain triangulation
                # Tri 1: (v00, v10, v11)
                tris.append(v00)
                tris.append(v10)
                tris.append(v11)
                # Tri 2: (v00, v11, v01)
                tris.append(v00)
                tris.append(v11)
                tris.append(v01)

        vertex_data = np.array(tris, dtype=np.float32)
        return vertex_data

    # Export built vertex data (recomputed if necessary when not building VBO)
    def get_vertex_data(self) -> np.ndarray:
        if self.build_vbo:
            # If VBO path, regenerate to provide data
            return self._generate_vertex_data()
        if self._vertex_data is None:
            self._vertex_data = self._generate_vertex_data()
        return self._vertex_data
