"""Renderer utilities for batching static geometry into GPU buffers.

Currently focuses on generating a ground tile grid as a single VBO draw call
instead of many immediate-mode glBegin/glEnd calls. This yields large speedups
when the grid size increases beyond a trivial number of tiles.

Uses the fixed-function pipeline (legacy) for simplicity so it plugs directly
into the existing code base without introducing shaders. Geometry is uploaded
once (STATIC_DRAW) and rendered via glDrawArrays(GL_TRIANGLES, ...).
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Callable
from pygame.math import Vector3
from world.ground_tile import GroundTile
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
    glBindTexture,  # <- Add this
    glEnable,  # <- Add this
    glDisable,  # <- Add this
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
import ctypes
import numpy as np
from textures.texture_utils import get_texture_size
import math
import pygame
import os
from textures.resoucepath import TEXTURES_PATH


@dataclass
class BatchedMesh:
    vbo_vertices: int
    vertex_count: int
    texture: int | None = None  # texture ID or None for untextured
    # Optional height sampler for meshes that represent a terrain-like surface
    height_sampler: Optional[object] = None

    def draw_untextured(self):  # fixed-function pipeline draw helper
        if self.vertex_count == 0:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        stride = 6 * 4  # 6 floats (xyz + rgb) * 4 bytes
        # position (first 3 floats)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, stride, None)
        # color (next 3 floats) — offset = 12 bytes
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(12))
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def draw(self):  # texture-enabled draw helper
        if self.vertex_count == 0:
            return

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)

        if self.texture is not None:
            # Textured rendering (8 floats per vertex: xyz + rgb + uv)
            stride = 8 * 4  # 8 floats * 4 bytes

            # Position (first 3 floats)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, None)

            # Color (next 3 floats) - offset = 12 bytes
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(12))

            # Texture coordinates (last 2 floats) - offset = 24 bytes
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(24))

            # Enable texture
            glEnable(GL_TEXTURE_2D)
            # Enable alpha blending so textures with transparency (sprites, fences) render correctly
            glEnable(GL_BLEND)
            from OpenGL.GL import (
                glBlendFunc,
            )  # local import to avoid polluting namespace above

            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBindTexture(GL_TEXTURE_2D, self.texture)

            # Draw
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

            # Clean up
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        else:
            # Fallback to untextured rendering (6 floats per vertex: xyz + rgb)
            self.draw_untextured()


def build_textured_ground_grid(
    count: int = 10, tile_size: float = 100, gap: float = 1, texture=None
) -> BatchedMesh:
    """Create a ground grid where only the top face is textured."""
    w = tile_size / 2.0
    h = 5  # thin
    d = tile_size / 2.0
    # Derive geometry from GroundTile to keep a single source of truth
    tile = GroundTile(position=Vector3(0, 0, 0), width=w, height=h, depth=d)
    base = tile.local_vertices
    faces = tile.faces
    # Only the top face gets real UVs (for legacy cuboid). For plane tiles (our current GroundTile),
    # we assign uv_top to the sole face so it tiles per tile.
    uv_top = [(0, 0), (1, 0), (1, 1), (0, 1)]
    uv_dummy = [(0, 0)] * 4
    if len(faces) == 1:
        # Plane: single quad -> use proper UVs
        face_uvs = [uv_top]
    else:
        # Cuboid fallback: only the top face (index 3) gets proper UVs
        face_uvs = [
            uv_dummy,  # front
            uv_dummy,  # back
            uv_dummy,  # bottom
            uv_top,  # top
            uv_dummy,  # right
            uv_dummy,  # left
        ]
    spacing = tile_size + gap
    vertices = []
    tile_vertices = []
    for face_idx, (face, color) in enumerate(zip(faces, tile.face_colors)):
        a, b, c, d_idx = face
        face_uv = face_uvs[face_idx]
        # Triangulate quad → (a,b,c) and (a,c,d)
        tri_indices = [a, b, c, a, c, d_idx]
        tri_uvs = [
            face_uv[0],
            face_uv[1],
            face_uv[2],
            face_uv[0],
            face_uv[2],
            face_uv[3],
        ]
        for idx, uv in zip(tri_indices, tri_uvs):
            v = base[idx]
            tile_vertices.append((v.x, v.y, v.z, *color, uv[0], uv[1]))
    tile_vertex_array = np.array(tile_vertices, dtype=np.float32)

    # We'll also precompute a simple height sampler consistent with the generated vertices.
    # Use an external heightmap PNG (assets/textures/heightmap.png) to define vertex heights.
    # If the heightmap is missing or fails to load, fall back to a flat surface at `h`.
    heightmap_path = os.path.join(TEXTURES_PATH, "heightmap.png")
    heightmap_arr = None
    heightmap_amp = 80.0  # world-unit amplitude mapping for full [0..1] image range
    try:
        if not pygame.get_init():
            pygame.init()
        surf = pygame.image.load(heightmap_path)
        # Ensure a consistent RGB surface
        surf = surf.convert()
        import pygame.surfarray as surfarray

        # array3d -> shape (w, h, 3)
        hm = surfarray.array3d(surf).astype(np.float32) / 255.0
        heightmap_arr = hm
        hm_w, hm_h = hm.shape[0], hm.shape[1]
        # Compute world bounds that the heightmap covers (match worldscene expectations)
        world_min_x = -w
        world_max_x = (count - 1) * spacing + w
        world_min_z = -d
        world_max_z = (count - 1) * spacing + d
    except Exception as e:
        print(
            f"Warning: failed to load heightmap '{heightmap_path}': {e}; using flat ground"
        )
        heightmap_arr = None

    def sample_height_from_map(wx: float, wz: float) -> float:
        # Return top-surface Y for world (wx,wz) using the heightmap if available
        if heightmap_arr is None:
            return h
        # Map world coords into [0,1] across the heightmap bounds
        ux = (wx - world_min_x) / max(1e-12, (world_max_x - world_min_x))
        uz = (wz - world_min_z) / max(1e-12, (world_max_z - world_min_z))
        ux = min(1.0, max(0.0, ux))
        uz = min(1.0, max(0.0, uz))
        px = int(ux * (hm_w - 1))
        py = int(uz * (hm_h - 1))
        # array3d is (x,y,3) where x=0..w-1, y=0..h-1
        rgb = heightmap_arr[px, py]
        lum = float(rgb[0] + rgb[1] + rgb[2]) / 3.0
        # Map [0..1] image to [-amp..+amp] then add base height h so mid-gray => base height
        return h + (lum - 0.5) * 2.0 * heightmap_amp

    # Note: we no longer apply the procedural wave below; heights come from the image
    corner_heights = np.zeros((count, count, 4), dtype=np.float32)  # a,b,c,d per tile
    for gx in range(count):
        for gz in range(count):
            tx = gx * spacing
            tz = gz * spacing
            translated = tile_vertex_array.copy()
            translated[:, 0] += tx
            translated[:, 2] += tz

            # Sample heightmap per-vertex to set Y
            for i in range(len(translated)):
                x, y, z = translated[i, 0], translated[i, 1], translated[i, 2]
                sampled_y = sample_height_from_map(x, z)
                translated[i, 1] = sampled_y

            vertices.append(translated)

            # Compute the four unique corner heights for this tile (matching the quad order)
            # Corners in local space: a(-w,-d), b(w,-d), c(w,d), d(-w,d)
            # Compute corner heights from the heightmap (matching vertex sampling)
            a_y = sample_height_from_map(tx - w, tz - d)
            b_y = sample_height_from_map(tx + w, tz - d)
            c_y = sample_height_from_map(tx + w, tz + d)
            d_y = sample_height_from_map(tx - w, tz + d)
            corner_heights[gx, gz, 0] = a_y
            corner_heights[gx, gz, 1] = b_y
            corner_heights[gx, gz, 2] = c_y
            corner_heights[gx, gz, 3] = d_y

    if not vertices:
        empty = np.zeros((0, 8), dtype=np.float32)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, empty.nbytes, empty, GL_STATIC_DRAW)
        return BatchedMesh(vbo_vertices=vbo, vertex_count=0, texture=texture)
    vertex_data = np.vstack(vertices)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
    mesh = BatchedMesh(
        vbo_vertices=vbo, vertex_count=vertex_data.shape[0], texture=texture
    )

    # Attach a height sampler that matches the triangulation used when building each tile
    class GroundHeightSampler:
        __slots__ = ("_count", "_spacing", "_w", "_heights")

        def __init__(
            self, count: int, spacing: float, half: float, heights: np.ndarray
        ):
            self._count = count
            self._spacing = spacing
            self._w = half  # half-size in both x and z
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
            # Compute barycentric weights in 2D (x,z) and interpolate Y
            v0x, v0z = x1 - x0, z1 - z0
            v1x, v1z = x2 - x0, z2 - z0
            v2x, v2z = px - x0, pz - z0
            denom = v0x * v1z - v1x * v0z
            if abs(denom) < 1e-8:
                # Degenerate triangle; fall back to average
                return (y0 + y1 + y2) / 3.0
            inv = 1.0 / denom
            u = (v2x * v1z - v1x * v2z) * inv
            v = (v0x * v2z - v2x * v0z) * inv
            w = 1.0 - u - v
            return u * y1 + v * y2 + w * y0

        def height_at(self, x: float, z: float) -> float:
            # Map world (x,z) to tile indices (gx,gz)
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

            # Corner local coords relative to tile center
            x_a, z_a = -half, -half
            x_b, z_b = +half, -half
            x_c, z_c = +half, +half
            x_d, z_d = -half, +half
            a_y, b_y, c_y, d_y = self._heights[gx, gz]

            # Convert triangle vertices to WORLD coordinates (exact positions used when building the VBO)
            ax, az = tx + x_a, tz + z_a
            bx, bz = tx + x_b, tz + z_b
            cx, cz = tx + x_c, tz + z_c
            dx, dz = tx + x_d, tz + z_d

            # Helper: compute barycentric weights for triangle (p relative to tri)
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

            # Test triangle (a,b,c)
            w_abc = barycentric_weights(x, z, ax, az, bx, bz, cx, cz)
            eps = -1e-6
            if w_abc is not None:
                u, v, w = w_abc
                if u >= eps and v >= eps and w >= eps:
                    # Interpolate using the barycentric weights against the triangle's Y values
                    return u * float(b_y) + v * float(c_y) + w * float(a_y)

            # Otherwise, test triangle (a,c,d)
            w_acd = barycentric_weights(x, z, ax, az, cx, cz, dx, dz)
            if w_acd is not None:
                u, v, w = w_acd
                if u >= eps and v >= eps and w >= eps:
                    return u * float(c_y) + v * float(d_y) + w * float(a_y)

            # Fallback: degenerate or on-edge case — return bilinear interpolation of the quad
            # Map local coordinates into [0..1] across the quad
            u_lin = (lx + half) / (2.0 * half)
            v_lin = (lz + half) / (2.0 * half)
            # corners: a(lower-left), b(lower-right), c(upper-right), d(upper-left)
            # bilinear: (1-u)(1-v)*a + u(1-v)*b + u v * c + (1-u) v * d
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

    mesh.height_sampler = GroundHeightSampler(
        count=count, spacing=spacing, half=w, heights=corner_heights
    )
    return mesh


def build_textured_fence_ring(
    *,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    # If provided, `height_sampler(x,z)` will be called per-vertex to get ground height.
    # If None, the constant `ground_y` is used as a fallback.
    ground_y: float = 5.0,
    height_sampler=None,
    textures: list[int] | None = None,
    px_to_world: float = 1.0,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    # Wave controls (static shape). Set amp > 0 to enable.
    wave_amp: float = 0.0,  # amplitude in world units
    wave_freq: float = 0.02,  # cycles per world unit along a segment
    wave_phase: float = 0.0,  # radians
    slices_per_segment: int | None = None,  # auto if None
) -> list[BatchedMesh]:
    """Build a fence ring using multiple textures, alternating per segment.

    Returns a list of BatchedMesh, one per texture, so the fixed-function pipeline can
    bind each texture and draw its segments in a single call per texture.

    Notes
    -----
    - Segment width/height are derived from the first texture. For best visuals, use
      textures with matching dimensions. If sizes differ slightly, the first texture's
      size defines the fence height and nominal segment width.
        - To make the fence "wavey", set ``wave_amp`` > 0. The top edge will follow a
            sine wave along each segment. You can control the frequency (cycles per world
            unit) and an optional phase offset.
    """
    if not textures:
        return []

    # Use the first texture to establish nominal segment size
    first_tex = textures[0]
    size = get_texture_size(first_tex)
    if size:
        w_px, h_px = size
        seg_nominal_w = float(w_px) * px_to_world
        seg_h = float(h_px) * px_to_world
    else:
        seg_nominal_w = 200.0 * px_to_world
        seg_h = 200.0 * px_to_world

    r, g, b = color

    # Normalize provided height_sampler into a callable `sampler_fn(x,z)`.
    # Acceptable inputs: None, a plain callable, or an object with `height_at(x,z)`.
    if height_sampler is None:
        sampler_fn = None
    elif callable(height_sampler):
        sampler_fn = height_sampler
    elif hasattr(height_sampler, "height_at"):
        sampler_fn = lambda x, z, _hs=height_sampler: _hs.height_at(x, z)
    else:
        sampler_fn = None

    # Collect segment endpoints for all 4 edges in order
    segments: list[tuple[float, float, float, float]] = []

    def add_edge_constant_x_segments(x: float, z_start: float, z_end: float):
        length = z_end - z_start
        dir_sign = 1.0 if length >= 0 else -1.0
        L = abs(length)
        if L <= 1e-6:
            return
        n = max(1, int(np.ceil(L / seg_nominal_w)))
        step = L / n
        for i in range(n):
            z0 = z_start + dir_sign * (i * step)
            z1 = z_start + dir_sign * ((i + 1) * step)
            segments.append((x, z0, x, z1))

    def add_edge_constant_z_segments(z: float, x_start: float, x_end: float):
        length = x_end - x_start
        dir_sign = 1.0 if length >= 0 else -1.0
        L = abs(length)
        if L <= 1e-6:
            return
        n = max(1, int(np.ceil(L / seg_nominal_w)))
        step = L / n
        for i in range(n):
            x0 = x_start + dir_sign * (i * step)
            x1 = x_start + dir_sign * ((i + 1) * step)
            segments.append((x0, z, x1, z))

    # West (x=min_x), South to North
    add_edge_constant_x_segments(min_x, min_z, max_z)
    # East (x=max_x), North to South
    add_edge_constant_x_segments(max_x, max_z, min_z)
    # North (z=min_z), West to East
    add_edge_constant_z_segments(min_z, min_x, max_x)
    # South (z=max_z), East to West
    add_edge_constant_z_segments(max_z, max_x, min_x)

    if not segments:
        return []

    # Group vertices per texture
    verts_by_tex: dict[
        int, list[tuple[float, float, float, float, float, float, float, float]]
    ] = {t: [] for t in textures}

    def add_panel_flat(verts_list: list, x0: float, z0: float, x1: float, z1: float):
        # Sample ground height at each end of the panel so the fence follows terrain
        y0_left = (
            float(sampler_fn(x0, z0)) if sampler_fn is not None else float(ground_y)
        )
        y0_right = (
            float(sampler_fn(x1, z1)) if sampler_fn is not None else float(ground_y)
        )
        y_top0 = y0_left + seg_h
        y_top1 = y0_right + seg_h

        # Top-left, Top-right, Bottom-right
        verts_list.append((x0, y_top0, z0, r, g, b, 0.0, 1.0))
        verts_list.append((x1, y_top1, z1, r, g, b, 1.0, 1.0))
        verts_list.append((x1, y0_right, z1, r, g, b, 1.0, 0.0))
        # Top-left, Bottom-right, Bottom-left
        verts_list.append((x0, y_top0, z0, r, g, b, 0.0, 1.0))
        verts_list.append((x1, y0_right, z1, r, g, b, 1.0, 0.0))
        verts_list.append((x0, y0_left, z0, r, g, b, 0.0, 0.0))

    def add_panel_wavey(verts_list: list, x0: float, z0: float, x1: float, z1: float):
        # Subdivide the segment into slices and apply vertical sine to the top edge.
        dx = x1 - x0
        dz = z1 - z0
        L = float(np.hypot(dx, dz))
        if L <= 1e-6:
            return
        if slices_per_segment is not None and slices_per_segment > 0:
            n = slices_per_segment
        else:
            # Aim for ~1 slice per (seg_nominal_w/4), min 2 for visible curvature when amp>0
            target_slice_len = max(seg_nominal_w / 4.0, 1.0)
            n = max(2, int(np.ceil(L / target_slice_len)))

        # Per-end ground heights so the fence follows terrain
        # base_top will be computed per-slice endpoint as ground + seg_h

        for i in range(n):
            t0 = i / n
            t1 = (i + 1) / n
            # Positions along the segment
            sx0 = x0 + dx * t0
            sz0 = z0 + dz * t0
            sx1 = x0 + dx * t1
            sz1 = z0 + dz * t1

            # Distance along segment for wave argument (world units)
            s0 = L * t0
            s1 = L * t1
            # Sine offsets (top edge). Sample bottoms per endpoint.
            off0 = wave_amp * float(np.sin(2.0 * np.pi * wave_freq * s0 + wave_phase))
            off1 = wave_amp * float(np.sin(2.0 * np.pi * wave_freq * s1 + wave_phase))

            y_bottom0 = (
                float(sampler_fn(sx0, sz0))
                if sampler_fn is not None
                else float(ground_y)
            )
            y_bottom1 = (
                float(sampler_fn(sx1, sz1))
                if sampler_fn is not None
                else float(ground_y)
            )
            y_top0 = y_bottom0 + seg_h + off0
            y_top1 = y_bottom1 + seg_h + off1

            # Texture U coordinates follow along the segment; V: bottom=0, top=1
            u0 = t0
            u1 = t1

            # Two triangles per slice using per-end top/bottom heights
            # Top-left, Top-right, Bottom-right
            verts_list.append((sx0, y_top0, sz0, r, g, b, u0, 1.0))
            verts_list.append((sx1, y_top1, sz1, r, g, b, u1, 1.0))
            verts_list.append((sx1, y_bottom1, sz1, r, g, b, u1, 0.0))
            # Top-left, Bottom-right, Bottom-left
            verts_list.append((sx0, y_top0, sz0, r, g, b, u0, 1.0))
            verts_list.append((sx1, y_bottom1, sz1, r, g, b, u1, 0.0))
            verts_list.append((sx0, y_bottom0, sz0, r, g, b, u0, 0.0))

    # Assign textures randomly per segment
    for x0, z0, x1, z1 in segments:
        tex = random.choice(textures)
        if wave_amp > 0.0:
            add_panel_wavey(verts_by_tex[tex], x0, z0, x1, z1)
        else:
            add_panel_flat(verts_by_tex[tex], x0, z0, x1, z1)

    # Create a mesh per texture with its vertex group
    meshes: list[BatchedMesh] = []
    for tex, verts in verts_by_tex.items():
        if not verts:
            continue
        vertex_data = np.array(verts, dtype=np.float32)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        meshes.append(
            BatchedMesh(
                vbo_vertices=vbo, vertex_count=vertex_data.shape[0], texture=tex
            )
        )

    return meshes
