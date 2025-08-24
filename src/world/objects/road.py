"""KISS Road: a simple textured strip between two points.

This stripped-down version draws a single straight road segment from ``start``
to ``end`` at a fixed ``ground_y`` with a total ``width``. The road texture is
repeated along its length (U) and across its width (V).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, Callable
from pygame.math import Vector3
import numpy as np
from OpenGL.GL import (
    glGenBuffers,
    glBindBuffer,
    glBufferData,
    glBindTexture,
    glTexParameteri,
    GL_ARRAY_BUFFER,
    GL_STATIC_DRAW,
    GL_TEXTURE_2D,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_REPEAT,
)

from core.renderer import BatchedMesh
from textures.texture_utils import (
    get_texture_size,
)  # kept for potential future sizing (unused)


def _perp_right(dx: float, dz: float) -> Tuple[float, float]:
    """Right-hand perpendicular in XZ plane, normalized.

    Falls back to (1, 0) if the input direction is near-zero.
    """
    L = (dx * dx + dz * dz) ** 0.5
    if L <= 1e-6:
        return 1.0, 0.0
    ux, uz = dx / L, dz / L
    return uz, -ux


def _ensure_texture_repeat(tex_id: int) -> None:
    """Set wrap mode to REPEAT for the given texture (both S and T)."""
    if not tex_id:
        return
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)


@dataclass
class Road:
    """Simple straight road strip with repeating texture."""

    start: Vector3
    end: Vector3
    ground_y: float
    width: float
    texture: int
    v_tiles: float = 1.0
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Optional terrain conformance
    height_fn: Optional[Callable[[float, float], float]] = None
    elevation: float = 0.02  # lift above terrain to avoid z-fighting
    segment_length: float = 20.0  # segment length for tessellation when conforming

    # Internal
    _mesh: BatchedMesh | None = None

    def __init__(
        self,
        *,
        start: Vector3 | Tuple[float, float] | None = None,
        end: Vector3 | Tuple[float, float] | None = None,
        ground_y: float,
        width: float,
        texture: int,
        px_to_world: float = 1.0,  # accepted for compatibility; not used
        v_tiles: Optional[float] = 1.0,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        # Compatibility-only parameters (ignored)
        points: Optional[Sequence[Tuple[float, float]] | Sequence[Vector3]] = None,
        # New: sample heights to conform to terrain (either via callable or sampler object)
        height_fn: Optional[Callable[[float, float], float]] = None,
        height_sampler: Optional[object] = None,
        elevation: float = 0.02,
        segment_length: float = 20.0,
    ) -> None:
        # Derive start/end from points if provided with at least 2 entries
        if (start is None or end is None) and points is not None and len(points) >= 2:
            s = points[0]
            e = points[-1]
            start = (
                float(getattr(s, "x", s[0])),
                float(getattr(s, "z", s[1])),
            )
            end = (
                float(getattr(e, "x", e[0])),
                float(getattr(e, "z", e[1])),
            )

        assert (
            start is not None and end is not None
        ), "Road requires start and end or two points"

        self.start = (
            start if isinstance(start, Vector3) else Vector3(start[0], 0.0, start[1])
        )
        self.end = end if isinstance(end, Vector3) else Vector3(end[0], 0.0, end[1])
        self.ground_y = float(ground_y)
        self.width = float(width)
        self.texture = int(texture)
        self.v_tiles = float(v_tiles) if v_tiles is not None else 1.0
        self.color = color
        # Prefer explicit height_fn, else use sampler.height_at if provided
        if height_fn is not None:
            self.height_fn = height_fn
        elif height_sampler is not None and hasattr(height_sampler, "height_at"):
            self.height_fn = lambda x, z: float(height_sampler.height_at(x, z))
        else:
            self.height_fn = None
        self.elevation = float(elevation)
        self.segment_length = max(1.0, float(segment_length))
        self._mesh = None

        _ensure_texture_repeat(self.texture)
        self._rebuild()

    def _rebuild(self) -> None:
        """Build the road strip.

        If a terrain height function is available, tessellate along the path and
        sample the ground height at the left/right edges so the road conforms.
        Otherwise, build a single flat quad at ground_y.
        """
        dx = self.end.x - self.start.x
        dz = self.end.z - self.start.z
        length = (dx * dx + dz * dz) ** 0.5

        if length <= 1e-6 or self.width <= 1e-6:
            empty = np.zeros((0, 8), dtype=np.float32)
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, empty.nbytes, empty, GL_STATIC_DRAW)
            self._mesh = BatchedMesh(
                vbo_vertices=vbo, vertex_count=0, texture=self.texture
            )
            return

        nx, nz = _perp_right(dx, dz)
        half_w = self.width * 0.5
        y = self.ground_y + self.elevation  # base height when flat
        r, g, b = self.color

        verts = []

        if self.height_fn is None:
            # Flat: split width into 4 strips (5 columns) so we can curve the
            # outermost vertices into the ground. Columns are equally spaced
            # from -half_w..+half_w.
            cols = np.linspace(-half_w, half_w, 5)
            u_vals = np.linspace(0.0, max(1e-6, float(self.v_tiles)), 5)
            # V coordinates: along length (same as before)
            v0 = 0.0
            v1 = length / max(1e-6, self.width)

            # Y profile across the width to create a soft curve into the ground:
            # outermost columns at ground_y, next-inner columns partially
            # elevated, center column at full elevation.
            y_center = self.ground_y + self.elevation
            y_inner = self.ground_y + (self.elevation * 0.5)
            y_outer = self.ground_y
            y_vals = [y_outer, y_inner, y_center, y_inner, y_outer]

            # Compute column positions for start and end
            s_cols = [
                (self.start.x + nx * off, self.start.z + nz * off) for off in cols
            ]
            e_cols = [(self.end.x + nx * off, self.end.z + nz * off) for off in cols]

            # For each of the 4 lateral strips generate two triangles
            for i in range(4):
                sx_l, sz_l = s_cols[i]
                sx_r, sz_r = s_cols[i + 1]
                ex_l, ez_l = e_cols[i]
                ex_r, ez_r = e_cols[i + 1]

                y_sl = y_vals[i]
                y_sr = y_vals[i + 1]
                y_el = y_vals[i]
                y_er = y_vals[i + 1]

                u_l = float(u_vals[i])
                u_r = float(u_vals[i + 1])

                verts.extend(
                    [
                        # tri 1: start-left, start-right, end-right
                        (sx_l, y_sl, sz_l, r, g, b, u_l, v0),
                        (sx_r, y_sr, sz_r, r, g, b, u_r, v0),
                        (ex_r, y_er, ez_r, r, g, b, u_r, v1),
                        # tri 2: start-left, end-right, end-left
                        (sx_l, y_sl, sz_l, r, g, b, u_l, v0),
                        (ex_r, y_er, ez_r, r, g, b, u_r, v1),
                        (ex_l, y_el, ez_l, r, g, b, u_l, v1),
                    ]
                )
        else:
            # Conforming: subdivide along length and sample heights at edges
            n = max(1, int(np.ceil(length / self.segment_length)))
            # Direction unit vector
            if length > 0:
                ux, uz = dx / length, dz / length
            else:
                ux, uz = 0.0, 0.0

            # Offsets for the 5 column positions across the road width
            col_offsets = np.linspace(-half_w, half_w, 5)

            for i in range(n):
                t0 = i / n
                t1 = (i + 1) / n
                # Positions along centerline
                cx0 = self.start.x + ux * (length * t0)
                cz0 = self.start.z + uz * (length * t0)
                cx1 = self.start.x + ux * (length * t1)
                cz1 = self.start.z + uz * (length * t1)
                # Compute per-column positions and sample heights. For the
                # outermost columns we'll use the raw sampled height (no
                # elevation) so the strip visually sinks into the ground; the
                # inner columns get partial/full elevation for a smooth blend.
                col_pos0 = []
                col_pos1 = []
                for off in col_offsets:
                    col_pos0.append((cx0 + nx * off, cz0 + nz * off))
                    col_pos1.append((cx1 + nx * off, cz1 + nz * off))

                # Sample heights for each column at t0 and t1, apply elevation
                # profile across columns: [outer, inner, center, inner, outer]
                heights0 = []
                heights1 = []
                for j, (px0, pz0) in enumerate(col_pos0):
                    base_h0 = float(self.height_fn(px0, pz0))
                    if j == 0 or j == 4:
                        heights0.append(base_h0)
                    elif j == 1 or j == 3:
                        heights0.append(base_h0 + (self.elevation * 0.5))
                    else:
                        heights0.append(base_h0 + self.elevation)

                for j, (px1, pz1) in enumerate(col_pos1):
                    base_h1 = float(self.height_fn(px1, pz1))
                    if j == 0 or j == 4:
                        heights1.append(base_h1)
                    elif j == 1 or j == 3:
                        heights1.append(base_h1 + (self.elevation * 0.5))
                    else:
                        heights1.append(base_h1 + self.elevation)

                # Texture coordinates: U across width (0..v_tiles), V along length
                v_len0 = (t0 * length) / max(1e-6, self.width)
                v_len1 = (t1 * length) / max(1e-6, self.width)
                u_vals = np.linspace(0.0, max(1e-6, float(self.v_tiles)), 5)

                # Emit two tris per lateral strip (4 strips per slice)
                for j in range(4):
                    lx0, lz0 = col_pos0[j]
                    rx0, rz0 = col_pos0[j + 1]
                    lx1, lz1 = col_pos1[j]
                    rx1, rz1 = col_pos1[j + 1]

                    y_l0 = heights0[j]
                    y_r0 = heights0[j + 1]
                    y_l1 = heights1[j]
                    y_r1 = heights1[j + 1]

                    u_l = float(u_vals[j])
                    u_r = float(u_vals[j + 1])

                    # tri 1: left0, right0, right1
                    verts.append((lx0, y_l0, lz0, r, g, b, u_l, v_len0))
                    verts.append((rx0, y_r0, rz0, r, g, b, u_r, v_len0))
                    verts.append((rx1, y_r1, rz1, r, g, b, u_r, v_len1))
                    # tri 2: left0, right1, left1
                    verts.append((lx0, y_l0, lz0, r, g, b, u_l, v_len0))
                    verts.append((rx1, y_r1, rz1, r, g, b, u_r, v_len1))
                    verts.append((lx1, y_l1, lz1, r, g, b, u_l, v_len1))

        vertex_data = np.array(verts, dtype=np.float32)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        self._mesh = BatchedMesh(
            vbo_vertices=vbo, vertex_count=vertex_data.shape[0], texture=self.texture
        )

    # No centerline/path logic — intentionally removed for simplicity

    # Public API ---------------------------------------------------------
    def update_endpoints(
        self, start: Vector3 | Tuple[float, float], end: Vector3 | Tuple[float, float]
    ) -> None:
        """Update start/end and rebuild the quad."""
        self.start = (
            start if isinstance(start, Vector3) else Vector3(start[0], 0.0, start[1])
        )
        self.end = end if isinstance(end, Vector3) else Vector3(end[0], 0.0, end[1])
        self._rebuild()

    def draw_untextured(self) -> None:  # parity with Drawable
        self.draw()

    def draw(self) -> None:
        if self._mesh is None:
            return
        _ensure_texture_repeat(self.texture)
        self._mesh.draw()

    # Geometry queries ----------------------------------------------------
    def contains_point(self, x: float, z: float, *, margin: float = 0.0) -> bool:
        """Return True if the XZ point lies over the straight road strip."""
        half_w = (self.width * 0.5) + max(0.0, float(margin))
        half_w2 = half_w * half_w

        vx = self.end.x - self.start.x
        vz = self.end.z - self.start.z
        len2 = vx * vx + vz * vz
        if len2 <= 1e-12:
            return False
        px = x - self.start.x
        pz = z - self.start.z
        t = (px * vx + pz * vz) / len2
        if t < 0.0 or t > 1.0:
            return False
        cx = self.start.x + vx * t
        cz = self.start.z + vz * t
        dx = x - cx
        dz = z - cz
        return (dx * dx + dz * dz) <= half_w2
