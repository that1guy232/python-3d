"""Road mesh generation.

The world planner owns route selection. This module owns the renderable road
object and the builder that turns a cleaned centerline into a textured VBO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from OpenGL.GL import (
    GL_REPEAT,
    GL_TEXTURE_2D,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    glBindTexture,
    glTexParameteri,
)
from pygame.math import Vector3

from core.mesh import BatchedMesh
from core.compat_shader import texture_color_exposure_shader_available
from engine.rendering.lighting import (
    apply_brightness_modifiers,
    apply_directional_sunlight,
    with_textured_normals,
)

PointXZ = tuple[float, float]
SegmentXZ = tuple[PointXZ, PointXZ]
HeightFn = Callable[[float, float], float]
RoadColumn = tuple[float, float, float, float, float]

_EPSILON = 1e-6
_MITER_LIMIT = 3.0
_ROAD_COLUMNS = 5


def _as_xz(point: Vector3 | Sequence[float]) -> PointXZ:
    """Return a point as XZ coordinates.

    Tuple/list inputs keep the historical ``(x, z)`` interpretation. Vector3
    inputs use their named ``x`` and ``z`` attributes.
    """
    if hasattr(point, "x") and hasattr(point, "z"):
        return float(getattr(point, "x")), float(getattr(point, "z"))
    return float(point[0]), float(point[1])


def _distance_xz(dx: float, dz: float) -> float:
    return (dx * dx + dz * dz) ** 0.5


def _normalize_xz(dx: float, dz: float) -> PointXZ:
    length = _distance_xz(dx, dz)
    if length <= _EPSILON:
        return 0.0, 0.0
    return dx / length, dz / length


def _clean_centerline(
    points: Sequence[PointXZ],
    *,
    min_segment_length: float = 1e-4,
) -> list[PointXZ]:
    """Drop duplicate and collinear centerline points."""
    clean: list[PointXZ] = []
    for point in points:
        if clean:
            dx = point[0] - clean[-1][0]
            dz = point[1] - clean[-1][1]
            if _distance_xz(dx, dz) <= min_segment_length:
                continue
        clean.append(point)

    index = 1
    while index < len(clean) - 1:
        prev_point = clean[index - 1]
        point = clean[index]
        next_point = clean[index + 1]
        first = _normalize_xz(point[0] - prev_point[0], point[1] - prev_point[1])
        second = _normalize_xz(next_point[0] - point[0], next_point[1] - point[1])
        cross = abs(first[0] * second[1] - first[1] * second[0])
        dot = first[0] * second[0] + first[1] * second[1]
        if cross <= 1e-4 and dot > 0.0:
            del clean[index]
        else:
            index += 1

    return clean


def _line_intersection(
    p: PointXZ,
    d: PointXZ,
    q: PointXZ,
    e: PointXZ,
) -> PointXZ | None:
    """Return the XZ intersection of two infinite 2D lines."""
    denom = d[0] * e[1] - d[1] * e[0]
    if abs(denom) <= _EPSILON:
        return None

    qmp_x = q[0] - p[0]
    qmp_z = q[1] - p[1]
    t = (qmp_x * e[1] - qmp_z * e[0]) / denom
    return p[0] + d[0] * t, p[1] + d[1] * t


def _point_segment_distance_sq(point: PointXZ, segment: SegmentXZ) -> float:
    px, pz = point
    (x0, z0), (x1, z1) = segment
    vx = x1 - x0
    vz = z1 - z0
    length_sq = vx * vx + vz * vz
    if length_sq <= _EPSILON:
        dx = px - x0
        dz = pz - z0
        return dx * dx + dz * dz

    t = ((px - x0) * vx + (pz - z0) * vz) / length_sq
    t = max(0.0, min(1.0, t))
    cx = x0 + vx * t
    cz = z0 + vz * t
    dx = px - cx
    dz = pz - cz
    return dx * dx + dz * dz


def _ensure_texture_repeat(tex_id: int) -> None:
    """Set wrap mode to REPEAT for the given texture."""
    if not tex_id:
        return
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)


def _upload_mesh(
    vertex_data: np.ndarray,
    texture: int,
    *,
    exposure_baseline: float = 1.0,
) -> BatchedMesh:
    return BatchedMesh.from_vertex_data(
        vertex_data,
        texture=texture,
        exposure_baseline=exposure_baseline,
    )


def _empty_mesh(texture: int, *, exposure_baseline: float = 1.0) -> BatchedMesh:
    return _upload_mesh(
        np.zeros((0, 8), dtype=np.float32),
        texture,
        exposure_baseline=exposure_baseline,
    )


def _apply_brightness(
    vertex_data: np.ndarray,
    *,
    modifiers: Sequence[object],
    default_brightness: float,
) -> None:
    """Apply the same brightness-modifier contract as the ground builder."""
    apply_brightness_modifiers(
        vertex_data,
        modifiers=modifiers,
        default_brightness=default_brightness,
    )


@dataclass
class RoadMeshBuilder:
    """Build a terrain-conforming road mesh from a centerline path."""

    ground_y: float
    width: float
    texture: int
    v_tiles: float = 1.0
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    height_fn: HeightFn | None = None
    elevation: float = 0.02
    segment_length: float = 20.0
    brightness_modifiers: Sequence[object] = ()
    default_brightness: float = 1.0
    lighting: object | None = None
    sun_direction: object | None = None

    def build(self, centerline: Sequence[Vector3]) -> BatchedMesh:
        sampled_points, distances = self._sample_centerline(centerline)
        if len(sampled_points) < 2 or self.width <= _EPSILON:
            return _empty_mesh(self.texture, exposure_baseline=self.default_brightness)

        directions, normals, lengths = self._path_frames(sampled_points)
        if not lengths or sum(lengths) <= _EPSILON:
            return _empty_mesh(self.texture, exposure_baseline=self.default_brightness)

        sections = self._build_sections(sampled_points, distances, directions, normals)
        vertex_data = self._build_vertex_data(sections)
        if texture_color_exposure_shader_available():
            vertex_data = with_textured_normals(
                vertex_data,
                prefer_upward_normals=True,
            )
        else:
            _apply_brightness(
                vertex_data,
                modifiers=self.brightness_modifiers,
                default_brightness=self.default_brightness,
            )
            apply_directional_sunlight(
                vertex_data,
                lighting=self.lighting,
                sun_direction=self.sun_direction,
                prefer_upward_normals=True,
            )
        return _upload_mesh(
            vertex_data,
            self.texture,
            exposure_baseline=self.default_brightness,
        )

    def _sample_centerline(
        self, centerline: Sequence[Vector3]
    ) -> tuple[list[PointXZ], list[float]]:
        base_points = [(point.x, point.z) for point in centerline]
        if len(base_points) < 2:
            return base_points, [0.0] * len(base_points)

        sampled = [base_points[0]]
        distances = [0.0]
        cumulative = 0.0
        for p0, p1 in zip(base_points, base_points[1:]):
            dx = p1[0] - p0[0]
            dz = p1[1] - p0[1]
            segment_length = _distance_xz(dx, dz)
            if segment_length <= _EPSILON:
                continue

            divisions = 1
            if self.height_fn is not None:
                divisions = max(1, int(np.ceil(segment_length / self.segment_length)))

            for step in range(1, divisions + 1):
                t = step / divisions
                sampled.append((p0[0] + dx * t, p0[1] + dz * t))
                distances.append(cumulative + segment_length * t)
            cumulative += segment_length

        return sampled, distances

    @staticmethod
    def _path_frames(
        points: Sequence[PointXZ],
    ) -> tuple[list[PointXZ], list[PointXZ], list[float]]:
        directions: list[PointXZ] = []
        normals: list[PointXZ] = []
        lengths: list[float] = []

        for p0, p1 in zip(points, points[1:]):
            dx = p1[0] - p0[0]
            dz = p1[1] - p0[1]
            length = _distance_xz(dx, dz)
            if length <= _EPSILON:
                directions.append((0.0, 0.0))
                normals.append((1.0, 0.0))
                lengths.append(0.0)
                continue
            ux, uz = dx / length, dz / length
            directions.append((ux, uz))
            normals.append((uz, -ux))
            lengths.append(length)

        return directions, normals, lengths

    def _build_sections(
        self,
        points: Sequence[PointXZ],
        distances: Sequence[float],
        directions: Sequence[PointXZ],
        normals: Sequence[PointXZ],
    ) -> list[list[RoadColumn]]:
        half_width = self.width * 0.5
        column_offsets = np.linspace(-half_width, half_width, _ROAD_COLUMNS)
        u_values = np.linspace(0.0, max(_EPSILON, self.v_tiles), _ROAD_COLUMNS)

        sections: list[list[RoadColumn]] = []
        for point_index, distance in enumerate(distances):
            section: list[RoadColumn] = []
            v_coord = distance / max(_EPSILON, self.width)
            for column, offset in enumerate(column_offsets):
                x, z = self._join_column_position(
                    points,
                    directions,
                    normals,
                    point_index,
                    float(offset),
                )
                y = self._column_y(x, z, column)
                section.append((x, y, z, float(u_values[column]), v_coord))
            sections.append(section)

        return sections

    @staticmethod
    def _join_column_position(
        points: Sequence[PointXZ],
        directions: Sequence[PointXZ],
        normals: Sequence[PointXZ],
        index: int,
        offset: float,
    ) -> PointXZ:
        px, pz = points[index]
        if abs(offset) <= _EPSILON:
            return px, pz
        if index <= 0:
            nx, nz = normals[0]
            return px + nx * offset, pz + nz * offset
        if index >= len(points) - 1:
            nx, nz = normals[-1]
            return px + nx * offset, pz + nz * offset

        prev_dir = directions[index - 1]
        next_dir = directions[index]
        prev_normal = normals[index - 1]
        next_normal = normals[index]
        prev_line = (px + prev_normal[0] * offset, pz + prev_normal[1] * offset)
        next_line = (px + next_normal[0] * offset, pz + next_normal[1] * offset)
        joined = _line_intersection(prev_line, prev_dir, next_line, next_dir)

        if joined is None:
            nx, nz = _normalize_xz(
                prev_normal[0] + next_normal[0],
                prev_normal[1] + next_normal[1],
            )
            if abs(nx) <= _EPSILON and abs(nz) <= _EPSILON:
                nx, nz = prev_normal
            return px + nx * offset, pz + nz * offset

        rel_x = joined[0] - px
        rel_z = joined[1] - pz
        rel_len = _distance_xz(rel_x, rel_z)
        limit = max(abs(offset) * _MITER_LIMIT, abs(offset) + _EPSILON)
        if rel_len > limit:
            scale = limit / rel_len
            return px + rel_x * scale, pz + rel_z * scale

        return joined

    def _column_y(self, x: float, z: float, column: int) -> float:
        center_column = (_ROAD_COLUMNS - 1) * 0.5
        distance_from_center = abs(float(column) - center_column)
        elevation_fraction = 1.0 - (
            distance_from_center / max(center_column, 1.0)
        )
        elevation_fraction = max(0.0, min(1.0, elevation_fraction))
        base_y = (
            float(self.height_fn(x, z))
            if self.height_fn is not None
            else self.ground_y
        )
        return base_y + self.elevation * elevation_fraction

    def _build_vertex_data(
        self, sections: Sequence[Sequence[RoadColumn]]
    ) -> np.ndarray:
        r, g, b = self.color
        verts: list[tuple[float, float, float, float, float, float, float, float]] = []

        for first, second in zip(sections, sections[1:]):
            for column in range(_ROAD_COLUMNS - 1):
                lx0, y_l0, lz0, u_l, v0 = first[column]
                rx0, y_r0, rz0, u_r, _ = first[column + 1]
                lx1, y_l1, lz1, _, v1 = second[column]
                rx1, y_r1, rz1, _, _ = second[column + 1]

                verts.append((lx0, y_l0, lz0, r, g, b, u_l, v0))
                verts.append((rx0, y_r0, rz0, r, g, b, u_r, v0))
                verts.append((rx1, y_r1, rz1, r, g, b, u_r, v1))
                verts.append((lx0, y_l0, lz0, r, g, b, u_l, v0))
                verts.append((rx1, y_r1, rz1, r, g, b, u_r, v1))
                verts.append((lx1, y_l1, lz1, r, g, b, u_l, v1))

        if not verts:
            return np.zeros((0, 8), dtype=np.float32)
        return np.array(verts, dtype=np.float32)


@dataclass
class Road:
    """Textured road strip following a straight or polyline centerline."""

    start: Vector3
    end: Vector3
    points: list[Vector3]
    ground_y: float
    width: float
    texture: int
    v_tiles: float = 1.0
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    height_fn: HeightFn | None = None
    elevation: float = 0.02
    segment_length: float = 20.0
    brightness_modifiers: Sequence[object] | None = None
    default_brightness: float = 1.0
    lighting: object | None = None
    sun_direction: object | None = None
    _mesh: BatchedMesh | None = None

    def __init__(
        self,
        *,
        start: Vector3 | Tuple[float, float] | None = None,
        end: Vector3 | Tuple[float, float] | None = None,
        ground_y: float,
        width: float,
        texture: int,
        px_to_world: float = 1.0,
        v_tiles: Optional[float] = 1.0,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        points: Optional[Sequence[Tuple[float, float]] | Sequence[Vector3]] = None,
        height_fn: Optional[HeightFn] = None,
        height_sampler: Optional[object] = None,
        elevation: float = 0.02,
        segment_length: float = 20.0,
        brightness_modifiers: Optional[Sequence[object]] = None,
        default_brightness: float = 1.0,
        lighting=None,
        sun_direction=None,
    ) -> None:
        del px_to_world  # accepted for compatibility with older builders

        self.ground_y = float(ground_y)
        self.width = float(width)
        self.texture = int(texture)
        self.v_tiles = float(v_tiles) if v_tiles is not None else 1.0
        self.color = color
        self.brightness_modifiers = brightness_modifiers or ()
        self.default_brightness = float(default_brightness)
        self.lighting = lighting
        self.sun_direction = sun_direction
        self.height_fn = self._resolve_height_fn(height_fn, height_sampler)
        self.elevation = float(elevation)
        self.segment_length = max(1.0, float(segment_length))
        self._mesh = None

        self._set_centerline(
            self._resolve_points(start=start, end=end, points=points)
        )
        _ensure_texture_repeat(self.texture)
        self._rebuild()

    @staticmethod
    def _resolve_height_fn(
        height_fn: HeightFn | None,
        height_sampler: object | None,
    ) -> HeightFn | None:
        if height_fn is not None:
            return height_fn
        if height_sampler is not None and hasattr(height_sampler, "height_at"):
            return lambda x, z: float(height_sampler.height_at(x, z))
        return None

    @staticmethod
    def _resolve_points(
        *,
        start: Vector3 | Tuple[float, float] | None,
        end: Vector3 | Tuple[float, float] | None,
        points: Optional[Sequence[Tuple[float, float]] | Sequence[Vector3]],
    ) -> list[PointXZ]:
        raw_points: list[Vector3 | Sequence[float]] = []
        if points is not None:
            raw_points = list(points)
        if len(raw_points) < 2:
            if start is None or end is None:
                raise ValueError("Road requires start and end or at least two points")
            raw_points = [start, end]

        centerline = _clean_centerline([_as_xz(point) for point in raw_points])
        if len(centerline) < 2:
            raise ValueError("Road requires at least two distinct points")
        return centerline

    @staticmethod
    def _clean_centerline(
        points: list[PointXZ],
        *,
        min_segment_length: float = 1e-4,
    ) -> list[PointXZ]:
        return _clean_centerline(points, min_segment_length=min_segment_length)

    def _set_centerline(self, centerline: Sequence[PointXZ]) -> None:
        self.points = [Vector3(x, 0.0, z) for x, z in centerline]
        self.start = self.points[0]
        self.end = self.points[-1]

    def _mesh_builder(self) -> RoadMeshBuilder:
        return RoadMeshBuilder(
            ground_y=self.ground_y,
            width=self.width,
            texture=self.texture,
            v_tiles=self.v_tiles,
            color=self.color,
            height_fn=self.height_fn,
            elevation=self.elevation,
            segment_length=self.segment_length,
            brightness_modifiers=self.brightness_modifiers or (),
            default_brightness=self.default_brightness,
            lighting=self.lighting,
            sun_direction=self.sun_direction,
        )

    def _rebuild(self) -> None:
        if self._mesh is not None:
            self._mesh.dispose()
        self._mesh = self._mesh_builder().build(self.points)

    def dispose(self) -> None:
        if self._mesh is not None:
            self._mesh.dispose()
            self._mesh = None

    def update_points(
        self, points: Sequence[Tuple[float, float]] | Sequence[Vector3]
    ) -> None:
        """Update the road centerline and rebuild the mesh."""
        centerline = _clean_centerline([_as_xz(point) for point in points])
        if len(centerline) < 2:
            raise ValueError("Road requires at least two distinct points")
        self._set_centerline(centerline)
        self._rebuild()

    def update_endpoints(
        self, start: Vector3 | Tuple[float, float], end: Vector3 | Tuple[float, float]
    ) -> None:
        """Update start/end and rebuild the road."""
        self.update_points([start, end])

    def refresh_lighting(
        self,
        *,
        brightness_modifiers: Optional[Sequence[object]] = None,
        default_brightness: Optional[float] = None,
        lighting=None,
        sun_direction=None,
        height_sampler: Optional[object] = None,
    ) -> None:
        """Refresh baked vertex lighting without changing the road route."""
        if brightness_modifiers is not None:
            self.brightness_modifiers = brightness_modifiers
        if default_brightness is not None:
            self.default_brightness = float(default_brightness)
        if lighting is not None:
            self.lighting = lighting
        if sun_direction is not None:
            self.sun_direction = sun_direction
        if height_sampler is not None:
            self.height_fn = self._resolve_height_fn(None, height_sampler)
        self._rebuild()

    def set_exposure(self, exposure: float) -> None:
        if self._mesh is not None:
            self._mesh.set_exposure(exposure)

    def draw_untextured(self) -> None:
        self.draw()

    def draw(self) -> None:
        if self._mesh is None:
            return
        _ensure_texture_repeat(self.texture)
        self._mesh.draw()

    def contains_point(self, x: float, z: float, *, margin: float = 0.0) -> bool:
        """Return True if the XZ point lies over the road centerline strip."""
        half_width = (self.width * 0.5) + max(0.0, float(margin))
        half_width_sq = half_width * half_width

        query = (float(x), float(z))
        centerline = [(point.x, point.z) for point in self.points]
        return any(
            _point_segment_distance_sq(query, segment) <= half_width_sq
            for segment in zip(centerline, centerline[1:])
        )

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """Return conservative XZ extents as (min_x, max_x, min_z, max_z)."""
        pad = self.width * 0.5 * _MITER_LIMIT
        min_x = min(point.x for point in self.points) - pad
        max_x = max(point.x for point in self.points) + pad
        min_z = min(point.z for point in self.points) - pad
        max_z = max(point.z for point in self.points) + pad
        return (min_x, max_x, min_z, max_z)

    def get_bounds(self) -> tuple[float, float, float, float]:
        return self.get_bounding_box()
