"""Deterministic low-poly trees and spatially chunked render batches.

``Tree`` is deliberately a small CPU-side semantic object.  Its position is
the point where the trunk meets the terrain; generated vertices are already
in world space.  Rendering is kept separate so many trees can share a handful
of VBOs while placement, bounds, and procedural parameters remain available
without an OpenGL context.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
import math
import random
from typing import NamedTuple, TypeAlias

import numpy as np
from pygame.math import Vector3

from engine.rendering.mesh import BatchedMesh
from engine.textures.texture_utils import create_solid_color_texture
from game.world.rendering.receivers import TREE_LIGHTING_RECEIVER


TREE_VARIANT_COUNT = 7
TREE_LOD_COUNT = 3


class TreeLOD(IntEnum):
    """Procedural tree detail levels, ordered from most to least detailed."""

    HIGH = 0
    MEDIUM = 1
    LOW = 2


TreeChunkKey: TypeAlias = tuple[int, int]
TreeLODMeshGroup: TypeAlias = tuple[
    BatchedMesh | None,
    BatchedMesh | None,
    BatchedMesh | None,
]

_TAU = math.tau
_EPSILON = 1.0e-10
_UINT64_MASK = (1 << 64) - 1


class _VariantSpec(NamedTuple):
    """Stable proportions and palette for one authored tree family."""

    name: str
    conifer: bool
    min_height: float
    max_height: float
    trunk_ratio: float
    crown_ratio: float
    bark_color: tuple[float, float, float]
    foliage_color: tuple[float, float, float]


# Four broadleaf silhouettes followed by three conifer silhouettes.  These
# values are proportions, so explicit height overrides retain their family
# shape instead of merely stretching one axis of fixed geometry.
_VARIANTS: tuple[_VariantSpec, ...] = (
    _VariantSpec(
        "oak", False, 88.0, 118.0, 0.050, 0.305, (0.31, 0.20, 0.105), (0.22, 0.47, 0.16)
    ),
    _VariantSpec(
        "maple", False, 84.0, 112.0, 0.047, 0.335, (0.29, 0.17, 0.080), (0.32, 0.53, 0.15)
    ),
    _VariantSpec(
        "birch", False, 96.0, 128.0, 0.036, 0.245, (0.63, 0.59, 0.49), (0.30, 0.56, 0.19)
    ),
    _VariantSpec(
        "acacia", False, 80.0, 106.0, 0.046, 0.365, (0.34, 0.22, 0.105), (0.27, 0.45, 0.13)
    ),
    _VariantSpec(
        "pine", True, 110.0, 148.0, 0.040, 0.255, (0.29, 0.20, 0.105), (0.14, 0.32, 0.17)
    ),
    _VariantSpec(
        "fir", True, 98.0, 136.0, 0.048, 0.285, (0.27, 0.18, 0.090), (0.12, 0.37, 0.20)
    ),
    _VariantSpec(
        "cypress", True, 106.0, 144.0, 0.034, 0.180, (0.31, 0.21, 0.105), (0.18, 0.39, 0.18)
    ),
)


def _seeded_rng(seed: int, channel: int) -> random.Random:
    """Return an independent deterministic random stream for one concern."""

    value = int(seed) & _UINT64_MASK
    value ^= (int(channel) * 0x9E3779B97F4A7C15) & _UINT64_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return random.Random(value)


def _coerce_position(
    position: Vector3 | Sequence[float],
) -> Vector3:
    """Copy a two- or three-component position, mapping ``(x, z)`` to XZ."""

    if isinstance(position, Vector3):
        result = Vector3(position)
    else:
        try:
            values = tuple(position)
        except TypeError as exc:
            raise TypeError("tree position must be a Vector3 or a 2/3 item sequence") from exc
        if len(values) == 2:
            result = Vector3(float(values[0]), 0.0, float(values[1]))
        elif len(values) == 3:
            result = Vector3(float(values[0]), float(values[1]), float(values[2]))
        else:
            raise ValueError("tree position must contain two or three coordinates")

    if not all(math.isfinite(float(component)) for component in result):
        raise ValueError("tree position coordinates must be finite")
    return result


def _validated_variant(variant: int) -> int:
    value = int(variant)
    if not 0 <= value < TREE_VARIANT_COUNT:
        raise ValueError(
            f"tree variant must be between 0 and {TREE_VARIANT_COUNT - 1}"
        )
    return value


def _validated_lod(lod_level: TreeLOD | int) -> TreeLOD:
    try:
        return TreeLOD(int(lod_level))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"tree lod_level must be between 0 and {TREE_LOD_COUNT - 1}"
        ) from exc


def _height_for_seed(seed: int, variant: int) -> float:
    spec = _VARIANTS[variant]
    rng = _seeded_rng(seed, 0x48454947 + variant)
    return rng.uniform(spec.min_height, spec.max_height)


def _radii_for_seed(seed: int, variant: int, height: float) -> tuple[float, float]:
    """Derive radii independently of whether height was supplied explicitly."""

    spec = _VARIANTS[variant]
    rng = _seeded_rng(seed, 0x52414449 + variant)
    trunk_radius = height * spec.trunk_ratio * rng.uniform(0.90, 1.10)
    crown_radius = height * spec.crown_ratio * rng.uniform(0.92, 1.08)
    return max(0.35, trunk_radius), max(1.0, crown_radius)


@dataclass(slots=True, init=False)
class Tree:
    """A deterministic procedural-tree instance anchored at terrain height.

    ``position.y`` is the bottom of the trunk and ``height`` is the full
    above-ground height.  ``crown_radius`` is intentionally used for placement
    and XZ bounds: it is conservative for spawning and encloses every branch
    and leaf vertex generated by this module.
    """

    position: Vector3
    seed: int
    variant: int
    height: float
    trunk_radius: float
    crown_radius: float

    def __init__(
        self,
        position: Vector3 | Sequence[float],
        seed: int = 0,
        variant: int = 0,
        height: float | None = None,
        *,
        trunk_radius: float | None = None,
        crown_radius: float | None = None,
    ) -> None:
        position_value = _coerce_position(position)
        seed_value = int(seed)
        variant_value = _validated_variant(variant)
        height_value = (
            _height_for_seed(seed_value, variant_value)
            if height is None
            else float(height)
        )
        if not math.isfinite(height_value) or height_value <= 0.0:
            raise ValueError("tree height must be a positive finite number")

        derived_trunk, derived_crown = _radii_for_seed(
            seed_value, variant_value, height_value
        )
        trunk_value = derived_trunk if trunk_radius is None else float(trunk_radius)
        crown_value = derived_crown if crown_radius is None else float(crown_radius)
        if not math.isfinite(trunk_value) or trunk_value <= 0.0:
            raise ValueError("tree trunk radius must be a positive finite number")
        if not math.isfinite(crown_value) or crown_value <= 0.0:
            raise ValueError("tree crown radius must be a positive finite number")

        self.position = position_value
        self.seed = seed_value
        self.variant = variant_value
        self.height = height_value
        self.trunk_radius = trunk_value
        self.crown_radius = max(crown_value, trunk_value)

    @classmethod
    def from_seed(
        cls,
        position: Vector3 | Sequence[float],
        seed: int,
        variant: int | None = None,
        height: float | None = None,
    ) -> "Tree":
        """Create a repeatable variant and set of dimensions from ``seed``.

        Variant, height, dimensions, and geometry use separate random streams.
        Consequently rebuilding a prototype with its explicit height produces
        exactly the same shape and placement radius, which is useful when the
        terrain height is not known until after placement succeeds.
        """

        seed_value = int(seed)
        variant_value = (
            _seeded_rng(seed_value, 0x56415249).randrange(TREE_VARIANT_COUNT)
            if variant is None
            else _validated_variant(variant)
        )
        height_value = (
            _height_for_seed(seed_value, variant_value)
            if height is None
            else float(height)
        )
        return cls(
            position,
            seed=seed_value,
            variant=variant_value,
            height=height_value,
        )

    @property
    def size(self) -> tuple[float, float]:
        """Return billboard-compatible ``(crown diameter, full height)``."""

        return self.crown_radius * 2.0, self.height

    @property
    def placement_radius(self) -> float:
        """Conservative XZ radius used to space trees and avoid structures."""

        return self.crown_radius

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return XZ extents as ``(min_x, max_x, min_z, max_z)``."""

        radius = self.placement_radius
        return (
            float(self.position.x) - radius,
            float(self.position.x) + radius,
            float(self.position.z) - radius,
            float(self.position.z) + radius,
        )

    def get_bounds(self) -> tuple[float, float, float, float]:
        return self.bounds

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        return self.bounds

    def get_bounding_box3d(
        self,
    ) -> tuple[float, float, float, float, float, float]:
        """Return XYZ extents in the engine's min/max-per-axis ordering."""

        min_x, max_x, min_z, max_z = self.bounds
        return (
            min_x,
            max_x,
            float(self.position.y),
            float(self.position.y) + self.height,
            min_z,
            max_z,
        )

    def contains_point(self, x: float, z: float, margin: float = 0.0) -> bool:
        """Return whether an XZ point lies inside the circular crown footprint."""

        radius = self.placement_radius + max(0.0, float(margin))
        dx = float(x) - float(self.position.x)
        dz = float(z) - float(self.position.z)
        return dx * dx + dz * dz <= radius * radius


_Point = tuple[float, float, float]
_Color = tuple[float, float, float]
_UV = tuple[float, float]
_VertexRow = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


def _tint(color: _Color, rng: random.Random, variation: float) -> _Color:
    factor = 1.0 + rng.uniform(-variation, variation)
    return tuple(max(0.0, min(1.0, component * factor)) for component in color)  # type: ignore[return-value]


def _append_triangle(
    rows: list[_VertexRow],
    a: _Point,
    b: _Point,
    c: _Point,
    color: _Color,
    uvs: tuple[_UV, _UV, _UV],
) -> None:
    """Append one flat-shaded, non-degenerate textured triangle."""

    ab_x, ab_y, ab_z = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    ac_x, ac_y, ac_z = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = ab_y * ac_z - ab_z * ac_y
    ny = ab_z * ac_x - ab_x * ac_z
    nz = ab_x * ac_y - ab_y * ac_x
    length_sq = nx * nx + ny * ny + nz * nz
    if length_sq <= _EPSILON:
        return
    inverse_length = 1.0 / math.sqrt(length_sq)
    nx *= inverse_length
    ny *= inverse_length
    nz *= inverse_length
    r, g, blue = color
    for point, uv in zip((a, b, c), uvs):
        rows.append(
            (
                point[0],
                point[1],
                point[2],
                r,
                g,
                blue,
                nx,
                ny,
                nz,
                uv[0],
                uv[1],
            )
        )


def _cross(a: _Point, b: _Point) -> _Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalized(vector: _Point) -> _Point:
    length = math.sqrt(
        vector[0] * vector[0]
        + vector[1] * vector[1]
        + vector[2] * vector[2]
    )
    if length <= _EPSILON:
        raise ValueError("procedural tree segment endpoints must be distinct")
    inverse = 1.0 / length
    return vector[0] * inverse, vector[1] * inverse, vector[2] * inverse


def _add_tapered_segment(
    rows: list[_VertexRow],
    start: _Point,
    end: _Point,
    start_radius: float,
    end_radius: float,
    sides: int,
    color: _Color,
    rng: random.Random,
    *,
    phase: float,
) -> None:
    """Add the side faces of a low-poly tapered trunk or branch."""

    axis = _normalized(
        (end[0] - start[0], end[1] - start[1], end[2] - start[2])
    )
    helper: _Point = (1.0, 0.0, 0.0) if abs(axis[1]) > 0.90 else (0.0, 1.0, 0.0)
    basis_u = _normalized(_cross(helper, axis))
    basis_v = _cross(axis, basis_u)

    start_ring: list[_Point] = []
    end_ring: list[_Point] = []
    for index in range(sides):
        angle = phase + _TAU * index / sides
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rx = basis_u[0] * cosine + basis_v[0] * sine
        ry = basis_u[1] * cosine + basis_v[1] * sine
        rz = basis_u[2] * cosine + basis_v[2] * sine
        start_ring.append(
            (
                start[0] + rx * start_radius,
                start[1] + ry * start_radius,
                start[2] + rz * start_radius,
            )
        )
        end_ring.append(
            (
                end[0] + rx * end_radius,
                end[1] + ry * end_radius,
                end[2] + rz * end_radius,
            )
        )

    for index in range(sides):
        next_index = (index + 1) % sides
        u0 = index / sides
        u1 = (index + 1) / sides
        face_color = _tint(color, rng, 0.075)
        _append_triangle(
            rows,
            start_ring[index],
            start_ring[next_index],
            end_ring[next_index],
            face_color,
            ((u0, 0.0), (u1, 0.0), (u1, 1.0)),
        )
        _append_triangle(
            rows,
            start_ring[index],
            end_ring[next_index],
            end_ring[index],
            face_color,
            ((u0, 0.0), (u1, 1.0), (u0, 1.0)),
        )


def _ellipsoid_ring_point(
    center: _Point,
    radius_x: float,
    radius_z: float,
    radius_scale: float,
    y: float,
    angle: float,
    rotation: float,
) -> _Point:
    local_x = radius_x * radius_scale * math.cos(angle)
    local_z = radius_z * radius_scale * math.sin(angle)
    cos_rotation = math.cos(rotation)
    sin_rotation = math.sin(rotation)
    return (
        center[0] + local_x * cos_rotation - local_z * sin_rotation,
        y,
        center[2] + local_x * sin_rotation + local_z * cos_rotation,
    )


def _add_faceted_crown(
    rows: list[_VertexRow],
    center: _Point,
    radius_x: float,
    radius_y: float,
    radius_z: float,
    sides: int,
    color: _Color,
    rng: random.Random,
    *,
    rotation: float,
) -> None:
    """Add a two-ring, flat-shaded broadleaf crown (four faces per side)."""

    bottom = (center[0], center[1] - radius_y, center[2])
    top = (center[0], center[1] + radius_y, center[2])
    lower_y = center[1] - radius_y * 0.20
    upper_y = center[1] + radius_y * 0.22
    lower_ring = [
        _ellipsoid_ring_point(
            center,
            radius_x,
            radius_z,
            0.82,
            lower_y,
            _TAU * index / sides,
            rotation,
        )
        for index in range(sides)
    ]
    upper_ring = [
        _ellipsoid_ring_point(
            center,
            radius_x,
            radius_z,
            1.0,
            upper_y,
            _TAU * index / sides,
            rotation,
        )
        for index in range(sides)
    ]

    for index in range(sides):
        next_index = (index + 1) % sides
        u0 = index / sides
        u1 = (index + 1) / sides
        _append_triangle(
            rows,
            bottom,
            lower_ring[index],
            lower_ring[next_index],
            _tint(color, rng, 0.105),
            ((0.5, 0.0), (u0, 0.34), (u1, 0.34)),
        )
        middle_color = _tint(color, rng, 0.11)
        _append_triangle(
            rows,
            lower_ring[index],
            upper_ring[index],
            upper_ring[next_index],
            middle_color,
            ((u0, 0.34), (u0, 0.66), (u1, 0.66)),
        )
        _append_triangle(
            rows,
            lower_ring[index],
            upper_ring[next_index],
            lower_ring[next_index],
            middle_color,
            ((u0, 0.34), (u1, 0.66), (u1, 0.34)),
        )
        _append_triangle(
            rows,
            upper_ring[index],
            top,
            upper_ring[next_index],
            _tint(color, rng, 0.105),
            ((u0, 0.66), (0.5, 1.0), (u1, 0.66)),
        )


def _add_bipyramid_crown(
    rows: list[_VertexRow],
    center: _Point,
    radius_x: float,
    radius_y: float,
    radius_z: float,
    sides: int,
    color: _Color,
    rng: random.Random,
    *,
    rotation: float,
) -> None:
    """Add a compact two-fan crown used by medium and low tree LODs."""

    bottom = (center[0], center[1] - radius_y, center[2])
    top = (center[0], center[1] + radius_y, center[2])
    ring = [
        _ellipsoid_ring_point(
            center,
            radius_x,
            radius_z,
            1.0,
            center[1],
            _TAU * index / sides,
            rotation,
        )
        for index in range(sides)
    ]

    for index in range(sides):
        next_index = (index + 1) % sides
        u0 = index / sides
        u1 = (index + 1) / sides
        _append_triangle(
            rows,
            bottom,
            ring[index],
            ring[next_index],
            _tint(color, rng, 0.09),
            ((0.5, 0.0), (u0, 0.5), (u1, 0.5)),
        )
        _append_triangle(
            rows,
            ring[index],
            top,
            ring[next_index],
            _tint(color, rng, 0.09),
            ((u0, 0.5), (0.5, 1.0), (u1, 0.5)),
        )


def _add_foliage_cone(
    rows: list[_VertexRow],
    center_x: float,
    center_z: float,
    base_y: float,
    apex_y: float,
    radius: float,
    sides: int,
    color: _Color,
    rng: random.Random,
    *,
    phase: float,
) -> None:
    """Add an opaque conifer layer with both sloped and underside facets."""

    apex = (center_x, apex_y, center_z)
    base_center = (center_x, base_y, center_z)
    ring = [
        (
            center_x + math.cos(phase + _TAU * index / sides) * radius,
            base_y,
            center_z + math.sin(phase + _TAU * index / sides) * radius,
        )
        for index in range(sides)
    ]
    for index in range(sides):
        next_index = (index + 1) % sides
        u0 = index / sides
        u1 = (index + 1) / sides
        _append_triangle(
            rows,
            ring[index],
            apex,
            ring[next_index],
            _tint(color, rng, 0.095),
            ((u0, 0.0), (0.5, 1.0), (u1, 0.0)),
        )
        _append_triangle(
            rows,
            base_center,
            ring[next_index],
            ring[index],
            _tint(color, rng, 0.075),
            ((0.5, 0.5), (u1, 0.0), (u0, 0.0)),
        )


def _rotated_offset(x: float, z: float, rotation: float) -> tuple[float, float]:
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    return x * cosine - z * sine, x * sine + z * cosine


# ox, oz, center-y, radius-x, radius-y, radius-z.  Horizontal values are
# fractions of crown_radius; vertical values are fractions of full height.
_BROADLEAF_CLUSTERS: tuple[
    tuple[tuple[float, float, float, float, float, float], ...], ...
] = (
    (
        (0.00, 0.00, 0.73, 0.68, 0.27, 0.68),
        (0.34, 0.08, 0.72, 0.47, 0.19, 0.45),
        (-0.31, -0.10, 0.77, 0.45, 0.20, 0.48),
    ),
    (
        (0.00, 0.00, 0.75, 0.72, 0.25, 0.61),
        (0.31, 0.15, 0.69, 0.51, 0.18, 0.45),
        (-0.31, -0.12, 0.75, 0.49, 0.20, 0.47),
    ),
    (
        (-0.05, 0.00, 0.76, 0.48, 0.24, 0.43),
        (0.18, -0.04, 0.70, 0.38, 0.18, 0.40),
    ),
    (
        (0.00, 0.00, 0.80, 0.74, 0.20, 0.62),
        (0.32, 0.07, 0.78, 0.47, 0.14, 0.45),
        (-0.33, -0.06, 0.78, 0.46, 0.14, 0.44),
    ),
)


# center-y, radius-x, radius-y, radius-z.  LOD2 merges authored clumps into a
# single family-specific hull while retaining each variant's characteristic
# rounded, narrow, or spreading silhouette.
_BROADLEAF_LOW_CROWNS: tuple[tuple[float, float, float, float], ...] = (
    (0.73, 0.84, 0.27, 0.78),
    (0.74, 0.86, 0.26, 0.75),
    (0.76, 0.58, 0.24, 0.55),
    (0.80, 0.93, 0.20, 0.70),
)


def _build_broadleaf_geometry(
    tree: Tree, rows: list[_VertexRow], rng: random.Random
) -> None:
    spec = _VARIANTS[tree.variant]
    base_x, base_y, base_z = (
        float(tree.position.x),
        float(tree.position.y),
        float(tree.position.z),
    )
    height = tree.height
    crown_radius = tree.crown_radius
    rotation = rng.uniform(0.0, _TAU)
    trunk_top_ratio = (0.69, 0.66, 0.72, 0.68)[tree.variant]
    trunk_top = base_y + height * trunk_top_ratio
    trunk_sides = 6 if tree.variant == 2 else 7

    _add_tapered_segment(
        rows,
        (base_x, base_y, base_z),
        (base_x, trunk_top, base_z),
        tree.trunk_radius,
        tree.trunk_radius * (0.52 if tree.variant != 2 else 0.43),
        trunk_sides,
        spec.bark_color,
        rng,
        phase=rotation,
    )

    clusters = _BROADLEAF_CLUSTERS[tree.variant]
    resolved_clusters: list[tuple[_Point, float, float, float]] = []
    for ox_ratio, oz_ratio, cy_ratio, rx_ratio, ry_ratio, rz_ratio in clusters:
        offset_x, offset_z = _rotated_offset(
            ox_ratio * crown_radius,
            oz_ratio * crown_radius,
            rotation,
        )
        resolved_clusters.append(
            (
                (
                    base_x + offset_x,
                    base_y + cy_ratio * height,
                    base_z + offset_z,
                ),
                rx_ratio * crown_radius,
                ry_ratio * height,
                rz_ratio * crown_radius,
            )
        )

    # One structural branch supports each crown clump.  The primary branch is
    # given a small sideways target when its crown is centered so it remains a
    # visibly three-dimensional fork instead of overlapping the trunk.
    for index, (center, _rx, radius_y, _rz) in enumerate(resolved_clusters):
        direction_x = center[0] - base_x
        direction_z = center[2] - base_z
        if direction_x * direction_x + direction_z * direction_z < 0.01:
            branch_angle = rotation + index * 2.17 + 0.65
            direction_x = math.cos(branch_angle) * crown_radius * 0.22
            direction_z = math.sin(branch_angle) * crown_radius * 0.22
        start_y = base_y + height * (0.43 + 0.055 * index)
        end = (
            base_x + direction_x * 0.76,
            center[1] - radius_y * 0.28,
            base_z + direction_z * 0.76,
        )
        direction_length = math.hypot(direction_x, direction_z)
        if direction_length > _EPSILON:
            start_offset = tree.trunk_radius * 0.20 / direction_length
            start = (
                base_x + direction_x * start_offset,
                start_y,
                base_z + direction_z * start_offset,
            )
        else:
            start = (base_x, start_y, base_z)
        _add_tapered_segment(
            rows,
            start,
            end,
            tree.trunk_radius * 0.48,
            tree.trunk_radius * 0.16,
            5,
            spec.bark_color,
            rng,
            phase=rotation + index * 0.91,
        )

    crown_sides = 7 if tree.variant == 2 else 6
    for index, (center, radius_x, radius_y, radius_z) in enumerate(
        resolved_clusters
    ):
        cluster_color = _tint(spec.foliage_color, rng, 0.055)
        _add_faceted_crown(
            rows,
            center,
            radius_x,
            radius_y,
            radius_z,
            crown_sides,
            cluster_color,
            rng,
            rotation=rotation + index * 0.67 + rng.uniform(-0.18, 0.18),
        )


def _build_broadleaf_medium_geometry(
    tree: Tree, rows: list[_VertexRow], rng: random.Random
) -> None:
    """Build clumped broadleaf silhouettes at roughly half the LOD0 cost."""

    spec = _VARIANTS[tree.variant]
    base_x, base_y, base_z = (
        float(tree.position.x),
        float(tree.position.y),
        float(tree.position.z),
    )
    height = tree.height
    crown_radius = tree.crown_radius
    rotation = rng.uniform(0.0, _TAU)
    trunk_top_ratio = (0.69, 0.66, 0.72, 0.68)[tree.variant]
    trunk_top = base_y + height * trunk_top_ratio

    _add_tapered_segment(
        rows,
        (base_x, base_y, base_z),
        (base_x, trunk_top, base_z),
        tree.trunk_radius,
        tree.trunk_radius * (0.52 if tree.variant != 2 else 0.43),
        4 if tree.variant == 2 else 5,
        spec.bark_color,
        rng,
        phase=rotation,
    )

    resolved_clusters: list[tuple[_Point, float, float, float]] = []
    for ox_ratio, oz_ratio, cy_ratio, rx_ratio, ry_ratio, rz_ratio in (
        _BROADLEAF_CLUSTERS[tree.variant]
    ):
        offset_x, offset_z = _rotated_offset(
            ox_ratio * crown_radius,
            oz_ratio * crown_radius,
            rotation,
        )
        resolved_clusters.append(
            (
                (
                    base_x + offset_x,
                    base_y + cy_ratio * height,
                    base_z + offset_z,
                ),
                rx_ratio * crown_radius,
                ry_ratio * height,
                rz_ratio * crown_radius,
            )
        )

    # Three-sided branches keep the visible forks and authored clump layout,
    # but drop four side triangles per branch compared with the near model.
    for index, (center, _rx, radius_y, _rz) in enumerate(resolved_clusters):
        direction_x = center[0] - base_x
        direction_z = center[2] - base_z
        if direction_x * direction_x + direction_z * direction_z < 0.01:
            branch_angle = rotation + index * 2.17 + 0.65
            direction_x = math.cos(branch_angle) * crown_radius * 0.22
            direction_z = math.sin(branch_angle) * crown_radius * 0.22
        start_y = base_y + height * (0.43 + 0.055 * index)
        end = (
            base_x + direction_x * 0.76,
            center[1] - radius_y * 0.28,
            base_z + direction_z * 0.76,
        )
        direction_length = math.hypot(direction_x, direction_z)
        if direction_length > _EPSILON:
            start_offset = tree.trunk_radius * 0.20 / direction_length
            start = (
                base_x + direction_x * start_offset,
                start_y,
                base_z + direction_z * start_offset,
            )
        else:
            start = (base_x, start_y, base_z)
        _add_tapered_segment(
            rows,
            start,
            end,
            tree.trunk_radius * 0.48,
            tree.trunk_radius * 0.16,
            3,
            spec.bark_color,
            rng,
            phase=rotation + index * 0.91,
        )

    for index, (center, radius_x, radius_y, radius_z) in enumerate(
        resolved_clusters
    ):
        _add_bipyramid_crown(
            rows,
            center,
            radius_x,
            radius_y,
            radius_z,
            5,
            _tint(spec.foliage_color, rng, 0.055),
            rng,
            rotation=rotation + index * 0.67 + rng.uniform(-0.18, 0.18),
        )


def _build_broadleaf_low_geometry(
    tree: Tree, rows: list[_VertexRow], rng: random.Random
) -> None:
    """Build a trunk and one family-shaped crown for distant broadleaves."""

    spec = _VARIANTS[tree.variant]
    base_x, base_y, base_z = (
        float(tree.position.x),
        float(tree.position.y),
        float(tree.position.z),
    )
    height = tree.height
    rotation = rng.uniform(0.0, _TAU)
    trunk_top_ratio = (0.69, 0.66, 0.72, 0.68)[tree.variant]
    _add_tapered_segment(
        rows,
        (base_x, base_y, base_z),
        (base_x, base_y + height * trunk_top_ratio, base_z),
        tree.trunk_radius,
        tree.trunk_radius * (0.52 if tree.variant != 2 else 0.43),
        3,
        spec.bark_color,
        rng,
        phase=rotation,
    )

    cy_ratio, rx_ratio, ry_ratio, rz_ratio = _BROADLEAF_LOW_CROWNS[
        tree.variant
    ]
    _add_bipyramid_crown(
        rows,
        (base_x, base_y + cy_ratio * height, base_z),
        rx_ratio * tree.crown_radius,
        ry_ratio * height,
        rz_ratio * tree.crown_radius,
        5,
        _tint(spec.foliage_color, rng, 0.045),
        rng,
        rotation=rotation,
    )


def _build_conifer_geometry(
    tree: Tree, rows: list[_VertexRow], rng: random.Random
) -> None:
    spec = _VARIANTS[tree.variant]
    base_x, base_y, base_z = (
        float(tree.position.x),
        float(tree.position.y),
        float(tree.position.z),
    )
    height = tree.height
    crown_radius = tree.crown_radius
    phase = rng.uniform(0.0, _TAU)

    _add_tapered_segment(
        rows,
        (base_x, base_y, base_z),
        (base_x, base_y + height * 0.94, base_z),
        tree.trunk_radius,
        tree.trunk_radius * 0.30,
        6,
        spec.bark_color,
        rng,
        phase=phase,
    )

    if tree.variant == 4:
        layers = (
            (0.26, 0.70, 1.00),
            (0.45, 0.86, 0.79),
            (0.63, 1.00, 0.56),
        )
        cone_sides = 7
        branch_count = 6
    elif tree.variant == 5:
        layers = (
            (0.24, 0.64, 1.00),
            (0.38, 0.76, 0.87),
            (0.52, 0.88, 0.70),
            (0.66, 1.00, 0.52),
        )
        cone_sides = 6
        branch_count = 6
    else:
        layers = (
            (0.32, 0.76, 1.00),
            (0.50, 0.90, 0.82),
            (0.66, 1.00, 0.60),
        )
        cone_sides = 6
        branch_count = 4

    # Sparse radial limbs keep the silhouette dimensional when viewed between
    # cone tiers, without spending foliage-level polygon counts on every twig.
    for index in range(branch_count):
        angle = phase + _TAU * index / branch_count + rng.uniform(-0.11, 0.11)
        radial_scale = rng.uniform(0.68, 0.88)
        branch_y = base_y + height * (
            0.31 + 0.22 * (index % 2) + rng.uniform(-0.025, 0.025)
        )
        end = (
            base_x + math.cos(angle) * crown_radius * radial_scale,
            branch_y + height * rng.uniform(0.035, 0.075),
            base_z + math.sin(angle) * crown_radius * radial_scale,
        )
        start = (
            base_x + math.cos(angle) * tree.trunk_radius * 0.18,
            branch_y,
            base_z + math.sin(angle) * tree.trunk_radius * 0.18,
        )
        _add_tapered_segment(
            rows,
            start,
            end,
            tree.trunk_radius * 0.42,
            tree.trunk_radius * 0.10,
            3,
            spec.bark_color,
            rng,
            phase=angle,
        )

    for index, (base_ratio, apex_ratio, radius_ratio) in enumerate(layers):
        _add_foliage_cone(
            rows,
            base_x,
            base_z,
            base_y + height * base_ratio,
            base_y + height * apex_ratio,
            crown_radius * radius_ratio,
            cone_sides,
            _tint(spec.foliage_color, rng, 0.045),
            rng,
            phase=phase + index * 0.29,
        )


def _build_conifer_medium_geometry(
    tree: Tree, rows: list[_VertexRow], rng: random.Random
) -> None:
    """Build tiered conifers without the near model's radial branch mesh."""

    spec = _VARIANTS[tree.variant]
    base_x, base_y, base_z = (
        float(tree.position.x),
        float(tree.position.y),
        float(tree.position.z),
    )
    height = tree.height
    phase = rng.uniform(0.0, _TAU)
    _add_tapered_segment(
        rows,
        (base_x, base_y, base_z),
        (base_x, base_y + height * 0.94, base_z),
        tree.trunk_radius,
        tree.trunk_radius * 0.30,
        4,
        spec.bark_color,
        rng,
        phase=phase,
    )

    if tree.variant == 4:
        layers = (
            (0.26, 0.70, 1.00),
            (0.45, 0.86, 0.79),
            (0.63, 1.00, 0.56),
        )
    elif tree.variant == 5:
        layers = (
            (0.24, 0.64, 1.00),
            (0.38, 0.76, 0.87),
            (0.52, 0.88, 0.70),
            (0.66, 1.00, 0.52),
        )
    else:
        layers = (
            (0.32, 0.76, 1.00),
            (0.50, 0.90, 0.82),
            (0.66, 1.00, 0.60),
        )

    for index, (base_ratio, apex_ratio, radius_ratio) in enumerate(layers):
        _add_foliage_cone(
            rows,
            base_x,
            base_z,
            base_y + height * base_ratio,
            base_y + height * apex_ratio,
            tree.crown_radius * radius_ratio,
            4,
            _tint(spec.foliage_color, rng, 0.04),
            rng,
            phase=phase + index * 0.29,
        )


def _build_conifer_low_geometry(
    tree: Tree, rows: list[_VertexRow], rng: random.Random
) -> None:
    """Build a tiny trunk and one or two opaque cones for distant trees."""

    spec = _VARIANTS[tree.variant]
    base_x, base_y, base_z = (
        float(tree.position.x),
        float(tree.position.y),
        float(tree.position.z),
    )
    height = tree.height
    phase = rng.uniform(0.0, _TAU)
    _add_tapered_segment(
        rows,
        (base_x, base_y, base_z),
        (base_x, base_y + height * 0.94, base_z),
        tree.trunk_radius,
        tree.trunk_radius * 0.30,
        3,
        spec.bark_color,
        rng,
        phase=phase,
    )

    if tree.variant == 4:
        layers = ((0.26, 0.78, 1.00), (0.50, 1.00, 0.66))
    elif tree.variant == 5:
        layers = ((0.24, 0.76, 1.00), (0.50, 1.00, 0.70))
    else:
        layers = ((0.31, 1.00, 0.92),)

    for index, (base_ratio, apex_ratio, radius_ratio) in enumerate(layers):
        _add_foliage_cone(
            rows,
            base_x,
            base_z,
            base_y + height * base_ratio,
            base_y + height * apex_ratio,
            tree.crown_radius * radius_ratio,
            4,
            _tint(spec.foliage_color, rng, 0.035),
            rng,
            phase=phase + index * 0.29,
        )


def build_tree_vertex_data(
    tree: Tree,
    lod_level: TreeLOD | int = TreeLOD.HIGH,
) -> np.ndarray:
    """Return deterministic triangle rows in ``xyz rgb normal uv`` layout.

    The function performs CPU work only and has no global random or OpenGL
    side effects.  Each face has a geometric unit normal, giving the clustered
    crowns and tapered wood their deliberately faceted low-poly shading.
    """

    if not isinstance(tree, Tree):
        raise TypeError("build_tree_vertex_data expects a Tree instance")
    lod = _validated_lod(lod_level)

    rows: list[_VertexRow] = []
    rng = _seeded_rng(tree.seed, 0x47454F4D + tree.variant)
    if _VARIANTS[tree.variant].conifer:
        if lod is TreeLOD.HIGH:
            _build_conifer_geometry(tree, rows, rng)
        elif lod is TreeLOD.MEDIUM:
            _build_conifer_medium_geometry(tree, rows, rng)
        else:
            _build_conifer_low_geometry(tree, rows, rng)
    else:
        if lod is TreeLOD.HIGH:
            _build_broadleaf_geometry(tree, rows, rng)
        elif lod is TreeLOD.MEDIUM:
            _build_broadleaf_medium_geometry(tree, rows, rng)
        else:
            _build_broadleaf_low_geometry(tree, rows, rng)

    vertex_data = np.asarray(rows, dtype=np.float32)
    if vertex_data.size == 0:
        return np.empty((0, 11), dtype=np.float32)
    vertex_data = np.ascontiguousarray(vertex_data.reshape((-1, 11)))
    if not np.isfinite(vertex_data).all():
        raise ValueError("procedural tree generation produced non-finite vertices")
    return vertex_data


def _texture_id(texture: object) -> int:
    return int(getattr(texture, "texture", texture))


def _validated_batch_settings(
    chunk_size: float,
    exposure_baseline: float,
) -> tuple[float, float]:
    chunk_extent = float(chunk_size)
    if not math.isfinite(chunk_extent) or chunk_extent <= 0.0:
        raise ValueError("tree chunk_size must be a positive finite number")
    baseline = float(exposure_baseline)
    if not math.isfinite(baseline):
        raise ValueError("tree exposure_baseline must be finite")
    return chunk_extent, baseline


def _collect_tree_chunks(
    trees: Iterable[Tree],
    chunk_extent: float,
    *,
    caller: str,
) -> dict[TreeChunkKey, list[Tree]]:
    chunks: dict[TreeChunkKey, list[Tree]] = defaultdict(list)
    for tree in trees or ():
        if tree is None:
            continue
        if not isinstance(tree, Tree):
            raise TypeError(f"{caller} expects only Tree instances")
        chunk_key = (
            math.floor(float(tree.position.x) / chunk_extent),
            math.floor(float(tree.position.z) / chunk_extent),
        )
        chunks[chunk_key].append(tree)
    return chunks


def _upload_tree_chunk(
    trees: Sequence[Tree],
    lod_level: TreeLOD,
    *,
    texture: int,
    exposure_baseline: float,
) -> BatchedMesh | None:
    parts = [
        vertex_data
        for tree in trees
        if (vertex_data := build_tree_vertex_data(tree, lod_level)).size
    ]
    if not parts:
        return None
    combined = np.ascontiguousarray(
        parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0),
        dtype=np.float32,
    )
    return BatchedMesh.from_vertex_data(
        combined,
        texture=texture,
        alpha_test=False,
        casts_shadows=True,
        casts_sun_shadows=True,
        casts_point_shadows=False,
        exposure_baseline=exposure_baseline,
        keep_vertex_data=False,
        environment_lighting=True,
        shine_enabled=False,
        shader_lighting=True,
        lighting_receiver=TREE_LIGHTING_RECEIVER,
    )


def build_tree_meshes(
    trees: Iterable[Tree],
    *,
    material_texture: object | None = None,
    chunk_size: float = 1000.0,
    exposure_baseline: float = 1.0,
) -> list[BatchedMesh]:
    """Upload opaque tree geometry in deterministic spatial XZ chunks.

    A neutral one-pixel texture lets per-vertex bark and foliage colors use
    the packet renderer's normal-aware textured material path.  The texture
    helper caches that material, while each returned mesh owns only its VBO.
    """

    chunk_extent, baseline = _validated_batch_settings(
        chunk_size,
        exposure_baseline,
    )
    chunks = _collect_tree_chunks(
        trees,
        chunk_extent,
        caller="build_tree_meshes",
    )

    # Avoid both texture creation and empty VBO allocation for an empty forest.
    if not chunks:
        return []

    texture = (
        create_solid_color_texture((255, 255, 255, 255))
        if material_texture is None
        else material_texture
    )
    texture_value = _texture_id(texture)

    meshes: list[BatchedMesh] = []
    for chunk_key in sorted(chunks):
        mesh = _upload_tree_chunk(
            chunks[chunk_key],
            TreeLOD.HIGH,
            texture=texture_value,
            exposure_baseline=baseline,
        )
        if mesh is not None:
            meshes.append(mesh)
    return meshes


def build_tree_lod_meshes(
    trees: Iterable[Tree],
    *,
    material_texture: object | None = None,
    chunk_size: float = 250.0,
    exposure_baseline: float = 1.0,
) -> list[TreeLODMeshGroup]:
    """Upload three aligned LOD meshes for every sorted spatial XZ chunk.

    Each returned tuple is ordered ``(HIGH, MEDIUM, LOW)``.  Grouping all
    levels around the same semantic tree list guarantees stable chunk
    alignment, so a runtime LOD selector can choose exactly one mesh without
    matching three independently-built lists.  ``None`` is reserved for an
    empty geometry level; the built-in tree generators currently populate all
    three levels.
    """

    chunk_extent, baseline = _validated_batch_settings(
        chunk_size,
        exposure_baseline,
    )
    chunks = _collect_tree_chunks(
        trees,
        chunk_extent,
        caller="build_tree_lod_meshes",
    )
    if not chunks:
        return []

    material = (
        create_solid_color_texture((255, 255, 255, 255))
        if material_texture is None
        else material_texture
    )
    texture_value = _texture_id(material)
    groups: list[TreeLODMeshGroup] = []
    for chunk_key in sorted(chunks):
        chunk_trees = chunks[chunk_key]
        groups.append(
            (
                _upload_tree_chunk(
                    chunk_trees,
                    TreeLOD.HIGH,
                    texture=texture_value,
                    exposure_baseline=baseline,
                ),
                _upload_tree_chunk(
                    chunk_trees,
                    TreeLOD.MEDIUM,
                    texture=texture_value,
                    exposure_baseline=baseline,
                ),
                _upload_tree_chunk(
                    chunk_trees,
                    TreeLOD.LOW,
                    texture=texture_value,
                    exposure_baseline=baseline,
                ),
            )
        )
    return groups


__all__ = [
    "TREE_LOD_COUNT",
    "TREE_VARIANT_COUNT",
    "Tree",
    "TreeChunkKey",
    "TreeLOD",
    "TreeLODMeshGroup",
    "build_tree_vertex_data",
    "build_tree_lod_meshes",
    "build_tree_meshes",
]
