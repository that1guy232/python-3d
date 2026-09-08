"""Deterministic low-poly grass tufts and spatial render batches.

``Grass`` is a lightweight CPU-side semantic object.  Its position is the
point where the clump meets the terrain, and all generated vertices are in
world space.  Geometry generation remains separate from rendering so large
fields can retain useful placement metadata while sharing a small number of
VBOs.

The blades are folded, tapered surfaces with independently seeded bends and
leans.  They are genuine world-oriented geometry rather than camera-facing
cross billboards.  The engine renders meshes without back-face culling, so a
single thin surface is sufficient for each opaque leaf.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
import math
import random
from typing import NamedTuple

import numpy as np
from pygame.math import Vector3

from engine.rendering.mesh import BatchedMesh
from engine.textures.texture_utils import create_solid_color_texture
from game.gameplay.rendering.receivers import GRASS_LIGHTING_RECEIVER


GRASS_VARIANT_COUNT = 4


class GrassLOD(IntEnum):
    """Stable grass detail levels, ordered from most to least detailed."""

    HIGH = 0
    MEDIUM = 1
    LOW = 2


GRASS_LOD_COUNT = len(GrassLOD)

_TAU = math.tau
_EPSILON = 1.0e-10
_UINT64_MASK = (1 << 64) - 1


class _VariantSpec(NamedTuple):
    """Stable proportions and palette for one grass family."""

    name: str
    min_height: float
    max_height: float
    radius_ratio: float
    blade_count: int
    width_ratio: float
    base_color: tuple[float, float, float]
    tip_color: tuple[float, float, float]


# Heights retain the visual scale of the former grass sprites.  Radius and
# blade-width values are proportions, so an explicit height override keeps a
# family's recognizable silhouette.
_VARIANTS: tuple[_VariantSpec, ...] = (
    _VariantSpec(
        "meadow",
        19.0,
        27.0,
        0.34,
        4,
        0.145,
        (0.18, 0.39, 0.10),
        (0.36, 0.58, 0.16),
    ),
    _VariantSpec(
        "reed",
        23.0,
        32.0,
        0.29,
        4,
        0.125,
        (0.14, 0.34, 0.12),
        (0.29, 0.53, 0.20),
    ),
    _VariantSpec(
        "broadleaf",
        17.0,
        25.0,
        0.40,
        5,
        0.175,
        (0.20, 0.37, 0.08),
        (0.43, 0.57, 0.13),
    ),
    _VariantSpec(
        "dry",
        20.0,
        29.0,
        0.37,
        5,
        0.135,
        (0.38, 0.34, 0.12),
        (0.59, 0.52, 0.20),
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


def _coerce_position(position: Vector3 | Sequence[float]) -> Vector3:
    """Copy a two- or three-component position, mapping ``(x, z)`` to XZ."""

    if isinstance(position, Vector3):
        result = Vector3(position)
    else:
        try:
            values = tuple(position)
        except TypeError as exc:
            raise TypeError(
                "grass position must be a Vector3 or a 2/3 item sequence"
            ) from exc
        if len(values) == 2:
            result = Vector3(float(values[0]), 0.0, float(values[1]))
        elif len(values) == 3:
            result = Vector3(float(values[0]), float(values[1]), float(values[2]))
        else:
            raise ValueError("grass position must contain two or three coordinates")

    if not all(math.isfinite(float(component)) for component in result):
        raise ValueError("grass position coordinates must be finite")
    return result


def _validated_variant(variant: int) -> int:
    value = int(variant)
    if not 0 <= value < GRASS_VARIANT_COUNT:
        raise ValueError(
            f"grass variant must be between 0 and {GRASS_VARIANT_COUNT - 1}"
        )
    return value


def _validated_lod(lod_level: GrassLOD | int) -> GrassLOD:
    try:
        return GrassLOD(int(lod_level))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"grass lod_level must be between 0 and {GRASS_LOD_COUNT - 1}"
        ) from exc


def _height_for_seed(seed: int, variant: int) -> float:
    spec = _VARIANTS[variant]
    rng = _seeded_rng(seed, 0x48454947 + variant)
    return rng.uniform(spec.min_height, spec.max_height)


def _radius_for_seed(seed: int, variant: int, height: float) -> float:
    """Derive clump radius independently from geometry random choices."""

    spec = _VARIANTS[variant]
    rng = _seeded_rng(seed, 0x52414449 + variant)
    return max(0.25, height * spec.radius_ratio * rng.uniform(0.92, 1.08))


@dataclass(slots=True, init=False)
class Grass:
    """A deterministic grass clump anchored at terrain height.

    ``position.y`` is the base of every blade.  ``height`` is the full
    above-ground height and ``clump_radius`` conservatively encloses all blade
    vertices in XZ, making it suitable for both placement and broad-phase
    queries.
    """

    position: Vector3
    seed: int
    variant: int
    height: float
    clump_radius: float

    def __init__(
        self,
        position: Vector3 | Sequence[float],
        seed: int = 0,
        variant: int = 0,
        height: float | None = None,
        *,
        clump_radius: float | None = None,
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
            raise ValueError("grass height must be a positive finite number")

        radius_value = (
            _radius_for_seed(seed_value, variant_value, height_value)
            if clump_radius is None
            else float(clump_radius)
        )
        if not math.isfinite(radius_value) or radius_value <= 0.0:
            raise ValueError("grass clump radius must be a positive finite number")

        self.position = position_value
        self.seed = seed_value
        self.variant = variant_value
        self.height = height_value
        self.clump_radius = radius_value

    @classmethod
    def from_seed(
        cls,
        position: Vector3 | Sequence[float],
        *,
        seed: int,
        variant: int | None = None,
        height: float | None = None,
    ) -> "Grass":
        """Create repeatable variant and dimensions from ``seed``.

        Variant, dimensions, and blade geometry use independent random
        streams.  Rebuilding a prototype at its final terrain height therefore
        retains the same shape and placement radius.
        """

        seed_value = int(seed)
        variant_value = (
            _seeded_rng(seed_value, 0x56415249).randrange(GRASS_VARIANT_COUNT)
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
        """Return sprite-compatible ``(clump diameter, full height)``."""

        return self.clump_radius * 2.0, self.height

    @property
    def placement_radius(self) -> float:
        """Return the conservative XZ radius used to space detail objects."""

        return self.clump_radius

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
        """Return whether an XZ point lies inside the circular clump footprint."""

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


def _mix_color(a: _Color, b: _Color, amount: float) -> _Color:
    inverse = 1.0 - amount
    return (
        a[0] * inverse + b[0] * amount,
        a[1] * inverse + b[1] * amount,
        a[2] * inverse + b[2] * amount,
    )


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
    """Append one flat-shaded non-degenerate triangle."""

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
    red, green, blue = color
    for point, uv in zip((a, b, c), uvs):
        rows.append(
            (
                point[0],
                point[1],
                point[2],
                red,
                green,
                blue,
                nx,
                ny,
                nz,
                uv[0],
                uv[1],
            )
        )


def _offset(
    point: _Point,
    side_x: float,
    side_z: float,
    forward_x: float,
    forward_z: float,
    across: float,
    fold: float,
) -> _Point:
    return (
        point[0] + side_x * across + forward_x * fold,
        point[1],
        point[2] + side_z * across + forward_z * fold,
    )


def _add_folded_blade(
    rows: list[_VertexRow],
    root: _Point,
    blade_height: float,
    half_width: float,
    yaw: float,
    lean: float,
    curve: float,
    base_color: _Color,
    tip_color: _Color,
    rng: random.Random,
) -> None:
    """Add a bent, tapered V-fold leaf using six triangles."""

    forward_x = math.cos(yaw)
    forward_z = math.sin(yaw)
    side_x = -forward_z
    side_z = forward_x

    middle = (
        root[0] + forward_x * lean * 0.36 + side_x * curve,
        root[1] + blade_height * 0.54,
        root[2] + forward_z * lean * 0.36 + side_z * curve,
    )
    tip = (
        root[0] + forward_x * lean + side_x * curve * 0.35,
        root[1] + blade_height,
        root[2] + forward_z * lean + side_z * curve * 0.35,
    )

    base_fold = half_width * 0.22
    middle_width = half_width * rng.uniform(0.52, 0.64)
    middle_fold = middle_width * 0.18
    base_left = _offset(
        root, side_x, side_z, forward_x, forward_z, -half_width, 0.0
    )
    base_ridge = _offset(
        root, side_x, side_z, forward_x, forward_z, 0.0, base_fold
    )
    base_right = _offset(
        root, side_x, side_z, forward_x, forward_z, half_width, 0.0
    )
    middle_left = _offset(
        middle, side_x, side_z, forward_x, forward_z, -middle_width, 0.0
    )
    middle_ridge = _offset(
        middle, side_x, side_z, forward_x, forward_z, 0.0, middle_fold
    )
    middle_right = _offset(
        middle, side_x, side_z, forward_x, forward_z, middle_width, 0.0
    )

    lower_left_color = _tint(_mix_color(base_color, tip_color, 0.28), rng, 0.07)
    lower_right_color = _tint(
        _mix_color(base_color, tip_color, 0.34), rng, 0.07
    )
    upper_left_color = _tint(
        _mix_color(base_color, tip_color, 0.73), rng, 0.065
    )
    upper_right_color = _tint(
        _mix_color(base_color, tip_color, 0.82), rng, 0.065
    )

    # Two lower folded quads and two tapered upper triangles.  Back-face
    # culling is disabled globally, so duplicating coplanar reverse windings
    # would only consume vertex bandwidth and disturb their lighting normals.
    _append_triangle(
        rows,
        base_left,
        base_ridge,
        middle_ridge,
        lower_left_color,
        ((0.0, 0.0), (0.5, 0.0), (0.5, 0.54)),
    )
    _append_triangle(
        rows,
        base_left,
        middle_ridge,
        middle_left,
        lower_left_color,
        ((0.0, 0.0), (0.5, 0.54), (0.0, 0.54)),
    )
    _append_triangle(
        rows,
        base_ridge,
        base_right,
        middle_right,
        lower_right_color,
        ((0.5, 0.0), (1.0, 0.0), (1.0, 0.54)),
    )
    _append_triangle(
        rows,
        base_ridge,
        middle_right,
        middle_ridge,
        lower_right_color,
        ((0.5, 0.0), (1.0, 0.54), (0.5, 0.54)),
    )
    _append_triangle(
        rows,
        middle_left,
        middle_ridge,
        tip,
        upper_left_color,
        ((0.0, 0.54), (0.5, 0.54), (0.5, 1.0)),
    )
    _append_triangle(
        rows,
        middle_ridge,
        middle_right,
        tip,
        upper_right_color,
        ((0.5, 0.54), (1.0, 0.54), (0.5, 1.0)),
    )


def _add_flat_blade(
    rows: list[_VertexRow],
    root: _Point,
    blade_height: float,
    half_width: float,
    yaw: float,
    lean: float,
    base_color: _Color,
    tip_color: _Color,
    rng: random.Random,
) -> None:
    """Add an inexpensive tapered leaf using two non-degenerate triangles."""

    forward_x = math.cos(yaw)
    forward_z = math.sin(yaw)
    side_x = -forward_z
    side_z = forward_x
    tip_center = (
        root[0] + forward_x * lean,
        root[1] + blade_height,
        root[2] + forward_z * lean,
    )
    tip_half_width = half_width * 0.13
    base_left = (
        root[0] - side_x * half_width,
        root[1],
        root[2] - side_z * half_width,
    )
    base_right = (
        root[0] + side_x * half_width,
        root[1],
        root[2] + side_z * half_width,
    )
    tip_left = (
        tip_center[0] - side_x * tip_half_width,
        tip_center[1],
        tip_center[2] - side_z * tip_half_width,
    )
    tip_right = (
        tip_center[0] + side_x * tip_half_width,
        tip_center[1],
        tip_center[2] + side_z * tip_half_width,
    )
    lower_color = _tint(_mix_color(base_color, tip_color, 0.34), rng, 0.055)
    upper_color = _tint(_mix_color(base_color, tip_color, 0.74), rng, 0.05)
    _append_triangle(
        rows,
        base_left,
        base_right,
        tip_right,
        lower_color,
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
    )
    _append_triangle(
        rows,
        base_left,
        tip_right,
        tip_left,
        upper_color,
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    )


def _add_silhouette_blade(
    rows: list[_VertexRow],
    root: _Point,
    blade_height: float,
    half_width: float,
    yaw: float,
    lean: float,
    color: _Color,
) -> None:
    """Add one broad triangle for the distant crossed-blade silhouette."""

    forward_x = math.cos(yaw)
    forward_z = math.sin(yaw)
    side_x = -forward_z
    side_z = forward_x
    left = (
        root[0] - side_x * half_width,
        root[1],
        root[2] - side_z * half_width,
    )
    right = (
        root[0] + side_x * half_width,
        root[1],
        root[2] + side_z * half_width,
    )
    tip = (
        root[0] + forward_x * lean,
        root[1] + blade_height,
        root[2] + forward_z * lean,
    )
    _append_triangle(
        rows,
        left,
        right,
        tip,
        color,
        ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
    )


def build_grass_vertex_data(
    grass: Grass,
    lod_level: GrassLOD | int = GrassLOD.HIGH,
) -> np.ndarray:
    """Return deterministic GL_TRIANGLES rows in ``xyz rgb normal uv`` layout.

    This is a pure CPU operation: it neither touches OpenGL nor consumes the
    process-global random stream.  High detail uses four or five folded
    blades (24/30 triangles), medium uses two or three tapered surfaces (4/6
    triangles), and low uses two crossed silhouette triangles.
    """

    if not isinstance(grass, Grass):
        raise TypeError("build_grass_vertex_data expects a Grass instance")

    lod = _validated_lod(lod_level)
    spec = _VARIANTS[grass.variant]
    rng = _seeded_rng(
        grass.seed,
        0x47454F4D + grass.variant + int(lod) * 0x101,
    )
    rows: list[_VertexRow] = []
    base_x = float(grass.position.x)
    base_y = float(grass.position.y)
    base_z = float(grass.position.z)
    phase = rng.uniform(0.0, _TAU)

    if lod is GrassLOD.LOW:
        # Two perpendicular, double-sided-by-render-state triangles remain
        # legible from every horizontal view direction at a tiny vertex cost.
        for index, height_scale in enumerate((1.0, 0.82)):
            _add_silhouette_blade(
                rows,
                (base_x, base_y, base_z),
                grass.height * height_scale,
                grass.clump_radius * (0.25 if index == 0 else 0.21),
                phase + index * math.pi * 0.5,
                grass.clump_radius * (0.16 if index == 0 else -0.12),
                _tint(
                    _mix_color(spec.base_color, spec.tip_color, 0.58),
                    rng,
                    0.04,
                ),
            )
        return np.ascontiguousarray(np.asarray(rows, dtype=np.float32).reshape(-1, 11))

    blade_count = (
        spec.blade_count
        if lod is GrassLOD.HIGH
        else max(2, (spec.blade_count + 1) // 2)
    )
    height_scales = [1.0]
    height_scales.extend(
        rng.uniform(0.68, 0.94) for _ in range(blade_count - 1)
    )
    rng.shuffle(height_scales)

    for index, height_scale in enumerate(height_scales):
        angle = (
            phase
            + _TAU * index / blade_count
            + rng.uniform(-0.24, 0.24)
        )
        if index == 0:
            root_distance = rng.uniform(0.02, 0.10) * grass.clump_radius
        else:
            root_distance = rng.uniform(0.12, 0.34) * grass.clump_radius
        root_angle = angle + rng.uniform(-0.55, 0.55)
        root = (
            base_x + math.cos(root_angle) * root_distance,
            base_y,
            base_z + math.sin(root_angle) * root_distance,
        )
        blade_height = grass.height * height_scale
        half_width = (
            grass.clump_radius
            * spec.width_ratio
            * rng.uniform(0.82, 1.16)
        )
        lean = grass.clump_radius * rng.uniform(0.20, 0.40)
        curve = grass.clump_radius * rng.uniform(-0.055, 0.055)

        base_color = _tint(spec.base_color, rng, 0.045)
        tip_color = _tint(spec.tip_color, rng, 0.05)
        if lod is GrassLOD.HIGH:
            _add_folded_blade(
                rows,
                root,
                blade_height,
                half_width,
                angle,
                lean,
                curve,
                base_color,
                tip_color,
                rng,
            )
        else:
            _add_flat_blade(
                rows,
                root,
                blade_height,
                half_width * 1.08,
                angle,
                lean,
                base_color,
                tip_color,
                rng,
            )

    vertex_data = np.asarray(rows, dtype=np.float32)
    if vertex_data.size == 0:
        return np.empty((0, 11), dtype=np.float32)
    vertex_data = np.ascontiguousarray(vertex_data.reshape((-1, 11)))
    if not np.isfinite(vertex_data).all():
        raise ValueError("procedural grass generation produced non-finite vertices")
    return vertex_data


def _texture_id(texture: object) -> int:
    return int(getattr(texture, "texture", texture))


def _grass_mesh_from_vertex_data(
    vertex_data: np.ndarray,
    *,
    texture: int,
    exposure_baseline: float,
) -> BatchedMesh:
    return BatchedMesh.from_vertex_data(
        vertex_data,
        texture=texture,
        alpha_test=False,
        casts_shadows=False,
        casts_sun_shadows=False,
        casts_point_shadows=False,
        exposure_baseline=exposure_baseline,
        keep_vertex_data=False,
        environment_lighting=True,
        shine_enabled=False,
        shader_lighting=True,
        lighting_receiver=GRASS_LIGHTING_RECEIVER,
    )


def build_grass_meshes(
    grasses: Iterable[Grass],
    material_texture: object | None = None,
    chunk_size: float = 500.0,
    exposure_baseline: float = 1.0,
) -> list[BatchedMesh]:
    """Upload high-detail grass geometry in deterministic spatial XZ chunks.

    The default cached white texture exposes per-vertex grass colors through
    the normal-aware textured material path.  Empty inputs create neither a
    texture nor a VBO.  Use :func:`build_grass_lod_meshes` to upload all three
    runtime-selectable levels with matching chunk coverage.
    """

    chunk_extent = float(chunk_size)
    if not math.isfinite(chunk_extent) or chunk_extent <= 0.0:
        raise ValueError("grass chunk_size must be a positive finite number")
    baseline = float(exposure_baseline)
    if not math.isfinite(baseline):
        raise ValueError("grass exposure_baseline must be finite")

    chunks: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for grass in grasses or ():
        if grass is None:
            continue
        if not isinstance(grass, Grass):
            raise TypeError("build_grass_meshes expects only Grass instances")
        chunk_key = (
            math.floor(float(grass.position.x) / chunk_extent),
            math.floor(float(grass.position.z) / chunk_extent),
        )
        vertex_data = build_grass_vertex_data(grass, GrassLOD.HIGH)
        if vertex_data.size:
            chunks[chunk_key].append(vertex_data)

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
        parts = chunks[chunk_key]
        combined = np.ascontiguousarray(
            parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0),
            dtype=np.float32,
        )
        meshes.append(
            _grass_mesh_from_vertex_data(
                combined,
                texture=texture_value,
                exposure_baseline=baseline,
            )
        )
    return meshes


def build_grass_lod_meshes(
    grasses: Iterable[Grass],
    material_texture: object | None = None,
    chunk_size: float = 500.0,
    exposure_baseline: float = 1.0,
) -> list[tuple[BatchedMesh | None, BatchedMesh | None, BatchedMesh | None]]:
    """Upload HIGH/MEDIUM/LOW VBOs for every sorted spatial chunk.

    Tuple positions are stable and correspond to :class:`GrassLOD`.  A
    ``None`` slot represents intentionally empty geometry while preserving
    chunk alignment; all current levels are visible and therefore populated.
    """

    chunk_extent = float(chunk_size)
    if not math.isfinite(chunk_extent) or chunk_extent <= 0.0:
        raise ValueError("grass chunk_size must be a positive finite number")
    baseline = float(exposure_baseline)
    if not math.isfinite(baseline):
        raise ValueError("grass exposure_baseline must be finite")

    chunks: dict[tuple[int, int], list[Grass]] = defaultdict(list)
    source = () if grasses is None else grasses
    for grass in source:
        if grass is None:
            continue
        if not isinstance(grass, Grass):
            raise TypeError("build_grass_lod_meshes expects only Grass instances")
        chunk_key = (
            math.floor(float(grass.position.x) / chunk_extent),
            math.floor(float(grass.position.z) / chunk_extent),
        )
        chunks[chunk_key].append(grass)

    if not chunks:
        return []

    texture_object = (
        create_solid_color_texture((255, 255, 255, 255))
        if material_texture is None
        else material_texture
    )
    texture_value = _texture_id(texture_object)
    lod_chunks: list[
        tuple[BatchedMesh | None, BatchedMesh | None, BatchedMesh | None]
    ] = []
    for chunk_key in sorted(chunks):
        level_meshes: list[BatchedMesh | None] = []
        for lod in GrassLOD:
            parts = [
                build_grass_vertex_data(grass, lod)
                for grass in chunks[chunk_key]
            ]
            parts = [part for part in parts if part.size]
            if not parts:
                level_meshes.append(None)
                continue
            combined = np.ascontiguousarray(
                parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0),
                dtype=np.float32,
            )
            level_meshes.append(
                _grass_mesh_from_vertex_data(
                    combined,
                    texture=texture_value,
                    exposure_baseline=baseline,
                )
            )
        lod_chunks.append(
            (level_meshes[0], level_meshes[1], level_meshes[2])
        )
    return lod_chunks


__all__ = [
    "GRASS_LOD_COUNT",
    "GRASS_VARIANT_COUNT",
    "Grass",
    "GrassLOD",
    "build_grass_lod_meshes",
    "build_grass_vertex_data",
    "build_grass_meshes",
]
