"""Deterministic low-poly rocks and spatially chunked render batches.

``Rock`` is a lightweight CPU-side placement object.  Its position is the
point where the rock meets the terrain, and procedural vertices are emitted
directly in world space.  Rendering remains separate so hundreds of rocks can
share a small number of spatially culled VBOs without losing their semantic
bounds or repeatable procedural parameters.
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

from engine.core.mesh import BatchedMesh
from engine.textures.texture_utils import create_solid_color_texture
from game.world.lighting_receivers import ROCK_LIGHTING_RECEIVER


ROCK_VARIANT_COUNT = 6


class RockLOD(IntEnum):
    """Stable rock detail levels, ordered from most to least detailed."""

    HIGH = 0
    MEDIUM = 1
    LOW = 2


ROCK_LOD_COUNT = len(RockLOD)

_TAU = math.tau
_EPSILON = 1.0e-10
_UINT64_MASK = (1 << 64) - 1


class _VariantSpec(NamedTuple):
    """Stable proportions, topology, and palette for one rock family."""

    name: str
    min_height: float
    max_height: float
    radius_x_ratio: float
    radius_z_ratio: float
    sides: int
    # Each pair is (height fraction, horizontal radius fraction).  The top is
    # a separate point, allowing profiles of different complexity while every
    # family remains a closed, inexpensive polyhedron.
    profile: tuple[tuple[float, float], ...]
    color: tuple[float, float, float]
    color_variation: float


# The six families differ in aspect, ring profile, facet count, and mineral
# color.  Their closed meshes range from 32 to 60 triangles, keeping 750-rock
# worlds practical while avoiding billboard-like silhouettes from any angle.
_VARIANTS: tuple[_VariantSpec, ...] = (
    _VariantSpec(
        "fieldstone",
        14.0,
        25.0,
        0.69,
        0.57,
        7,
        ((0.00, 0.70), (0.25, 0.94), (0.61, 0.89), (0.84, 0.57)),
        (0.43, 0.44, 0.41),
        0.13,
    ),
    _VariantSpec(
        "granite",
        18.0,
        32.0,
        0.66,
        0.61,
        6,
        ((0.00, 0.73), (0.22, 0.92), (0.56, 0.95), (0.83, 0.66)),
        (0.48, 0.46, 0.44),
        0.16,
    ),
    _VariantSpec(
        "slate_spire",
        25.0,
        43.0,
        0.42,
        0.34,
        5,
        (
            (0.00, 0.67),
            (0.18, 0.81),
            (0.47, 0.95),
            (0.72, 0.71),
            (0.89, 0.39),
        ),
        (0.31, 0.34, 0.36),
        0.14,
    ),
    _VariantSpec(
        "sandstone_slab",
        10.0,
        19.0,
        0.96,
        0.73,
        8,
        ((0.00, 0.75), (0.29, 0.95), (0.72, 0.82)),
        (0.56, 0.43, 0.29),
        0.12,
    ),
    _VariantSpec(
        "basalt_block",
        20.0,
        35.0,
        0.61,
        0.58,
        4,
        ((0.00, 0.78), (0.27, 0.95), (0.63, 0.87), (0.86, 0.56)),
        (0.24, 0.26, 0.27),
        0.11,
    ),
    _VariantSpec(
        "limestone_crag",
        15.0,
        29.0,
        0.78,
        0.50,
        6,
        (
            (0.00, 0.69),
            (0.19, 0.88),
            (0.45, 0.95),
            (0.70, 0.77),
            (0.88, 0.43),
        ),
        (0.57, 0.55, 0.48),
        0.15,
    ),
)


def _seeded_rng(seed: int, channel: int) -> random.Random:
    """Return an independent deterministic stream for one rock concern."""

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
                "rock position must be a Vector3 or a 2/3 item sequence"
            ) from exc
        if len(values) == 2:
            result = Vector3(float(values[0]), 0.0, float(values[1]))
        elif len(values) == 3:
            result = Vector3(
                float(values[0]), float(values[1]), float(values[2])
            )
        else:
            raise ValueError("rock position must contain two or three coordinates")

    if not all(math.isfinite(float(component)) for component in result):
        raise ValueError("rock position coordinates must be finite")
    return result


def _validated_variant(variant: int) -> int:
    value = int(variant)
    if not 0 <= value < ROCK_VARIANT_COUNT:
        raise ValueError(
            f"rock variant must be between 0 and {ROCK_VARIANT_COUNT - 1}"
        )
    return value


def _validated_lod(lod_level: RockLOD | int) -> RockLOD:
    try:
        return RockLOD(int(lod_level))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"rock lod_level must be between 0 and {ROCK_LOD_COUNT - 1}"
        ) from exc


def _height_for_seed(seed: int, variant: int) -> float:
    spec = _VARIANTS[variant]
    rng = _seeded_rng(seed, 0x48454947 + variant)
    return rng.uniform(spec.min_height, spec.max_height)


def _radii_for_seed(seed: int, variant: int, height: float) -> tuple[float, float]:
    """Derive an irregular footprint independently from the height stream."""

    spec = _VARIANTS[variant]
    rng = _seeded_rng(seed, 0x52414449 + variant)
    radius_x = height * spec.radius_x_ratio * rng.uniform(0.90, 1.10)
    radius_z = height * spec.radius_z_ratio * rng.uniform(0.90, 1.10)

    # Alternating the long axis avoids lining up anisotropic rocks while
    # keeping their semantic bounds simple and exact in world X/Z.
    if rng.random() < 0.5:
        radius_x, radius_z = radius_z, radius_x
    minimum_radius = max(height * 0.05, 1.0e-4)
    return max(minimum_radius, radius_x), max(minimum_radius, radius_z)


@dataclass(slots=True, init=False)
class Rock:
    """A deterministic procedural boulder anchored at terrain height.

    ``position.y`` is the rock's lowest plane, and ``height`` is its full
    above-ground height.  ``radius_x`` and ``radius_z`` conservatively enclose
    the generated footprint in their respective world axes.
    """

    position: Vector3
    seed: int
    variant: int
    height: float
    radius_x: float
    radius_z: float

    def __init__(
        self,
        position: Vector3 | Sequence[float],
        seed: int = 0,
        variant: int = 0,
        height: float | None = None,
        *,
        radius_x: float | None = None,
        radius_z: float | None = None,
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
            raise ValueError("rock height must be a positive finite number")

        derived_x, derived_z = _radii_for_seed(
            seed_value, variant_value, height_value
        )
        radius_x_value = derived_x if radius_x is None else float(radius_x)
        radius_z_value = derived_z if radius_z is None else float(radius_z)
        if not math.isfinite(radius_x_value) or radius_x_value <= 0.0:
            raise ValueError("rock radius_x must be a positive finite number")
        if not math.isfinite(radius_z_value) or radius_z_value <= 0.0:
            raise ValueError("rock radius_z must be a positive finite number")

        self.position = position_value
        self.seed = seed_value
        self.variant = variant_value
        self.height = height_value
        self.radius_x = radius_x_value
        self.radius_z = radius_z_value

    @classmethod
    def from_seed(
        cls,
        position: Vector3 | Sequence[float],
        *,
        seed: int,
        variant: int | None = None,
        height: float | None = None,
    ) -> "Rock":
        """Create repeatable style and dimensions from ``seed``.

        Variant, dimensions, and facets use independent streams.  Rebuilding
        a placement candidate at its final terrain Y therefore preserves its
        exact footprint and shape.
        """

        seed_value = int(seed)
        variant_value = (
            _seeded_rng(seed_value, 0x56415249).randrange(ROCK_VARIANT_COUNT)
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
        """Return billboard-compatible ``(footprint diameter, height)``."""

        return self.placement_radius * 2.0, self.height

    @property
    def placement_radius(self) -> float:
        """Return a conservative circular radius for spatial placement."""

        return max(self.radius_x, self.radius_z)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return XZ extents as ``(min_x, max_x, min_z, max_z)``."""

        return (
            float(self.position.x) - self.radius_x,
            float(self.position.x) + self.radius_x,
            float(self.position.z) - self.radius_z,
            float(self.position.z) + self.radius_z,
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

    def get_bounding_sphere(
        self,
    ) -> tuple[tuple[float, float, float], float]:
        """Return a conservative sphere derived from the semantic 3D box."""

        center = (
            float(self.position.x),
            float(self.position.y) + self.height * 0.5,
            float(self.position.z),
        )
        radius = math.sqrt(
            self.radius_x * self.radius_x
            + (self.height * 0.5) ** 2
            + self.radius_z * self.radius_z
        )
        return center, radius

    def contains_point(self, x: float, z: float, margin: float = 0.0) -> bool:
        """Return whether an XZ point lies inside the elliptical footprint."""

        padding = max(0.0, float(margin))
        radius_x = self.radius_x + padding
        radius_z = self.radius_z + padding
        dx = (float(x) - float(self.position.x)) / radius_x
        dz = (float(z) - float(self.position.z)) / radius_z
        return dx * dx + dz * dz <= 1.0


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


def _tint(
    color: _Color,
    rng: random.Random,
    variation: float,
    brightness: float = 1.0,
) -> _Color:
    factor = brightness * (1.0 + rng.uniform(-variation, variation))
    return tuple(max(0.0, min(1.0, component * factor)) for component in color)  # type: ignore[return-value]


def _append_triangle(
    rows: list[_VertexRow],
    a: _Point,
    b: _Point,
    c: _Point,
    color: _Color,
    uvs: tuple[_UV, _UV, _UV],
) -> None:
    """Append one non-degenerate flat-shaded triangle."""

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


def _ring_uv(angle: float) -> _UV:
    return 0.5 + math.cos(angle) * 0.5, 0.5 + math.sin(angle) * 0.5


def _lod_topology(
    spec: _VariantSpec,
    lod: RockLOD,
) -> tuple[int, tuple[tuple[float, float], ...]]:
    """Resolve a progressively cheaper closed silhouette for one variant."""

    if lod is RockLOD.HIGH:
        return spec.sides, spec.profile

    widest = max(spec.profile, key=lambda ring: ring[1])
    if lod is RockLOD.MEDIUM:
        # A base and widest body ring retain the family's aspect while cutting
        # topology by at least half before the cap converges at the summit.
        return max(4, spec.sides - 2), (spec.profile[0], widest)

    # Four footprint vertices plus the bottom and top centers form a closed,
    # eight-triangle distant rock with an unmistakable 3D silhouette.
    return 4, ((0.0, widest[1]),)


def _build_polyhedron(
    rock: Rock,
    rows: list[_VertexRow],
    rng: random.Random,
    lod: RockLOD,
) -> None:
    spec = _VARIANTS[rock.variant]
    base_x = float(rock.position.x)
    base_y = float(rock.position.y)
    base_z = float(rock.position.z)
    sides, profile = _lod_topology(spec, lod)
    sector_angle = _TAU / sides
    phase = rng.uniform(0.0, _TAU)

    # A stable angular identity is shared between rings, while a small twist
    # at each level produces skewed facets rather than vertical prism walls.
    base_angles = [
        phase
        + sector_angle * index
        + rng.uniform(-sector_angle * 0.12, sector_angle * 0.12)
        for index in range(sides)
    ]
    rings: list[list[_Point]] = []
    ring_angles: list[list[float]] = []
    previous_height_ratio = 0.0

    for ring_index, (height_ratio, profile_radius) in enumerate(profile):
        if ring_index == 0:
            resolved_height_ratio = 0.0
            twist = 0.0
            offset_x_ratio = 0.0
            offset_z_ratio = 0.0
        else:
            resolved_height_ratio = max(
                previous_height_ratio + 0.06,
                min(0.96, height_ratio + rng.uniform(-0.018, 0.018)),
            )
            twist = rng.uniform(-sector_angle * 0.08, sector_angle * 0.08)
            offset_x_ratio = rng.uniform(-0.016, 0.016)
            offset_z_ratio = rng.uniform(-0.016, 0.016)
        previous_height_ratio = resolved_height_ratio

        points: list[_Point] = []
        angles: list[float] = []
        for side_index, base_angle in enumerate(base_angles):
            angle = base_angle + twist
            # Side-correlated and ring-local perturbations retain a coherent
            # boulder silhouette without repeating identical horizontal rings.
            side_wave = math.sin(side_index * 2.399963 + rock.variant * 0.71)
            radial_jitter = rng.uniform(0.92, 1.014) + side_wave * 0.018
            radial_scale = profile_radius * radial_jitter
            normalized_x = offset_x_ratio + radial_scale * math.cos(angle)
            normalized_z = offset_z_ratio + radial_scale * math.sin(angle)
            # The semantic radii are placement and culling contracts, so keep
            # even worst-case perturbations strictly inside those envelopes.
            normalized_x = max(-0.999, min(0.999, normalized_x))
            normalized_z = max(-0.999, min(0.999, normalized_z))
            points.append(
                (
                    base_x + rock.radius_x * normalized_x,
                    base_y + rock.height * resolved_height_ratio,
                    base_z + rock.radius_z * normalized_z,
                )
            )
            angles.append(angle)
        rings.append(points)
        ring_angles.append(angles)

    bottom_center: _Point = (base_x, base_y, base_z)
    bottom_ring = rings[0]
    for index in range(sides):
        next_index = (index + 1) % sides
        _append_triangle(
            rows,
            bottom_center,
            bottom_ring[index],
            bottom_ring[next_index],
            _tint(spec.color, rng, spec.color_variation * 0.45, 0.67),
            (
                (0.5, 0.5),
                _ring_uv(ring_angles[0][index]),
                _ring_uv(ring_angles[0][next_index]),
            ),
        )

    for ring_index in range(len(rings) - 1):
        lower = rings[ring_index]
        upper = rings[ring_index + 1]
        lower_v = profile[ring_index][0]
        upper_v = profile[ring_index + 1][0]
        brightness = 0.82 + upper_v * 0.15
        for index in range(sides):
            next_index = (index + 1) % sides
            u0 = index / sides
            u1 = (index + 1) / sides
            face_color = _tint(
                spec.color, rng, spec.color_variation, brightness
            )
            # This winding points away from the rock for increasing ring angle.
            _append_triangle(
                rows,
                lower[index],
                upper[index],
                upper[next_index],
                face_color,
                ((u0, lower_v), (u0, upper_v), (u1, upper_v)),
            )
            _append_triangle(
                rows,
                lower[index],
                upper[next_index],
                lower[next_index],
                face_color,
                ((u0, lower_v), (u1, upper_v), (u1, lower_v)),
            )

    top_angle = phase + rng.uniform(-math.pi, math.pi)
    top: _Point = (
        base_x + math.cos(top_angle) * rock.radius_x * rng.uniform(0.02, 0.12),
        base_y + rock.height,
        base_z + math.sin(top_angle) * rock.radius_z * rng.uniform(0.02, 0.12),
    )
    top_ring = rings[-1]
    top_v = profile[-1][0]
    for index in range(sides):
        next_index = (index + 1) % sides
        _append_triangle(
            rows,
            top_ring[index],
            top,
            top_ring[next_index],
            _tint(spec.color, rng, spec.color_variation, 1.04),
            (
                (index / sides, top_v),
                (0.5, 1.0),
                ((index + 1) / sides, top_v),
            ),
        )


def build_rock_vertex_data(
    rock: Rock,
    lod_level: RockLOD | int = RockLOD.HIGH,
) -> np.ndarray:
    """Return deterministic ``GL_TRIANGLES`` rows in ``xyz rgb normal uv``.

    This is pure CPU generation: it neither consumes global randomness nor
    requires an OpenGL context.  Every triangle has a geometric unit normal so
    the deliberately irregular rings read as true flat-shaded stone facets.
    """

    if not isinstance(rock, Rock):
        raise TypeError("build_rock_vertex_data expects a Rock instance")

    lod = _validated_lod(lod_level)
    rows: list[_VertexRow] = []
    rng = _seeded_rng(
        rock.seed,
        0x47454F4D + rock.variant + int(lod) * 0x101,
    )
    _build_polyhedron(rock, rows, rng, lod)

    vertex_data = np.asarray(rows, dtype=np.float32)
    if vertex_data.size == 0:
        return np.empty((0, 11), dtype=np.float32)
    vertex_data = np.ascontiguousarray(vertex_data.reshape((-1, 11)))
    if not np.isfinite(vertex_data).all():
        raise ValueError("procedural rock generation produced non-finite vertices")
    return vertex_data


def _texture_id(texture: object) -> int:
    return int(getattr(texture, "texture", texture))


def _rock_mesh_from_vertex_data(
    vertex_data: np.ndarray,
    *,
    texture: int,
    exposure_baseline: float,
) -> BatchedMesh:
    return BatchedMesh.from_vertex_data(
        vertex_data,
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
        lighting_receiver=ROCK_LIGHTING_RECEIVER,
    )


def build_rock_meshes(
    rocks: Iterable[Rock],
    material_texture: object | None = None,
    chunk_size: float = 500.0,
    exposure_baseline: float = 1.0,
) -> list[BatchedMesh]:
    """Upload high-detail rock geometry in deterministic spatial XZ chunks.

    A cached neutral texture lets per-vertex mineral colors use the normal-
    aware textured material path.  Empty input returns before creating that
    texture or allocating any VBO.  Use :func:`build_rock_lod_meshes` to
    upload all three runtime-selectable levels with matching chunk coverage.
    """

    chunk_extent = float(chunk_size)
    if not math.isfinite(chunk_extent) or chunk_extent <= 0.0:
        raise ValueError("rock chunk_size must be a positive finite number")
    baseline = float(exposure_baseline)
    if not math.isfinite(baseline):
        raise ValueError("rock exposure_baseline must be finite")

    chunks: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    source = () if rocks is None else rocks
    for rock in source:
        if rock is None:
            continue
        if not isinstance(rock, Rock):
            raise TypeError("build_rock_meshes expects only Rock instances")
        chunk_key = (
            math.floor(float(rock.position.x) / chunk_extent),
            math.floor(float(rock.position.z) / chunk_extent),
        )
        vertex_data = build_rock_vertex_data(rock, RockLOD.HIGH)
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
            _rock_mesh_from_vertex_data(
                combined,
                texture=texture_value,
                exposure_baseline=baseline,
            )
        )
    return meshes


def build_rock_lod_meshes(
    rocks: Iterable[Rock],
    material_texture: object | None = None,
    chunk_size: float = 500.0,
    exposure_baseline: float = 1.0,
) -> list[tuple[BatchedMesh | None, BatchedMesh | None, BatchedMesh | None]]:
    """Upload HIGH/MEDIUM/LOW VBOs for every sorted spatial chunk.

    Tuple positions are stable and correspond to :class:`RockLOD`.  The
    optional slots make the contract robust to future culling-only levels;
    all current rock levels retain a closed visible silhouette.
    """

    chunk_extent = float(chunk_size)
    if not math.isfinite(chunk_extent) or chunk_extent <= 0.0:
        raise ValueError("rock chunk_size must be a positive finite number")
    baseline = float(exposure_baseline)
    if not math.isfinite(baseline):
        raise ValueError("rock exposure_baseline must be finite")

    chunks: dict[tuple[int, int], list[Rock]] = defaultdict(list)
    source = () if rocks is None else rocks
    for rock in source:
        if rock is None:
            continue
        if not isinstance(rock, Rock):
            raise TypeError("build_rock_lod_meshes expects only Rock instances")
        chunk_key = (
            math.floor(float(rock.position.x) / chunk_extent),
            math.floor(float(rock.position.z) / chunk_extent),
        )
        chunks[chunk_key].append(rock)

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
        for lod in RockLOD:
            parts = [
                build_rock_vertex_data(rock, lod)
                for rock in chunks[chunk_key]
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
                _rock_mesh_from_vertex_data(
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
    "ROCK_LOD_COUNT",
    "ROCK_VARIANT_COUNT",
    "Rock",
    "RockLOD",
    "build_rock_lod_meshes",
    "build_rock_vertex_data",
    "build_rock_meshes",
]
