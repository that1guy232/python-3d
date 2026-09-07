"""Shared deterministic primitives for flat-shaded procedural creatures."""

from __future__ import annotations

from collections.abc import Sequence
import math
import random
from typing import TypeAlias

_EPSILON = 1.0e-10
_UINT64_MASK = (1 << 64) - 1
_TAU = math.tau

_Point: TypeAlias = tuple[float, float, float]
_Color: TypeAlias = tuple[float, float, float]
_UV: TypeAlias = tuple[float, float]
_VertexRow: TypeAlias = tuple[
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

def _seeded_rng(seed: int, channel: int) -> random.Random:
    """Return an independent deterministic stream for one model concern."""

    value = int(seed) & _UINT64_MASK
    value ^= (int(channel) * 0x9E3779B97F4A7C15) & _UINT64_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return random.Random(value)


def _coerce_position(position: object) -> _Point:
    if all(hasattr(position, name) for name in ("x", "y", "z")):
        values = (
            float(getattr(position, "x")),
            float(getattr(position, "y")),
            float(getattr(position, "z")),
        )
    else:
        try:
            raw = tuple(position)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "goblin position must expose x/y/z or be a 2/3 item sequence"
            ) from exc
        try:
            if len(raw) == 2:
                values = (float(raw[0]), 0.0, float(raw[1]))
            elif len(raw) == 3:
                values = (float(raw[0]), float(raw[1]), float(raw[2]))
            else:
                raise ValueError(
                    "goblin position must contain two or three coordinates"
                )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("goblin position"):
                raise
            raise ValueError("goblin position coordinates must be numeric") from exc

    if not all(math.isfinite(value) for value in values):
        raise ValueError("goblin position coordinates must be finite")
    return values


def _coerce_facing(facing: object) -> tuple[float, float]:
    if hasattr(facing, "x") and hasattr(facing, "z"):
        x = float(getattr(facing, "x"))
        z = float(getattr(facing, "z"))
    else:
        try:
            values = tuple(facing)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "goblin facing must expose x/z or be a 2/3 item sequence"
            ) from exc
        try:
            if len(values) == 2:
                x, z = float(values[0]), float(values[1])
            elif len(values) == 3:
                x, z = float(values[0]), float(values[2])
            else:
                raise ValueError("goblin facing must contain two or three coordinates")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("goblin facing"):
                raise
            raise ValueError("goblin facing coordinates must be numeric") from exc

    if not math.isfinite(x) or not math.isfinite(z):
        raise ValueError("goblin facing coordinates must be finite")
    length = math.hypot(x, z)
    if length <= _EPSILON:
        raise ValueError("goblin facing must have a non-zero XZ length")
    return x / length, z / length


def _validated_height(height: float) -> float:
    value = float(height)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("goblin height must be a positive finite number")
    return value


def _clamp_color(color: _Color) -> _Color:
    return tuple(max(0.0, min(1.0, float(component))) for component in color)  # type: ignore[return-value]


def _add(a: _Point, b: _Point) -> _Point:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub(a: _Point, b: _Point) -> _Point:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _scale(value: _Point, factor: float) -> _Point:
    return value[0] * factor, value[1] * factor, value[2] * factor


def _cross(a: _Point, b: _Point) -> _Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(value: _Point) -> float:
    return math.sqrt(
        value[0] * value[0] + value[1] * value[1] + value[2] * value[2]
    )


def _normalized(value: _Point) -> _Point:
    length = _length(value)
    if length <= _EPSILON:
        raise ValueError("cannot normalize a zero-length geometry vector")
    inverse = 1.0 / length
    return value[0] * inverse, value[1] * inverse, value[2] * inverse


class ModelBuilder:
    """Transform local points and append flat-shaded world-space triangles."""

    __slots__ = ("origin", "right", "forward", "rows", "rng")

    def __init__(
        self,
        origin: _Point,
        facing: tuple[float, float],
        rng: random.Random,
    ) -> None:
        forward_x, forward_z = facing
        self.origin = origin
        self.right = (forward_z, 0.0, -forward_x)
        self.forward = (forward_x, 0.0, forward_z)
        self.rows: list[_VertexRow] = []
        self.rng = rng

    def world_point(self, point: _Point) -> _Point:
        x, y, z = point
        return (
            self.origin[0] + self.right[0] * x + self.forward[0] * z,
            self.origin[1] + y,
            self.origin[2] + self.right[2] * x + self.forward[2] * z,
        )

    def face_color(self, base: _Color, variation: float = 0.025) -> _Color:
        if variation <= 0.0:
            return _clamp_color(base)
        lightness = 1.0 + self.rng.uniform(-variation, variation)
        warmth = self.rng.uniform(-variation * 0.18, variation * 0.18)
        return _clamp_color(
            (
                base[0] * lightness + warmth,
                base[1] * lightness,
                base[2] * lightness - warmth * 0.35,
            )
        )

    def triangle(
        self,
        a: _Point,
        b: _Point,
        c: _Point,
        color: _Color,
        uvs: tuple[_UV, _UV, _UV] = ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)),
    ) -> None:
        world_a = self.world_point(a)
        world_b = self.world_point(b)
        world_c = self.world_point(c)
        ab = _sub(world_b, world_a)
        ac = _sub(world_c, world_a)
        normal = _cross(ab, ac)
        normal_length = _length(normal)
        if normal_length <= _EPSILON:
            return
        nx, ny, nz = _scale(normal, 1.0 / normal_length)
        r, g, blue = _clamp_color(color)
        for point, uv in zip((world_a, world_b, world_c), uvs):
            self.rows.append(
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

    def quad(
        self,
        a: _Point,
        b: _Point,
        c: _Point,
        d: _Point,
        color: _Color,
    ) -> None:
        self.triangle(
            a,
            b,
            c,
            color,
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        )
        self.triangle(
            a,
            c,
            d,
            color,
            ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )


def _append_frustum(
    builder: ModelBuilder,
    *,
    bottom_y: float,
    top_y: float,
    bottom_half_x: float,
    bottom_half_z: float,
    top_half_x: float,
    top_half_z: float,
    color: _Color,
    center_x: float = 0.0,
    center_z: float = 0.0,
    variation: float = 0.025,
) -> None:
    xb, zb = bottom_half_x, bottom_half_z
    xt, zt = top_half_x, top_half_z
    bbl = (center_x - xb, bottom_y, center_z - zb)
    bbr = (center_x + xb, bottom_y, center_z - zb)
    bfr = (center_x + xb, bottom_y, center_z + zb)
    bfl = (center_x - xb, bottom_y, center_z + zb)
    tbl = (center_x - xt, top_y, center_z - zt)
    tbr = (center_x + xt, top_y, center_z - zt)
    tfr = (center_x + xt, top_y, center_z + zt)
    tfl = (center_x - xt, top_y, center_z + zt)

    builder.quad(bfl, bfr, tfr, tfl, builder.face_color(color, variation))
    builder.quad(bbr, bbl, tbl, tbr, builder.face_color(color, variation))
    builder.quad(bbl, bfl, tfl, tbl, builder.face_color(color, variation))
    builder.quad(bfr, bbr, tbr, tfr, builder.face_color(color, variation))
    builder.quad(tbl, tfl, tfr, tbr, builder.face_color(color, variation))
    builder.quad(bfl, bbl, bbr, bfr, builder.face_color(color, variation))


def _append_segment(
    builder: ModelBuilder,
    start: _Point,
    end: _Point,
    start_radius: float,
    end_radius: float,
    color: _Color,
    *,
    sides: int = 6,
    variation: float = 0.025,
    cap_start: bool = True,
    cap_end: bool = True,
) -> None:
    axis = _sub(end, start)
    if _length(axis) <= _EPSILON:
        return
    direction = _normalized(axis)
    reference = (0.0, 1.0, 0.0)
    if abs(direction[1]) > 0.90:
        reference = (1.0, 0.0, 0.0)
    basis_a = _normalized(_cross(direction, reference))
    basis_b = _normalized(_cross(direction, basis_a))

    start_ring: list[_Point] = []
    end_ring: list[_Point] = []
    for index in range(max(3, int(sides))):
        angle = _TAU * index / max(3, int(sides))
        radial = _add(_scale(basis_a, math.cos(angle)), _scale(basis_b, math.sin(angle)))
        start_ring.append(_add(start, _scale(radial, start_radius)))
        end_ring.append(_add(end, _scale(radial, end_radius)))

    count = len(start_ring)
    for index in range(count):
        next_index = (index + 1) % count
        face_color = builder.face_color(color, variation)
        builder.quad(
            start_ring[index],
            start_ring[next_index],
            end_ring[next_index],
            end_ring[index],
            face_color,
        )
        if cap_start:
            builder.triangle(
                start,
                start_ring[next_index],
                start_ring[index],
                builder.face_color(color, variation * 0.65),
            )
        if cap_end:
            builder.triangle(
                end,
                end_ring[index],
                end_ring[next_index],
                builder.face_color(color, variation * 0.65),
            )


def _append_ellipsoid(
    builder: ModelBuilder,
    center: _Point,
    radii: _Point,
    color: _Color,
    *,
    sides: int = 8,
    rings: int = 4,
    variation: float = 0.025,
) -> None:
    side_count = max(4, int(sides))
    ring_count = max(2, int(rings))
    cx, cy, cz = center
    rx, ry, rz = radii
    interior: list[list[_Point]] = []
    for ring_index in range(1, ring_count + 1):
        latitude = -math.pi * 0.5 + math.pi * ring_index / (ring_count + 1)
        horizontal = math.cos(latitude)
        vertical = math.sin(latitude)
        ring: list[_Point] = []
        for side_index in range(side_count):
            angle = _TAU * side_index / side_count
            ring.append(
                (
                    cx + rx * horizontal * math.cos(angle),
                    cy + ry * vertical,
                    cz + rz * horizontal * math.sin(angle),
                )
            )
        interior.append(ring)

    bottom = (cx, cy - ry, cz)
    top = (cx, cy + ry, cz)
    first_ring = interior[0]
    for index in range(side_count):
        next_index = (index + 1) % side_count
        builder.triangle(
            bottom,
            first_ring[index],
            first_ring[next_index],
            builder.face_color(color, variation),
        )

    for lower, upper in zip(interior, interior[1:]):
        for index in range(side_count):
            next_index = (index + 1) % side_count
            builder.quad(
                lower[index],
                upper[index],
                upper[next_index],
                lower[next_index],
                builder.face_color(color, variation),
            )

    last_ring = interior[-1]
    for index in range(side_count):
        next_index = (index + 1) % side_count
        builder.triangle(
            last_ring[index],
            top,
            last_ring[next_index],
            builder.face_color(color, variation),
        )


def _append_octahedron(
    builder: ModelBuilder,
    center: _Point,
    radii: _Point,
    color: _Color,
    *,
    variation: float = 0.02,
) -> None:
    cx, cy, cz = center
    rx, ry, rz = radii
    top = (cx, cy + ry, cz)
    bottom = (cx, cy - ry, cz)
    equator = (
        (cx, cy, cz + rz),
        (cx + rx, cy, cz),
        (cx, cy, cz - rz),
        (cx - rx, cy, cz),
    )
    for index in range(4):
        next_index = (index + 1) % 4
        face_color = builder.face_color(color, variation)
        builder.triangle(top, equator[index], equator[next_index], face_color)
        builder.triangle(
            bottom,
            equator[next_index],
            equator[index],
            builder.face_color(color, variation),
        )


def _append_cone_z(
    builder: ModelBuilder,
    base_center: _Point,
    tip: _Point,
    radius_x: float,
    radius_y: float,
    color: _Color,
    *,
    sides: int = 7,
    variation: float = 0.025,
) -> None:
    count = max(3, int(sides))
    bx, by, bz = base_center
    ring = [
        (
            bx + radius_x * math.cos(_TAU * index / count),
            by + radius_y * math.sin(_TAU * index / count),
            bz,
        )
        for index in range(count)
    ]
    for index in range(count):
        next_index = (index + 1) % count
        builder.triangle(
            ring[index],
            ring[next_index],
            tip,
            builder.face_color(color, variation),
        )
        builder.triangle(
            base_center,
            ring[next_index],
            ring[index],
            builder.face_color(color, variation * 0.6),
        )


def _ordered_xy_polygon(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    return sorted(
        points,
        key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x),
    )


def _append_triangular_prism_xy(
    builder: ModelBuilder,
    points: Sequence[tuple[float, float]],
    z_back: float,
    z_front: float,
    color: _Color,
    *,
    front_color: _Color | None = None,
    variation: float = 0.025,
) -> tuple[_Point, _Point, _Point]:
    if len(points) != 3:
        raise ValueError("triangular prism requires exactly three XY points")
    ordered = _ordered_xy_polygon(points)
    back = [(x, y, z_back) for x, y in ordered]
    front = [(x, y, z_front) for x, y in ordered]
    builder.triangle(
        front[0],
        front[1],
        front[2],
        builder.face_color(front_color or color, variation),
    )
    builder.triangle(
        back[0],
        back[2],
        back[1],
        builder.face_color(color, variation),
    )
    for index in range(3):
        next_index = (index + 1) % 3
        builder.quad(
            back[index],
            back[next_index],
            front[next_index],
            front[index],
            builder.face_color(color, variation),
        )
    return front[0], front[1], front[2]


