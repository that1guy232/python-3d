"""Pure-CPU procedural geometry for the game's low-poly goblin.

The model is authored in normalized local space, posed, then emitted directly
in world space.  Local ``+Z`` is the goblin's forward direction and is aligned
to the supplied XZ ``facing`` vector.  ``position.y`` is the terrain contact
height: at least one shoe remains planted there throughout the walk cycle.

Rows use the same ``xyz rgb normal uv`` layout as the procedural nature
modules.  Every face owns its vertices and therefore its geometric unit normal,
which gives the character a deliberately faceted appearance without requiring
an OpenGL context, a texture, or an imported model asset.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import math
import random
from typing import TypeAlias

import numpy as np


GOBLIN_MODEL_DEFAULT_HEIGHT = 42.0
GOBLIN_MODEL_VERTEX_WIDTH = 11
GOBLIN_MODEL_POSE_COUNT = 8
GOBLIN_MODEL_VARIANT_COUNT = 8
GOBLIN_MODEL_MAX_HORIZONTAL_RADIUS_RATIO = 0.35
GOBLIN_MODEL_BOUNDING_RADIUS_RATIO = math.sqrt(
    GOBLIN_MODEL_MAX_HORIZONTAL_RADIUS_RATIO**2 + 0.5**2
)

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

_TUNIC_PALETTES: tuple[tuple[_Color, _Color], ...] = (
    ((0.39, 0.16, 0.10), (0.25, 0.085, 0.055)),
    ((0.20, 0.25, 0.14), (0.115, 0.15, 0.075)),
    ((0.18, 0.23, 0.28), (0.09, 0.12, 0.16)),
    ((0.35, 0.24, 0.09), (0.20, 0.125, 0.045)),
)


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


class _GoblinBuilder:
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
    builder: _GoblinBuilder,
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
    builder: _GoblinBuilder,
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
    builder: _GoblinBuilder,
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
    builder: _GoblinBuilder,
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
    builder: _GoblinBuilder,
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
    builder: _GoblinBuilder,
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


def _append_ear(
    builder: _GoblinBuilder,
    side: float,
    height: float,
    skin: _Color,
    inner_ear: _Color,
    reach_scale: float,
) -> None:
    h = height
    outer_x = side * 0.32 * h * reach_scale
    points = (
        (side * 0.125 * h, 0.915 * h),
        (side * 0.145 * h, 0.735 * h),
        (outer_x, 0.825 * h),
    )
    front = _append_triangular_prism_xy(
        builder,
        points,
        -0.030 * h,
        0.060 * h,
        skin,
        variation=0.03,
    )
    center = (
        sum(point[0] for point in front) / 3.0,
        sum(point[1] for point in front) / 3.0,
        0.063 * h,
    )
    inset: list[_Point] = []
    for point in front:
        inset.append(
            (
                center[0] + (point[0] - center[0]) * 0.63,
                center[1] + (point[1] - center[1]) * 0.63,
                center[2],
            )
        )
    builder.triangle(
        inset[0],
        inset[1],
        inset[2],
        builder.face_color(inner_ear, 0.018),
    )


def _append_face(
    builder: _GoblinBuilder,
    height: float,
    skin: _Color,
    inner_ear: _Color,
    *,
    head_width_scale: float,
    left_ear_scale: float,
    right_ear_scale: float,
    nose_skew: float,
) -> None:
    h = height
    _append_ellipsoid(
        builder,
        (0.0, 0.82 * h, 0.0),
        (0.17 * h * head_width_scale, 0.18 * h, 0.145 * h),
        skin,
        sides=9,
        rings=4,
        variation=0.035,
    )
    _append_ear(builder, -1.0, h, skin, inner_ear, left_ear_scale)
    _append_ear(builder, 1.0, h, skin, inner_ear, right_ear_scale)

    eye_color = (0.93, 0.76, 0.12)
    pupil_color = (0.075, 0.045, 0.020)
    brow_color = (0.105, 0.075, 0.030)
    for side in (-1.0, 1.0):
        eye_center = (side * 0.066 * h, 0.855 * h, 0.132 * h)
        _append_octahedron(
            builder,
            eye_center,
            (0.027 * h, 0.022 * h, 0.016 * h),
            eye_color,
            variation=0.01,
        )
        _append_octahedron(
            builder,
            (side * 0.061 * h, 0.852 * h, 0.148 * h),
            (0.009 * h, 0.011 * h, 0.006 * h),
            pupil_color,
            variation=0.0,
        )
        outer = (side * 0.105 * h, 0.899 * h, 0.132 * h)
        inner = (side * 0.028 * h, 0.883 * h, 0.151 * h)
        _append_segment(
            builder,
            outer,
            inner,
            0.008 * h,
            0.007 * h,
            brow_color,
            sides=4,
            variation=0.01,
        )

    _append_cone_z(
        builder,
        (nose_skew, 0.795 * h, 0.116 * h),
        (nose_skew * 1.25, 0.755 * h, 0.235 * h),
        0.058 * h,
        0.061 * h,
        builder.face_color(skin, 0.015),
        sides=7,
        variation=0.035,
    )

    mouth_color = (0.14, 0.045, 0.025)
    _append_frustum(
        builder,
        bottom_y=0.708 * h,
        top_y=0.728 * h,
        bottom_half_x=0.070 * h,
        bottom_half_z=0.006 * h,
        top_half_x=0.058 * h,
        top_half_z=0.006 * h,
        center_z=0.137 * h,
        color=mouth_color,
        variation=0.01,
    )
    tusk_color = (0.82, 0.78, 0.57)
    for side in (-1.0, 1.0):
        _append_segment(
            builder,
            (side * 0.047 * h, 0.719 * h, 0.143 * h),
            (side * 0.060 * h, 0.753 * h, 0.170 * h),
            0.010 * h,
            0.002 * h,
            tusk_color,
            sides=5,
            variation=0.015,
        )


def _append_body(
    builder: _GoblinBuilder,
    height: float,
    skin: _Color,
    tunic: _Color,
    tunic_dark: _Color,
) -> None:
    h = height
    _append_segment(
        builder,
        (0.0, 0.615 * h, 0.0),
        (0.0, 0.690 * h, 0.006 * h),
        0.080 * h,
        0.074 * h,
        skin,
        sides=7,
        variation=0.025,
    )
    _append_frustum(
        builder,
        bottom_y=0.320 * h,
        top_y=0.650 * h,
        bottom_half_x=0.180 * h,
        bottom_half_z=0.115 * h,
        top_half_x=0.145 * h,
        top_half_z=0.092 * h,
        center_z=-0.006 * h,
        color=tunic,
        variation=0.045,
    )

    belt_color = (0.165, 0.085, 0.035)
    _append_frustum(
        builder,
        bottom_y=0.375 * h,
        top_y=0.415 * h,
        bottom_half_x=0.185 * h,
        bottom_half_z=0.121 * h,
        top_half_x=0.180 * h,
        top_half_z=0.119 * h,
        center_z=-0.004 * h,
        color=belt_color,
        variation=0.025,
    )
    _append_frustum(
        builder,
        bottom_y=0.380 * h,
        top_y=0.410 * h,
        bottom_half_x=0.030 * h,
        bottom_half_z=0.010 * h,
        top_half_x=0.027 * h,
        top_half_z=0.010 * h,
        center_z=0.128 * h,
        color=(0.70, 0.49, 0.12),
        variation=0.02,
    )

    _append_triangular_prism_xy(
        builder,
        ((-0.088 * h, 0.384 * h), (0.088 * h, 0.384 * h), (0.0, 0.245 * h)),
        0.110 * h,
        0.132 * h,
        tunic_dark,
        front_color=tunic,
        variation=0.035,
    )


def _append_limbs(
    builder: _GoblinBuilder,
    height: float,
    skin: _Color,
    tunic: _Color,
    tunic_dark: _Color,
    *,
    phase: float,
    moving: bool,
    stance_scale: float,
) -> None:
    h = height
    gait = math.sin(phase) if moving else 0.0
    lift_wave = math.cos(phase) if moving else 0.0
    stride = 0.078 * h * gait
    left_lift = 0.034 * h * max(0.0, lift_wave)
    right_lift = 0.034 * h * max(0.0, -lift_wave)
    stance = 0.095 * h * stance_scale
    shoe_color = (0.145, 0.080, 0.040)

    for side, foot_stride, lift in (
        (-1.0, stride, left_lift),
        (1.0, -stride, right_lift),
    ):
        hip = (side * 0.086 * h, 0.355 * h, -0.006 * h)
        swing_amount = lift / max(0.034 * h, _EPSILON)
        knee = (
            side * stance * 0.94,
            0.205 * h + lift * 0.30,
            foot_stride * 0.45 + 0.032 * h * swing_amount,
        )
        ankle = (
            side * stance,
            0.080 * h + lift,
            foot_stride * 0.90 + 0.010 * h,
        )
        _append_segment(
            builder,
            hip,
            knee,
            0.058 * h,
            0.047 * h,
            tunic_dark,
            sides=6,
            variation=0.04,
        )
        _append_segment(
            builder,
            knee,
            ankle,
            0.044 * h,
            0.032 * h,
            skin,
            sides=6,
            variation=0.03,
        )
        _append_ellipsoid(
            builder,
            (
                side * stance,
                lift + 0.035 * h,
                foot_stride + 0.055 * h,
            ),
            (0.058 * h, 0.035 * h, 0.096 * h),
            shoe_color,
            sides=7,
            rings=2,
            variation=0.035,
        )

    for side, leg_stride in ((-1.0, stride), (1.0, -stride)):
        arm_swing = -leg_stride * 0.92
        shoulder = (side * 0.145 * h, 0.602 * h, -0.004 * h)
        elbow = (
            side * 0.215 * h,
            0.492 * h,
            arm_swing * 0.42 + 0.010 * h,
        )
        wrist = (
            side * 0.198 * h,
            0.345 * h,
            arm_swing,
        )
        _append_segment(
            builder,
            shoulder,
            elbow,
            0.052 * h,
            0.042 * h,
            tunic,
            sides=6,
            variation=0.04,
        )
        _append_segment(
            builder,
            elbow,
            wrist,
            0.039 * h,
            0.029 * h,
            skin,
            sides=6,
            variation=0.03,
        )
        _append_octahedron(
            builder,
            (wrist[0], wrist[1] - 0.012 * h, wrist[2] + 0.006 * h),
            (0.040 * h, 0.050 * h, 0.037 * h),
            skin,
            variation=0.03,
        )


def goblin_model_bounds(
    position: object,
    *,
    height: float = GOBLIN_MODEL_DEFAULT_HEIGHT,
) -> tuple[float, float, float, float, float, float]:
    """Return conservative world bounds in min/max-per-axis engine ordering."""

    x, y, z = _coerce_position(position)
    resolved_height = _validated_height(height)
    radius = resolved_height * GOBLIN_MODEL_MAX_HORIZONTAL_RADIUS_RATIO
    return (
        x - radius,
        x + radius,
        y,
        y + resolved_height,
        z - radius,
        z + radius,
    )


def build_goblin_vertex_data(
    position: object,
    *,
    facing: object = (0.0, 1.0),
    height: float = GOBLIN_MODEL_DEFAULT_HEIGHT,
    walk_phase: float = 0.0,
    moving: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Build one deterministic, posed low-poly goblin in world space.

    ``walk_phase`` is expressed in radians.  When ``moving`` is false it has no
    effect, making the neutral pose stable for previews and CPU tests.  Seeded
    variation changes skin/tunic shades and a few facial proportions without
    changing topology or the conservative bounds.
    """

    origin = _coerce_position(position)
    resolved_facing = _coerce_facing(facing)
    resolved_height = _validated_height(height)
    phase = float(walk_phase)
    if not math.isfinite(phase):
        raise ValueError("goblin walk_phase must be finite")
    try:
        seed_value = int(seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("goblin seed must be integer-compatible") from exc

    parameter_rng = _seeded_rng(seed_value, 0x50415241)
    face_rng = _seeded_rng(seed_value, 0x46414345)
    skin = _clamp_color(
        (
            0.265 + parameter_rng.uniform(-0.025, 0.025),
            0.515 + parameter_rng.uniform(-0.035, 0.035),
            0.135 + parameter_rng.uniform(-0.018, 0.018),
        )
    )
    inner_ear = _clamp_color(
        (skin[0] * 1.08 + 0.05, skin[1] * 0.68, skin[2] * 0.72 + 0.045)
    )
    tunic, tunic_dark = parameter_rng.choice(_TUNIC_PALETTES)
    tunic_factor = parameter_rng.uniform(0.92, 1.08)
    tunic = _clamp_color(tuple(component * tunic_factor for component in tunic))  # type: ignore[arg-type]
    tunic_dark = _clamp_color(
        tuple(component * tunic_factor for component in tunic_dark)  # type: ignore[arg-type]
    )
    head_width_scale = parameter_rng.uniform(0.96, 1.04)
    left_ear_scale = parameter_rng.uniform(0.96, 1.04)
    right_ear_scale = parameter_rng.uniform(0.96, 1.04)
    stance_scale = parameter_rng.uniform(0.97, 1.03)
    nose_skew = parameter_rng.uniform(-0.010, 0.010) * resolved_height

    builder = _GoblinBuilder(origin, resolved_facing, face_rng)
    _append_limbs(
        builder,
        resolved_height,
        skin,
        tunic,
        tunic_dark,
        phase=phase,
        moving=bool(moving),
        stance_scale=stance_scale,
    )
    _append_body(builder, resolved_height, skin, tunic, tunic_dark)
    _append_face(
        builder,
        resolved_height,
        skin,
        inner_ear,
        head_width_scale=head_width_scale,
        left_ear_scale=left_ear_scale,
        right_ear_scale=right_ear_scale,
        nose_skew=nose_skew,
    )

    vertex_data = np.asarray(builder.rows, dtype=np.float32)
    if vertex_data.size == 0:
        return np.empty((0, GOBLIN_MODEL_VERTEX_WIDTH), dtype=np.float32)
    vertex_data = np.ascontiguousarray(
        vertex_data.reshape((-1, GOBLIN_MODEL_VERTEX_WIDTH)),
        dtype=np.float32,
    )
    if not np.isfinite(vertex_data).all():
        raise ValueError("procedural goblin generation produced non-finite vertices")
    normal_lengths = np.linalg.norm(vertex_data[:, 6:9], axis=1)
    if not np.allclose(normal_lengths, 1.0, rtol=2.0e-5, atol=2.0e-5):
        raise ValueError("procedural goblin generation produced invalid normals")
    return vertex_data


@lru_cache(maxsize=GOBLIN_MODEL_VARIANT_COUNT * (GOBLIN_MODEL_POSE_COUNT + 1))
def _cached_local_pose(
    height: float,
    variant: int,
    moving: bool,
    pose_index: int,
) -> np.ndarray:
    """Build one immutable local-space template for the dynamic actor batch."""

    phase = (
        math.tau * int(pose_index) / GOBLIN_MODEL_POSE_COUNT
        if moving
        else 0.0
    )
    vertex_data = build_goblin_vertex_data(
        (0.0, 0.0, 0.0),
        facing=(0.0, 1.0),
        height=height,
        walk_phase=phase,
        moving=moving,
        seed=variant,
    )
    vertex_data.setflags(write=False)
    return vertex_data


def goblin_model_pose_index(walk_phase: float, *, moving: bool) -> int:
    """Return the shared pose-cache index for one animation state."""

    phase = float(walk_phase)
    if not math.isfinite(phase):
        raise ValueError("goblin walk_phase must be finite")
    if not moving:
        return 0
    return (
        int(
            math.floor(
                (phase % math.tau) * GOBLIN_MODEL_POSE_COUNT / math.tau + 0.5
            )
        )
        % GOBLIN_MODEL_POSE_COUNT
    )


def cached_goblin_local_pose_vertex_data(
    *,
    height: float = GOBLIN_MODEL_DEFAULT_HEIGHT,
    walk_phase: float = 0.0,
    moving: bool = False,
    variant: int = 0,
) -> np.ndarray:
    """Return an immutable local-space pose suitable for a static GPU atlas."""

    resolved_height = _validated_height(height)
    variant_index = int(variant) % GOBLIN_MODEL_VARIANT_COUNT
    pose_index = goblin_model_pose_index(walk_phase, moving=bool(moving))
    return _cached_local_pose(
        resolved_height,
        variant_index,
        bool(moving),
        pose_index,
    )


def transformed_cached_goblin_vertex_data(
    position: object,
    *,
    facing: object = (0.0, 1.0),
    height: float = GOBLIN_MODEL_DEFAULT_HEIGHT,
    walk_phase: float = 0.0,
    moving: bool = False,
    variant: int = 0,
) -> np.ndarray:
    """Transform a shared generated pose into world space with NumPy.

    Runtime goblins choose among a small set of walk poses and visual variants.
    The expensive topology and flat-normal construction happens once per shared
    template; ordinary movement only rotates/translates its position and normal
    columns. The returned array is independent and safe for batch concatenation.
    """

    origin_x, origin_y, origin_z = _coerce_position(position)
    forward_x, forward_z = _coerce_facing(facing)
    resolved_height = _validated_height(height)
    phase = float(walk_phase)
    variant_index = int(variant) % GOBLIN_MODEL_VARIANT_COUNT
    pose_index = goblin_model_pose_index(phase, moving=bool(moving))
    local = _cached_local_pose(
        resolved_height,
        variant_index,
        bool(moving),
        pose_index,
    )
    result = np.array(local, dtype=np.float32, copy=True, order="C")

    local_x = local[:, 0]
    local_z = local[:, 2]
    result[:, 0] = origin_x + forward_z * local_x + forward_x * local_z
    result[:, 1] = origin_y + local[:, 1]
    result[:, 2] = origin_z - forward_x * local_x + forward_z * local_z

    local_nx = local[:, 6]
    local_nz = local[:, 8]
    result[:, 6] = forward_z * local_nx + forward_x * local_nz
    result[:, 8] = -forward_x * local_nx + forward_z * local_nz
    return result


def prewarm_goblin_pose_cache(
    variants: Sequence[int],
    *,
    height: float = GOBLIN_MODEL_DEFAULT_HEIGHT,
) -> None:
    """Generate the requested variants' idle and walk templates during loading."""

    resolved_height = _validated_height(height)
    for variant in sorted(
        {int(value) % GOBLIN_MODEL_VARIANT_COUNT for value in variants}
    ):
        _cached_local_pose(resolved_height, variant, False, 0)
        for pose_index in range(GOBLIN_MODEL_POSE_COUNT):
            _cached_local_pose(resolved_height, variant, True, pose_index)


__all__ = [
    "GOBLIN_MODEL_BOUNDING_RADIUS_RATIO",
    "GOBLIN_MODEL_DEFAULT_HEIGHT",
    "GOBLIN_MODEL_MAX_HORIZONTAL_RADIUS_RATIO",
    "GOBLIN_MODEL_POSE_COUNT",
    "GOBLIN_MODEL_VARIANT_COUNT",
    "GOBLIN_MODEL_VERTEX_WIDTH",
    "build_goblin_vertex_data",
    "cached_goblin_local_pose_vertex_data",
    "goblin_model_bounds",
    "goblin_model_pose_index",
    "prewarm_goblin_pose_cache",
    "transformed_cached_goblin_vertex_data",
]
