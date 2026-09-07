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

import numpy as np

from game.actors.model_geometry import (
    ModelBuilder as _GoblinBuilder,
    _Color, _Point, _EPSILON,
    _append_cone_z, _append_ellipsoid, _append_frustum,
    _append_octahedron, _append_segment, _append_triangular_prism_xy,
    _clamp_color, _coerce_facing, _coerce_position, _seeded_rng, _validated_height,
    _clamp_color
)



GOBLIN_MODEL_DEFAULT_HEIGHT = 42.0
GOBLIN_MODEL_VERTEX_WIDTH = 11
GOBLIN_MODEL_POSE_COUNT = 8
GOBLIN_MODEL_VARIANT_COUNT = 8
GOBLIN_MODEL_MAX_HORIZONTAL_RADIUS_RATIO = 0.35
GOBLIN_MODEL_BOUNDING_RADIUS_RATIO = math.sqrt(
    GOBLIN_MODEL_MAX_HORIZONTAL_RADIUS_RATIO**2 + 0.5**2
)

_TUNIC_PALETTES: tuple[tuple[_Color, _Color], ...] = (
    ((0.39, 0.16, 0.10), (0.25, 0.085, 0.055)),
    ((0.20, 0.25, 0.14), (0.115, 0.15, 0.075)),
    ((0.18, 0.23, 0.28), (0.09, 0.12, 0.16)),
    ((0.35, 0.24, 0.09), (0.20, 0.125, 0.045)),
)


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
