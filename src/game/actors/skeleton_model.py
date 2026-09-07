"""Faceted skeleton geometry with a hollow rib cage and cached walking poses.

Local +Z faces forward, and the origin is the ground contact point. Geometry
uses xyz/rgb/normal/uv rows and the same neutral material as other creatures.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from game.actors.model_geometry import (
    ModelBuilder,
    _append_ellipsoid,
    _append_frustum,
    _append_segment,
    _append_triangular_prism_xy,
    _coerce_facing,
    _coerce_position,
    _seeded_rng,
    _validated_height,
)

SKELETON_MODEL_DEFAULT_HEIGHT = 56.0
SKELETON_MODEL_POSE_COUNT = 8
SKELETON_MODEL_VARIANT_COUNT = 4
SKELETON_MODEL_VERTEX_WIDTH = 11


def build_skeleton_vertex_data(
    position=(0.0, 0.0, 0.0),
    *,
    facing=(0.0, 1.0),
    height: float = SKELETON_MODEL_DEFAULT_HEIGHT,
    walk_phase: float = 0.0,
    moving: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Build an independently posed skeleton without an OpenGL context."""
    h = _validated_height(height)
    phase = float(walk_phase)
    if not math.isfinite(phase):
        raise ValueError("skeleton walk_phase must be finite")
    rng = _seeded_rng(seed, 1)
    tint = rng.uniform(0.88, 1.0)
    bone = (0.91 * tint, 0.86 * tint, 0.71 * tint)
    worn_bone = (0.70 * tint, 0.65 * tint, 0.52 * tint)
    socket = (0.065, 0.043, 0.030)
    builder = ModelBuilder(_coerce_position(position), _coerce_facing(facing), rng)

    def point(value):
        return tuple(component * h for component in value)

    def joint(center, radii, color=bone, *, sides=6, rings=2):
        _append_ellipsoid(builder, point(center), point(radii), color,
                          sides=sides, rings=rings, variation=0.025)

    def segment(start, end, radius=0.015, color=bone, *, caps=True):
        _append_segment(builder, point(start), point(end), radius*h, radius*0.85*h,
                        color, sides=5, cap_start=caps, cap_end=caps)

    def box(center_x, low_y, high_y, half_x, half_z, center_z, color=bone):
        _append_frustum(
            builder, bottom_y=low_y*h, top_y=high_y*h,
            bottom_half_x=half_x*h, bottom_half_z=half_z*h,
            top_half_x=half_x*0.92*h, top_half_z=half_z*0.92*h,
            center_x=center_x*h, center_z=center_z*h, color=color,
        )

    # Skull, recessed dark sockets, separate jaw and uneven individual teeth.
    joint((0.0, 0.887, 0.0), (0.108, 0.113, 0.092), sides=10, rings=4)
    for side in (-1.0, 1.0):
        joint((side*0.046, 0.903, 0.078), (0.031, 0.034, 0.020), socket)
        joint((side*0.046, 0.902, 0.097), (0.008, 0.010, 0.003),
              (0.96, 0.29, 0.055), sides=4, rings=1)
        segment((side*0.016, 0.928, 0.083), (side*0.082, 0.943, 0.070), 0.013)
        segment((side*0.074, 0.802, 0.025), (side*0.090, 0.850, 0.002), 0.016)
    _append_triangular_prism_xy(
        builder, [(-0.017*h, 0.853*h), (0.017*h, 0.853*h), (0.0, 0.880*h)],
        0.085*h, 0.101*h, socket, variation=0.0,
    )
    box(0.0, 0.787, 0.812, 0.071, 0.042, 0.031, worn_bone)
    for index in range(7):
        tooth_x = (index-3)*0.018
        box(tooth_x, 0.812, 0.830 - (index % 2)*0.003, 0.0065, 0.008, 0.075)
    for y in (0.758, 0.780):
        joint((0.0, y, -0.010), (0.022, 0.011, 0.024), worn_bone)

    # The ribs are separate curved bones, leaving visible openings between
    # them; there is no solid torso hidden underneath the cage.
    for index in range(8):
        joint((0.0, 0.486 + index*0.034, -0.032), (0.022, 0.012, 0.026), worn_bone)
    segment((0.0, 0.565, 0.071), (0.0, 0.730, 0.071), 0.011)
    for index in range(5):
        y = 0.583 + index*0.034
        width = 0.116 - abs(index-2)*0.010
        for side in (-1.0, 1.0):
            arc = (
                (side*0.017, y+0.012, -0.032),
                (side*0.076, y+0.012, -0.054),
                (side*width, y, -0.012),
                (side*width*0.93, y-0.010, 0.039),
                (side*0.064, y-0.016, 0.072),
                (side*0.005, y-0.020, 0.071),
            )
            for start, end in zip(arc, arc[1:]):
                segment(start, end, 0.0085, caps=False)
    for side in (-1.0, 1.0):
        segment((0.0, 0.740, 0.026), (side*0.148, 0.735, -0.005), 0.013)
        box(side*0.062, 0.437, 0.510, 0.036, 0.024, -0.010, worn_bone)
        segment((side*0.084, 0.455, 0.0), (0.0, 0.426, 0.031), 0.019)

    gait = math.sin(phase) if moving else 0.0
    lift_wave = math.cos(phase) if moving else 0.0
    for side in (-1.0, 1.0):
        stride = -side * 0.085 * gait
        lift = 0.045 * max(0.0, -side*lift_wave)
        hip = (side*0.068, 0.444, -0.005)
        knee = (side*0.076, 0.242+lift*0.3, stride*0.45)
        ankle = (side*0.081, 0.057+lift, stride)
        segment(hip, knee, 0.019)
        joint(knee, (0.024, 0.024, 0.023), worn_bone)
        segment(knee, ankle, 0.015)
        segment((knee[0]+side*0.022, knee[1], knee[2]),
                (ankle[0]+side*0.010, ankle[1], ankle[2]), 0.008, worn_bone)
        joint((ankle[0], 0.021+lift, stride+0.030), (0.034, 0.021, 0.055))
        for toe in range(3):
            joint((ankle[0]+(toe-1)*0.019, 0.012+lift, stride+0.072),
                  (0.010, 0.012, 0.027), sides=4, rings=1)

        shoulder = (side*0.151, 0.731, -0.010)
        elbow = (side*0.190, 0.582, -stride*0.42)
        wrist = (side*0.175, 0.447, -stride+0.025)
        joint(shoulder, (0.026, 0.026, 0.026), worn_bone)
        segment(shoulder, elbow, 0.016)
        joint(elbow, (0.020, 0.020, 0.020), worn_bone)
        for offset in (-0.008, 0.008):
            segment((elbow[0]+offset, elbow[1], elbow[2]),
                    (wrist[0]+offset, wrist[1], wrist[2]), 0.0085)
        joint((wrist[0], wrist[1]-0.018, wrist[2]), (0.024, 0.027, 0.014))
        for finger in range(3):
            x = wrist[0] + (finger-1)*0.016
            start = (x, wrist[1]-0.032, wrist[2])
            bend = (x, wrist[1]-0.063, wrist[2]+0.006)
            segment(start, bend, 0.0065)
            segment(bend, (x, wrist[1]-0.069, wrist[2]+0.024), 0.006)

    return np.ascontiguousarray(builder.rows, dtype=np.float32)


def skeleton_model_pose_index(walk_phase: float, *, moving: bool) -> int:
    phase = float(walk_phase)
    if not math.isfinite(phase):
        raise ValueError("skeleton walk_phase must be finite")
    if not moving:
        return 0
    return int(math.floor((phase % math.tau)*SKELETON_MODEL_POSE_COUNT/math.tau + 0.5)) % SKELETON_MODEL_POSE_COUNT


@lru_cache(maxsize=128)
def _cached_local_pose(height: float, variant: int, moving: bool, pose: int) -> np.ndarray:
    data = build_skeleton_vertex_data(
        height=height, seed=variant, moving=moving,
        walk_phase=math.tau*pose/SKELETON_MODEL_POSE_COUNT,
    )
    data.setflags(write=False)
    return data


def cached_skeleton_local_pose_vertex_data(
    *, height=SKELETON_MODEL_DEFAULT_HEIGHT, walk_phase=0.0, moving=False, variant=0
) -> np.ndarray:
    return _cached_local_pose(
        _validated_height(height), int(variant) % SKELETON_MODEL_VARIANT_COUNT,
        bool(moving), skeleton_model_pose_index(walk_phase, moving=bool(moving)),
    )


def transformed_cached_skeleton_vertex_data(position, *, facing=(0.0, 1.0), **pose) -> np.ndarray:
    x, y, z = _coerce_position(position)
    fx, fz = _coerce_facing(facing)
    local = cached_skeleton_local_pose_vertex_data(**pose)
    result = np.array(local, copy=True, order="C")
    result[:, 0] = x + fz*local[:, 0] + fx*local[:, 2]
    result[:, 1] = y + local[:, 1]
    result[:, 2] = z - fx*local[:, 0] + fz*local[:, 2]
    result[:, 6] = fz*local[:, 6] + fx*local[:, 8]
    result[:, 8] = -fx*local[:, 6] + fz*local[:, 8]
    return result


def prewarm_skeleton_pose_cache(variants, *, height=SKELETON_MODEL_DEFAULT_HEIGHT) -> None:
    for variant in set(variants):
        cached_skeleton_local_pose_vertex_data(height=height, variant=variant)
        for pose in range(SKELETON_MODEL_POSE_COUNT):
            cached_skeleton_local_pose_vertex_data(
                height=height, variant=variant, moving=True,
                walk_phase=math.tau*pose/SKELETON_MODEL_POSE_COUNT,
            )
