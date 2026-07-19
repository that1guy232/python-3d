"""Deterministic framebuffer comparison metrics for renderer cutovers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameComparisonThresholds:
    channel_tolerance: int = 2
    min_stable_pixel_ratio: float = 0.995
    max_mean_absolute_error: float = 1.0
    max_p99_absolute_error: float = 2.0
    max_absolute_error: int = 16
    max_changed_pixel_ratio: float = 0.002


@dataclass(frozen=True, slots=True)
class FrameComparison:
    width: int
    height: int
    stable_pixel_ratio: float
    mean_absolute_error: float
    p95_absolute_error: float
    p99_absolute_error: float
    max_absolute_error: int
    changed_pixel_ratio: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_rgb_frames(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    drift_reference: np.ndarray | None = None,
    thresholds: FrameComparisonThresholds | None = None,
) -> FrameComparison:
    """Compare RGB uint8 frames, excluding pixels with measured reference drift."""

    policy = thresholds or FrameComparisonThresholds()
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"frame shapes differ: reference={reference.shape}, candidate={candidate.shape}"
        )
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frames, received {reference.shape}")
    if drift_reference is not None:
        drift_reference = np.asarray(drift_reference)
        if drift_reference.shape != reference.shape:
            raise ValueError(
                "drift-reference shape differs: "
                f"reference={reference.shape}, drift={drift_reference.shape}"
            )

    ref = reference.astype(np.int16, copy=False)
    other = candidate.astype(np.int16, copy=False)
    if drift_reference is None:
        stable = np.ones(reference.shape[:2], dtype=bool)
    else:
        drift = np.abs(drift_reference.astype(np.int16, copy=False) - ref)
        stable = np.max(drift, axis=2) <= int(policy.channel_tolerance)

    stable_count = int(np.count_nonzero(stable))
    pixel_count = int(stable.size)
    stable_ratio = stable_count / max(1, pixel_count)
    if stable_count == 0:
        return FrameComparison(
            width=int(reference.shape[1]),
            height=int(reference.shape[0]),
            stable_pixel_ratio=0.0,
            mean_absolute_error=float("inf"),
            p95_absolute_error=float("inf"),
            p99_absolute_error=float("inf"),
            max_absolute_error=255,
            changed_pixel_ratio=1.0,
            passed=False,
        )

    absolute = np.abs(other - ref)
    stable_absolute = absolute[stable]
    per_pixel_max = np.max(stable_absolute, axis=1)
    mean_error = float(np.mean(stable_absolute))
    p95 = float(np.percentile(stable_absolute, 95))
    p99 = float(np.percentile(stable_absolute, 99))
    max_error = int(np.max(stable_absolute))
    changed_ratio = float(
        np.count_nonzero(per_pixel_max > int(policy.channel_tolerance))
        / stable_count
    )
    passed = bool(
        stable_ratio >= policy.min_stable_pixel_ratio
        and mean_error <= policy.max_mean_absolute_error
        and p99 <= policy.max_p99_absolute_error
        and max_error <= policy.max_absolute_error
        and changed_ratio <= policy.max_changed_pixel_ratio
    )
    return FrameComparison(
        width=int(reference.shape[1]),
        height=int(reference.shape[0]),
        stable_pixel_ratio=stable_ratio,
        mean_absolute_error=mean_error,
        p95_absolute_error=p95,
        p99_absolute_error=p99,
        max_absolute_error=max_error,
        changed_pixel_ratio=changed_ratio,
        passed=passed,
    )


def amplified_rgb_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    scale: int = 4,
) -> np.ndarray:
    """Return a displayable amplified absolute RGB difference image."""

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("difference frames must have matching shapes")
    difference = np.abs(
        candidate.astype(np.int16, copy=False)
        - reference.astype(np.int16, copy=False)
    )
    return np.clip(difference * max(1, int(scale)), 0, 255).astype(np.uint8)
