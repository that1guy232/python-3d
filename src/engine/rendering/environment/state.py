"""Immutable typed environment inputs for render backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RenderEnvironmentPortal:
    portal_id: str
    kind: str
    side: str
    center_x: float
    center_z: float
    width: float
    depth: float
    side_fade: float
    factor: float


@dataclass(frozen=True, slots=True)
class RenderEnvironmentRegion:
    volume_id: str
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    indoor_factor: float
    portals: tuple[RenderEnvironmentPortal, ...] = ()


@dataclass(frozen=True, slots=True)
class RenderEnvironmentSnapshot:
    """Complete immutable environment-region input for one render frame."""

    regions: tuple[RenderEnvironmentRegion, ...] = ()
