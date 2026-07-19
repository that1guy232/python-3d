"""Typed indoor-volume and portal state shared outside the renderer.

This is the first migration boundary away from lighting-owned ``dict`` values.
The legacy renderer still consumes covered-region dictionaries, so the typed
model can project a compatibility view while gameplay and audio use the model
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from engine.rendering.render_environment import (
    RenderEnvironmentPortal,
    RenderEnvironmentRegion,
    RenderEnvironmentSnapshot,
)


_VALID_PORTAL_SIDES = frozenset(("north", "east", "south", "west"))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _smooth01(value: float) -> float:
    value = _clamp01(value)
    return value * value * (3.0 - 2.0 * value)


@dataclass(slots=True)
class EnvironmentPortal:
    """An opening that blends an indoor volume toward an exterior factor."""

    portal_id: str
    kind: str
    side: str
    center_x: float
    center_z: float
    width: float
    depth: float
    side_fade: float
    closed_factor: float
    open_factor: float
    openness: float = 0.0

    def __post_init__(self) -> None:
        self.portal_id = str(self.portal_id)
        self.kind = str(self.kind).lower()
        self.side = str(self.side).lower()
        if self.side not in _VALID_PORTAL_SIDES:
            raise ValueError(f"invalid environment portal side: {self.side!r}")
        self.center_x = float(self.center_x)
        self.center_z = float(self.center_z)
        self.width = max(1.0, float(self.width))
        self.depth = max(1.0, float(self.depth))
        self.side_fade = max(1.0, float(self.side_fade))
        self.closed_factor = _clamp01(self.closed_factor)
        self.open_factor = _clamp01(self.open_factor)
        self.openness = _clamp01(self.openness)

    @property
    def factor(self) -> float:
        amount = _smooth01(self.openness)
        return self.closed_factor + (self.open_factor - self.closed_factor) * amount

    def set_openness(self, value: float) -> float:
        """Update portal state and return its current edge-light factor."""

        self.openness = _clamp01(value)
        return self.factor

    def factor_at(self, volume: "EnvironmentVolume", x: float, z: float) -> float:
        """Return this portal's influence at one point inside ``volume``."""

        px = float(x)
        pz = float(z)
        if self.side == "north":
            inward_depth = volume.max_z - pz
            lateral = px - self.center_x
        elif self.side == "south":
            inward_depth = pz - volume.min_z
            lateral = px - self.center_x
        elif self.side == "east":
            inward_depth = volume.max_x - px
            lateral = pz - self.center_z
        else:  # west
            inward_depth = px - volume.min_x
            lateral = pz - self.center_z

        if inward_depth < 0.0 or inward_depth > self.depth:
            return volume.indoor_factor

        half_width = self.width * 0.5
        lateral_abs = abs(lateral)
        if lateral_abs >= half_width + self.side_fade:
            return volume.indoor_factor

        if lateral_abs <= half_width:
            width_influence = 1.0
        else:
            width_influence = 1.0 - _smooth01(
                (lateral_abs - half_width) / self.side_fade
            )
        depth_influence = 1.0 - _smooth01(inward_depth / self.depth)
        influence = _clamp01(width_influence * depth_influence)
        return volume.indoor_factor + (self.factor - volume.indoor_factor) * influence

    def to_legacy_dict(self) -> dict[str, object]:
        """Project this portal into the current covered-region contract."""

        result: dict[str, object] = {
            "portal_id": self.portal_id,
            "type": self.kind,
            "side": self.side,
            "center_x": self.center_x,
            "center_z": self.center_z,
            "width": self.width,
            "depth": self.depth,
            "side_fade": self.side_fade,
            "edge_factor": self.factor,
        }
        if self.kind == "doorway":
            result["closed_edge_factor"] = self.closed_factor
            result["open_edge_factor"] = self.open_factor
        return result


@dataclass(slots=True)
class EnvironmentVolume:
    """A typed indoor X/Z footprint with stable identity and portals."""

    volume_id: str
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    indoor_factor: float
    portals: tuple[EnvironmentPortal, ...] = ()

    def __post_init__(self) -> None:
        self.volume_id = str(self.volume_id)
        self.min_x = float(self.min_x)
        self.max_x = float(self.max_x)
        self.min_z = float(self.min_z)
        self.max_z = float(self.max_z)
        if self.max_x < self.min_x:
            self.min_x, self.max_x = self.max_x, self.min_x
        if self.max_z < self.min_z:
            self.min_z, self.max_z = self.max_z, self.min_z
        self.indoor_factor = _clamp01(self.indoor_factor)
        self.portals = tuple(self.portals)

    @property
    def doorway(self) -> EnvironmentPortal | None:
        return next(
            (portal for portal in self.portals if portal.kind == "doorway"),
            None,
        )

    def contains(self, x: float, z: float) -> bool:
        px = float(x)
        pz = float(z)
        return self.min_x <= px <= self.max_x and self.min_z <= pz <= self.max_z

    def factor_at(self, x: float, z: float) -> float:
        if not self.contains(x, z):
            return 1.0
        factor = self.indoor_factor
        for portal in self.portals:
            factor = max(factor, portal.factor_at(self, x, z))
        return factor

    def to_legacy_dict(self) -> dict[str, object]:
        """Project one volume while preserving shared opening-dict identity."""

        opening_values = [portal.to_legacy_dict() for portal in self.portals]
        doorway = next(
            (opening for opening in opening_values if opening.get("type") == "doorway"),
            None,
        )
        windows = [
            opening for opening in opening_values if opening.get("type") == "window"
        ]
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_z": self.min_z,
            "max_z": self.max_z,
            "factor": self.indoor_factor,
            "doorway": doorway,
            "windows": windows,
            "openings": opening_values,
        }


def environment_factor_at(
    x: float,
    z: float,
    *,
    volumes: Iterable[EnvironmentVolume] | None,
) -> float:
    """Return the darkest containing volume, including portal influence."""

    factor = 1.0
    for volume in volumes or ():
        if volume.contains(x, z):
            factor = min(factor, volume.factor_at(x, z))
    return factor


def environment_render_snapshot(
    volumes: Iterable[EnvironmentVolume] | None,
) -> RenderEnvironmentSnapshot:
    """Project typed world environment state into immutable render records."""

    return RenderEnvironmentSnapshot(
        regions=tuple(
            RenderEnvironmentRegion(
                volume_id=volume.volume_id,
                min_x=volume.min_x,
                max_x=volume.max_x,
                min_z=volume.min_z,
                max_z=volume.max_z,
                indoor_factor=volume.indoor_factor,
                portals=tuple(
                    RenderEnvironmentPortal(
                        portal_id=portal.portal_id,
                        kind=portal.kind,
                        side=portal.side,
                        center_x=portal.center_x,
                        center_z=portal.center_z,
                        width=portal.width,
                        depth=portal.depth,
                        side_fade=portal.side_fade,
                        factor=portal.factor,
                    )
                    for portal in volume.portals
                ),
            )
            for volume in volumes or ()
        )
    )
