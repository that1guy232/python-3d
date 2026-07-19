"""Declarative world content for the game layer.

The runtime still builds meshes, lighting, roads, and entities, but this module
owns the "what is in this scene?" side of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from collections.abc import Iterable, Sequence
from typing import Any

from pygame.math import Vector3

from game.world.interior_layout import (
    create_building_interior_layout,
    exterior_partition_blocked_ranges,
)
from game.world.objects import Door, Window

DEFAULT_BUILDING_HEIGHT = 66.0

_BUILDING_FEATURE_SIDES = ("north", "east", "south", "west")
_FEATURE_WALL_MARGIN = 18.0
_WINDOW_WALL_MIN_SPAN = 130.0
_WINDOW_WALL_SPACING = 190.0
_WINDOW_WALL_MAX_COUNT = 2
_WINDOW_WALL_SKIP_CHANCE = 0.3
_WINDOW_BUILDING_SPACING = 280.0
_WINDOW_BUILDING_MAX_COUNT = 5
_WINDOW_FEATURE_CLEARANCE = 16.0
_TORCH_WALL_MIN_SPAN = 150.0
_TORCH_WALL_SPACING = 220.0
_TORCH_WALL_MAX_COUNT = 1
_TORCH_WALL_SKIP_CHANCE = 0.6
_TORCH_BUILDING_SPACING = 520.0
_TORCH_BUILDING_MAX_COUNT = 3
_TORCH_FEATURE_WIDTH = 18.0
_TORCH_FEATURE_CLEARANCE = 18.0


def _coerce_vector3(
    value: Vector3 | Sequence[float],
    *,
    default_y: float = 0.0,
) -> Vector3:
    if isinstance(value, Vector3):
        return value.copy()
    if len(value) == 2:
        return Vector3(float(value[0]), float(default_y), float(value[1]))
    return Vector3(float(value[0]), float(value[1]), float(value[2]))


def _copy_feature_specs(features: Iterable[dict[str, Any]] | None) -> tuple[dict, ...]:
    return tuple(dict(feature) for feature in (features or ()))


@dataclass(frozen=True)
class BuildingSpec:
    """Game-facing declaration for one building."""

    position: Vector3 | Sequence[float]
    width: float = 180.0
    depth: float = 140.0
    height: float = DEFAULT_BUILDING_HEIGHT
    doorway_side: str = "south"
    doorway_width: float = Door.DEFAULT_WIDTH
    doorway_height: float = Door.DEFAULT_HEIGHT
    windows: tuple[dict, ...] = field(default_factory=tuple)
    torches: tuple[dict, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _coerce_vector3(self.position))
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "depth", float(self.depth))
        object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "doorway_side", str(self.doorway_side).lower())
        object.__setattr__(self, "doorway_width", float(self.doorway_width))
        object.__setattr__(self, "doorway_height", float(self.doorway_height))
        object.__setattr__(self, "windows", _copy_feature_specs(self.windows))
        object.__setattr__(self, "torches", _copy_feature_specs(self.torches))

    def to_runtime_dict(self) -> dict[str, Any]:
        """Return a mutable dict for the current mesh/entity builders."""
        return {
            "position": self.position.copy(),
            "width": float(self.width),
            "depth": float(self.depth),
            "height": float(self.height),
            "doorway_side": self.doorway_side,
            "doorway_width": float(self.doorway_width),
            "doorway_height": float(self.doorway_height),
            "windows": [dict(window) for window in self.windows],
            "torches": [dict(torch) for torch in self.torches],
        }


@dataclass(frozen=True)
class WorldContent:
    """High-level game content consumed by the world build pipeline."""

    buildings: tuple[BuildingSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "buildings",
            tuple(as_building_spec(building) for building in self.buildings),
        )

    @classmethod
    def with_buildings(
        cls,
        buildings: Iterable[BuildingSpec | dict[str, Any]],
    ) -> "WorldContent":
        return cls(
            buildings=tuple(as_building_spec(building) for building in buildings)
        )

    def to_building_specs(self) -> list[dict[str, Any]]:
        return [building.to_runtime_dict() for building in self.buildings]


def building(
    position: Vector3 | Sequence[float],
    *,
    width: float = 180.0,
    depth: float = 140.0,
    height: float = DEFAULT_BUILDING_HEIGHT,
    doorway_side: str = "south",
    doorway_width: float = Door.DEFAULT_WIDTH,
    doorway_height: float = Door.DEFAULT_HEIGHT,
    windows: Iterable[dict[str, Any]] | None = None,
    torches: Iterable[dict[str, Any]] | None = None,
) -> BuildingSpec:
    """Convenience helper for hand-authored scenes."""
    return BuildingSpec(
        position=position,
        width=width,
        depth=depth,
        height=height,
        doorway_side=doorway_side,
        doorway_width=doorway_width,
        doorway_height=doorway_height,
        windows=_copy_feature_specs(windows),
        torches=_copy_feature_specs(torches),
    )


def as_building_spec(value: BuildingSpec | dict[str, Any]) -> BuildingSpec:
    if isinstance(value, BuildingSpec):
        return value
    if isinstance(value, dict):
        return BuildingSpec(
            position=value["position"],
            width=value.get("width", 180.0),
            depth=value.get("depth", 140.0),
            height=value.get("height", DEFAULT_BUILDING_HEIGHT),
            doorway_side=value.get("doorway_side", "south"),
            doorway_width=value.get("doorway_width", Door.DEFAULT_WIDTH),
            doorway_height=value.get("doorway_height", Door.DEFAULT_HEIGHT),
            windows=_copy_feature_specs(value.get("windows", ())),
            torches=_copy_feature_specs(value.get("torches", ())),
        )
    raise TypeError(f"Unsupported building declaration: {type(value)!r}")


def _wall_span_for_side(width: float, depth: float, side: str) -> float:
    side_key = str(side).lower()
    return float(width if side_key in {"north", "south"} else depth)


def _feature_count_for_wall(
    rng: random.Random,
    span: float,
    *,
    min_span: float,
    spacing: float,
    max_count: int,
    skip_chance: float,
) -> int:
    if span < min_span:
        return 0

    exact = 1.0 + max(0.0, span - min_span) / max(1.0, spacing)
    count = int(exact)
    if count < max_count and rng.random() < exact - count:
        count += 1
    if count > 0 and rng.random() < skip_chance:
        return 0
    return max(0, min(max_count, count))


def _feature_count_for_building(
    rng: random.Random,
    *,
    width: float,
    depth: float,
    spacing: float,
    max_count: int,
) -> int:
    perimeter = max(0.0, (float(width) + float(depth)) * 2.0)
    exact = max(1.0, perimeter / max(1.0, spacing))
    count = int(exact)
    if count < max_count and rng.random() < exact - count:
        count += 1
    return max(1, min(max_count, count))


def _limit_feature_specs(
    rng: random.Random,
    features: list[dict],
    max_count: int,
) -> list[dict]:
    if len(features) <= max_count:
        return features
    limited = list(features)
    rng.shuffle(limited)
    return limited[:max_count]


def _door_blocked_ranges_for_side(
    side: str,
    doorway_side: str,
    doorway_width: float,
) -> list[tuple[float, float]]:
    if str(side).lower() != str(doorway_side).lower():
        return []
    half_width = max(0.0, float(doorway_width)) * 0.5
    return [(-half_width, half_width)]


def _feature_center_segments_for_wall(
    span: float,
    feature_width: float,
    blocked_ranges: list[tuple[float, float]],
    *,
    wall_margin: float,
    clearance: float,
) -> list[tuple[float, float]]:
    half_span = max(0.0, float(span)) * 0.5
    half_feature = max(0.0, float(feature_width)) * 0.5
    center_min = -half_span + max(0.0, wall_margin) + half_feature
    center_max = half_span - max(0.0, wall_margin) - half_feature
    if center_max < center_min:
        return []

    segments = [(center_min, center_max)]
    for blocked_min, blocked_max in blocked_ranges:
        if blocked_max < blocked_min:
            blocked_min, blocked_max = blocked_max, blocked_min
        blocked_min -= half_feature + max(0.0, clearance)
        blocked_max += half_feature + max(0.0, clearance)

        next_segments: list[tuple[float, float]] = []
        for segment_min, segment_max in segments:
            if blocked_max <= segment_min or blocked_min >= segment_max:
                next_segments.append((segment_min, segment_max))
                continue
            if blocked_min > segment_min:
                next_segments.append((segment_min, min(segment_max, blocked_min)))
            if blocked_max < segment_max:
                next_segments.append((max(segment_min, blocked_max), segment_max))
        segments = [
            (segment_min, segment_max)
            for segment_min, segment_max in next_segments
            if segment_max >= segment_min
        ]
        if not segments:
            break

    return segments


def _pick_feature_offset(
    rng: random.Random,
    segments: list[tuple[float, float]],
) -> float | None:
    total = sum(
        max(0.0, segment_max - segment_min) for segment_min, segment_max in segments
    )
    if total <= 0.0:
        return None

    choice = rng.uniform(0.0, total)
    for segment_min, segment_max in segments:
        length = max(0.0, segment_max - segment_min)
        if choice <= length:
            return segment_min + choice
        choice -= length
    return segments[-1][1]


def _random_feature_offsets_for_wall(
    rng: random.Random,
    span: float,
    count: int,
    feature_width: float,
    blocked_ranges: list[tuple[float, float]],
    *,
    wall_margin: float = _FEATURE_WALL_MARGIN,
    clearance: float,
) -> list[float]:
    offsets: list[float] = []
    occupied = list(blocked_ranges)
    for _ in range(max(0, int(count))):
        segments = _feature_center_segments_for_wall(
            span,
            feature_width,
            occupied,
            wall_margin=wall_margin,
            clearance=clearance,
        )
        offset = _pick_feature_offset(rng, segments)
        if offset is None:
            break
        offsets.append(offset)
        half_width = feature_width * 0.5
        occupied.append((offset - half_width, offset + half_width))
    return offsets


def _random_window_specs_for_building(
    rng: random.Random,
    *,
    width: float,
    depth: float,
    doorway_side: str,
    doorway_width: float,
    partition_ranges_by_side: dict[str, list[tuple[float, float]]] | None = None,
) -> list[dict]:
    windows: list[dict] = []
    sides = list(_BUILDING_FEATURE_SIDES)
    rng.shuffle(sides)

    for side in sides:
        span = _wall_span_for_side(width, depth, side)
        count = _feature_count_for_wall(
            rng,
            span,
            min_span=_WINDOW_WALL_MIN_SPAN,
            spacing=_WINDOW_WALL_SPACING,
            max_count=_WINDOW_WALL_MAX_COUNT,
            skip_chance=_WINDOW_WALL_SKIP_CHANCE,
        )
        blocked_ranges = _door_blocked_ranges_for_side(
            side,
            doorway_side,
            doorway_width,
        )
        blocked_ranges.extend((partition_ranges_by_side or {}).get(side, ()))
        offsets = _random_feature_offsets_for_wall(
            rng,
            span,
            count,
            Window.DEFAULT_WIDTH,
            blocked_ranges,
            clearance=_WINDOW_FEATURE_CLEARANCE,
        )
        for offset in offsets:
            windows.append(
                {
                    "side": side,
                    "offset": offset,
                    "width": Window.DEFAULT_WIDTH,
                    "height": Window.DEFAULT_HEIGHT,
                    "sill_height": Window.DEFAULT_SILL_HEIGHT,
                }
            )

    if windows:
        max_windows = _feature_count_for_building(
            rng,
            width=width,
            depth=depth,
            spacing=_WINDOW_BUILDING_SPACING,
            max_count=_WINDOW_BUILDING_MAX_COUNT,
        )
        return _limit_feature_specs(rng, windows, max_windows)

    fallback_sides = sorted(
        _BUILDING_FEATURE_SIDES,
        key=lambda side: _wall_span_for_side(width, depth, side),
        reverse=True,
    )
    for side in fallback_sides:
        span = _wall_span_for_side(width, depth, side)
        blocked_ranges = _door_blocked_ranges_for_side(
            side,
            doorway_side,
            doorway_width,
        )
        blocked_ranges.extend((partition_ranges_by_side or {}).get(side, ()))
        offsets = _random_feature_offsets_for_wall(
            rng,
            span,
            1,
            Window.DEFAULT_WIDTH,
            blocked_ranges,
            clearance=_WINDOW_FEATURE_CLEARANCE,
        )
        if not offsets:
            continue
        return [
            {
                "side": side,
                "offset": offsets[0],
                "width": Window.DEFAULT_WIDTH,
                "height": Window.DEFAULT_HEIGHT,
                "sill_height": Window.DEFAULT_SILL_HEIGHT,
            }
        ]
    return []


def _window_ranges_by_side(windows: list[dict]) -> dict[str, list[tuple[float, float]]]:
    ranges_by_side = {side: [] for side in _BUILDING_FEATURE_SIDES}
    for window in windows:
        try:
            side = str(window.get("side", "")).lower()
            offset = float(window.get("offset", 0.0))
            width = max(1.0, float(window.get("width", Window.DEFAULT_WIDTH)))
        except (TypeError, ValueError, AttributeError):
            continue
        if side not in ranges_by_side:
            continue
        half_width = width * 0.5
        ranges_by_side[side].append((offset - half_width, offset + half_width))
    return ranges_by_side


def _random_torch_specs_for_building(
    rng: random.Random,
    *,
    width: float,
    depth: float,
    doorway_side: str,
    doorway_width: float,
    windows: list[dict],
    partition_ranges_by_side: dict[str, list[tuple[float, float]]] | None = None,
) -> list[dict]:
    torches: list[dict] = []
    windows_by_side = _window_ranges_by_side(windows)
    sides = list(_BUILDING_FEATURE_SIDES)
    rng.shuffle(sides)

    for side in sides:
        span = _wall_span_for_side(width, depth, side)
        count = _feature_count_for_wall(
            rng,
            span,
            min_span=_TORCH_WALL_MIN_SPAN,
            spacing=_TORCH_WALL_SPACING,
            max_count=_TORCH_WALL_MAX_COUNT,
            skip_chance=_TORCH_WALL_SKIP_CHANCE,
        )
        blocked_ranges = _door_blocked_ranges_for_side(
            side,
            doorway_side,
            doorway_width,
        )
        blocked_ranges.extend(windows_by_side.get(side, ()))
        blocked_ranges.extend((partition_ranges_by_side or {}).get(side, ()))
        offsets = _random_feature_offsets_for_wall(
            rng,
            span,
            count,
            _TORCH_FEATURE_WIDTH,
            blocked_ranges,
            clearance=_TORCH_FEATURE_CLEARANCE,
        )
        for offset in offsets:
            torches.append({"side": side, "offset": offset})

    if torches:
        max_torches = _feature_count_for_building(
            rng,
            width=width,
            depth=depth,
            spacing=_TORCH_BUILDING_SPACING,
            max_count=_TORCH_BUILDING_MAX_COUNT,
        )
        return _limit_feature_specs(rng, torches, max_torches)

    fallback_sides = sorted(
        _BUILDING_FEATURE_SIDES,
        key=lambda side: _wall_span_for_side(width, depth, side),
        reverse=True,
    )
    for side in fallback_sides:
        span = _wall_span_for_side(width, depth, side)
        blocked_ranges = _door_blocked_ranges_for_side(
            side,
            doorway_side,
            doorway_width,
        )
        blocked_ranges.extend(windows_by_side.get(side, ()))
        blocked_ranges.extend((partition_ranges_by_side or {}).get(side, ()))
        offsets = _random_feature_offsets_for_wall(
            rng,
            span,
            1,
            _TORCH_FEATURE_WIDTH,
            blocked_ranges,
            clearance=_TORCH_FEATURE_CLEARANCE,
        )
        if offsets:
            return [{"side": side, "offset": offsets[0]}]
    return []


def create_world_content(
    scene,
    *,
    building_count: int = 10,
    rng: random.Random | None = None,
) -> WorldContent:
    rng = rng or random.Random()
    min_x, max_x, min_z, max_z = scene.ground_bounds
    map_width = max_x - min_x
    map_depth = max_z - min_z
    cols = max(1, math.ceil(math.sqrt(building_count)))
    rows = max(1, math.ceil(building_count / cols))
    cell_width = map_width / cols
    cell_depth = map_depth / rows
    cells = [(col, row) for row in range(rows) for col in range(cols)]
    rng.shuffle(cells)

    buildings: list[BuildingSpec] = []
    center_z = (min_z + max_z) * 0.5
    doorway_sides = ("north", "east", "south", "west")

    for col, row in cells[:building_count]:
        width = rng.uniform(140.0, 420.0)
        depth = rng.uniform(100.0, 260.0)
        doorway_side = rng.choice(doorway_sides)
        doorway_width = Door.DEFAULT_WIDTH
        doorway_height = Door.DEFAULT_HEIGHT

        cell_min_x = min_x + col * cell_width
        cell_max_x = min_x + (col + 1) * cell_width
        cell_min_z = min_z + row * cell_depth
        cell_max_z = min_z + (row + 1) * cell_depth

        x = rng.uniform(
            cell_min_x + cell_width * 0.2,
            cell_max_x - cell_width * 0.2,
        )
        z = rng.uniform(
            cell_min_z + cell_depth * 0.2,
            cell_max_z - cell_depth * 0.2,
        )

        road_clearance = depth * 0.5 + 110.0
        if abs(z - center_z) < road_clearance:
            z = center_z + math.copysign(road_clearance, z - center_z or 1.0)

        driveway_spawn_margin = 95.0
        x_margin = width * 0.5 + driveway_spawn_margin
        z_margin = depth * 0.5 + driveway_spawn_margin
        x = max(min_x + x_margin, min(max_x - x_margin, x))
        z = max(min_z + z_margin, min(max_z - z_margin, z))

        draft_layout = create_building_interior_layout(
            {
                "width": width,
                "depth": depth,
                "height": DEFAULT_BUILDING_HEIGHT,
                "doorway_side": doorway_side,
                "doorway_width": doorway_width,
                "doorway_height": doorway_height,
                "wall_thickness": 2.5,
            }
        )
        partition_ranges = exterior_partition_blocked_ranges(
            draft_layout,
            wall_thickness=2.5,
        )

        windows = _random_window_specs_for_building(
            rng,
            width=width,
            depth=depth,
            doorway_side=doorway_side,
            doorway_width=doorway_width,
            partition_ranges_by_side=partition_ranges,
        )
        torches = _random_torch_specs_for_building(
            rng,
            width=width,
            depth=depth,
            doorway_side=doorway_side,
            doorway_width=doorway_width,
            windows=windows,
            partition_ranges_by_side=partition_ranges,
        )

        buildings.append(
            building(
                Vector3(x, 0.0, z),
                width=width,
                depth=depth,
                height=DEFAULT_BUILDING_HEIGHT,
                doorway_side=doorway_side,
                doorway_width=doorway_width,
                doorway_height=doorway_height,
                windows=windows,
                torches=torches,
            )
        )

    return WorldContent(buildings=tuple(buildings))


def resolve_world_content(
    scene,
    *,
    building_count: int = 10,
) -> WorldContent:
    content = getattr(scene, "world_content", None)
    if content is None:
        content = create_world_content(
            scene,
            building_count=building_count,
            rng=random.Random(getattr(scene, "world_random_seed", None)),
        )
        scene.world_content = content
        return content
    if isinstance(content, WorldContent):
        return content
    content = WorldContent.with_buildings(getattr(content, "buildings", content))
    scene.world_content = content
    return content
