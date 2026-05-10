"""World object construction for WorldScene."""

from __future__ import annotations

import math
import random
import time
from typing import Iterator

from pygame.math import Vector3

from config import (
    GOBLIN_COUNT,
    GOBLIN_MIN_SEPARATION,
    GOBLIN_SPAWN_ATTEMPTS,
    GOBLIN_SPAWN_CLEARANCE,
    GOBLIN_SPAWN_TREE_RADIUS,
)
from engine.rendering.decal import Decal
from engine.rendering.decal_batch import DecalBatch
from engine.rendering.lighting import INDOOR_LIGHT_FACTOR
from engine.rendering.sprite import WorldSprite
from textures.resource_path import WALL1_TEXTURE_PATH
from textures.texture_utils import (
    create_shadow_texture,
    create_tree_shadow_texture,
    load_texture,
)
from world.objects import Door, Goblin, Road, Torch, Window
from world.objects.building import Building
from world.objects.door import build_door_render_batch
from world.objects.fence import build_textured_fence_ring
from world.objects.ground import TexturedGroundGridBuilder
from world.objects.polygon import Polygon, build_polygon_render_batch
from world.objects.road import build_road_render_batch
from world.objects.window import build_window_render_batch
from world.objects.wall_tile import build_wall_tile_batches
from world.world_road_planner import create_building_access_roads
from world.world_spawner import spawn_world_sprites


CREATE_WORLD_OBJECT_STEPS = 11
BUILDING_HEIGHT = 66.0
BUILDING_ROOF_OVERHANG = 6.0
BUILDING_WALL_TERRAIN_EMBED_DEPTH = 8.0
BUILDING_WALL_TERRAIN_SAMPLE_SPACING = 18.0
SHADOW_BUILDING_CLIP_MARGIN = 2.0
WINDOW_LIGHT_EDGE_FACTOR = 0.86
DOORWAY_WALL_LIGHT_VALUE = 1.0
DOORWAY_WALL_SPLASH_VALUE = 1.36
DOORWAY_WALL_LIGHT_FALLOFF = 1.65
DOORWAY_WALL_LIGHT_MIN_RADIUS = 54.0
DOORWAY_WALL_LIGHT_MAX_RADIUS = 118.0
WINDOW_WALL_LIGHT_VALUE = 0.96
WINDOW_WALL_SPLASH_VALUE = 1.26
WINDOW_WALL_LIGHT_FALLOFF = 1.9
WINDOW_WALL_LIGHT_MIN_RADIUS = 42.0
WINDOW_WALL_LIGHT_MAX_RADIUS = 96.0
OPENING_WALL_LIGHT_INSET = 10.0
OPENING_WALL_LIGHT_BOUNDS_INSET = 2.0
OPENING_WALL_LIGHT_BAND_DEPTH = 18.0
OPENING_WALL_LIGHT_LATERAL_PAD = 28.0
OPENING_WALL_LIGHT_FLOOR_SCALE = 0.0

_SIDE_NORMALS = {
    "north": (0.0, 1.0),
    "east": (1.0, 0.0),
    "south": (0.0, -1.0),
    "west": (-1.0, 0.0),
}
_OPPOSITE_SIDES = {
    "north": "south",
    "east": "west",
    "south": "north",
    "west": "east",
}
_WINDOW_SIDE_BY_DOORWAY = {
    "north": "east",
    "east": "south",
    "south": "west",
    "west": "north",
}
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
        max(0.0, segment_max - segment_min)
        for segment_min, segment_max in segments
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


def _dispose_value(obj) -> None:
    dispose = getattr(obj, "dispose", None)
    if callable(dispose):
        try:
            dispose()
        except Exception:
            pass


def _dispose_values(values) -> None:
    for value in values or ():
        _dispose_value(value)


def _build_road_batches(scene) -> None:
    _dispose_values(getattr(scene, "road_batches", ()))
    roads = getattr(scene, "roads", ()) or ()
    for road in roads:
        setattr(road, "render_batched", False)
    road_batch = build_road_render_batch(roads)
    scene.road_batches = [road_batch] if road_batch is not None else []


def _opening_light_center(spec: dict, side: str, offset: float) -> tuple[float, float]:
    position = spec["position"]
    x = float(position.x)
    z = float(position.z)
    side_key = str(side).lower()
    if side_key in {"north", "south"}:
        return x + float(offset), z
    if side_key in {"east", "west"}:
        return x, z + float(offset)
    return x, z


def _window_light_opening(spec: dict, window: dict) -> dict | None:
    try:
        side = str(window.get("side", "north")).lower()
        width = max(1.0, float(window.get("width", Window.DEFAULT_WIDTH)))
        offset = float(window.get("offset", 0.0))
        half_x = float(spec["width"]) * 0.5
        half_z = float(spec["depth"]) * 0.5
    except (KeyError, TypeError, ValueError, AttributeError):
        return None

    center_x, center_z = _opening_light_center(spec, side, offset)
    depth = max(36.0, min(86.0, min(half_x, half_z) * 0.58))
    return {
        "type": "window",
        "side": side,
        "center_x": center_x,
        "center_z": center_z,
        "width": max(width * 1.4, width + 10.0),
        "depth": depth,
        "side_fade": max(8.0, width * 0.42),
        "edge_factor": WINDOW_LIGHT_EDGE_FACTOR,
    }


def _building_covered_regions(scene) -> list[object]:
    regions: list[dict] = []
    for spec in getattr(scene, "building_specs", ()) or ():
        try:
            position = spec["position"]
            half_x = float(spec["width"]) * 0.5
            half_z = float(spec["depth"]) * 0.5
            x = float(position.x)
            z = float(position.z)
            side = str(spec.get("doorway_side", "south")).lower()
            doorway_width = float(spec.get("doorway_width", 48.0))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        min_x = x - half_x
        max_x = x + half_x
        min_z = z - half_z
        max_z = z + half_z
        doorway_depth = max(42.0, min(78.0, min(half_x, half_z) * 0.78))
        indoor_factor = INDOOR_LIGHT_FACTOR
        doorway = {
            "type": "doorway",
            "side": side,
            "center_x": x,
            "center_z": z,
            "width": max(doorway_width * 1.16, doorway_width + 8.0),
            "depth": doorway_depth,
            "side_fade": max(10.0, doorway_width * 0.26),
            "edge_factor": indoor_factor,
            "closed_edge_factor": indoor_factor,
            "open_edge_factor": 1.0,
        }
        windows = [
            opening
            for opening in (
                _window_light_opening(spec, window)
                for window in (spec.get("windows", ()) or ())
            )
            if opening is not None
        ]
        regions.append(
            {
                "min_x": min_x,
                "max_x": max_x,
                "min_z": min_z,
                "max_z": max_z,
                "factor": indoor_factor,
                "doorway": doorway,
                "windows": windows,
                "openings": [doorway, *windows],
            }
        )
    return regions


def _building_shadow_clip_bounds(
    scene,
) -> list[tuple[float, float, float, float]]:
    bounds: list[tuple[float, float, float, float]] = []
    margin = SHADOW_BUILDING_CLIP_MARGIN
    for building in getattr(scene, "buildings", ()) or ():
        try:
            min_x, max_x, min_z, max_z = building.bounds
        except (TypeError, ValueError, AttributeError):
            continue
        bounds.append(
            (
                float(min_x) - margin,
                float(max_x) + margin,
                float(min_z) - margin,
                float(max_z) + margin,
            )
        )
    return bounds


def _outside_building_shadow_receiver(
    bounds: list[tuple[float, float, float, float]],
):
    if not bounds:
        return None

    def receives_shadow(x: float, z: float) -> bool:
        px = float(x)
        pz = float(z)
        for min_x, max_x, min_z, max_z in bounds:
            if min_x <= px <= max_x and min_z <= pz <= max_z:
                return False
        return True

    return receives_shadow


def _opening_wall_light_center(
    region: dict,
    opening: dict,
) -> tuple[float, float]:
    side = str(opening.get("side", "")).lower()
    min_x = float(region["min_x"])
    max_x = float(region["max_x"])
    min_z = float(region["min_z"])
    max_z = float(region["max_z"])
    center_x = float(opening.get("center_x", (min_x + max_x) * 0.5))
    center_z = float(opening.get("center_z", (min_z + max_z) * 0.5))
    inset = max(0.0, float(opening.get("wall_light_inset", OPENING_WALL_LIGHT_INSET)))

    if side == "north":
        return center_x, max_z - inset
    if side == "south":
        return center_x, min_z + inset
    if side == "east":
        return max_x - inset, center_z
    if side == "west":
        return min_x + inset, center_z
    return center_x, center_z


def _opening_wall_light_bounds(
    region: dict,
    opening: dict,
    *,
    radius: float | None = None,
) -> tuple[float, float, float, float]:
    inset = OPENING_WALL_LIGHT_BOUNDS_INSET
    min_x = float(region["min_x"]) + inset
    max_x = float(region["max_x"]) - inset
    min_z = float(region["min_z"]) + inset
    max_z = float(region["max_z"]) - inset
    if max_x < min_x:
        min_x, max_x = max_x, min_x
    if max_z < min_z:
        min_z, max_z = max_z, min_z

    side = str(opening.get("side", "")).lower()
    center_x = float(opening.get("center_x", (min_x + max_x) * 0.5))
    center_z = float(opening.get("center_z", (min_z + max_z) * 0.5))
    width = max(1.0, float(opening.get("width", 48.0)))
    side_fade = max(1.0, float(opening.get("side_fade", width * 0.25)))
    influence_radius = max(0.0, float(radius)) if radius is not None else 0.0
    wall_light_inset = max(
        0.0,
        float(opening.get("wall_light_inset", OPENING_WALL_LIGHT_INSET)),
    )
    lateral_half = max(
        width * 0.5 + side_fade + OPENING_WALL_LIGHT_LATERAL_PAD,
        influence_radius,
    )
    band_depth = max(
        4.0,
        float(OPENING_WALL_LIGHT_BAND_DEPTH),
        influence_radius + wall_light_inset,
    )

    if side == "north":
        return (
            max(min_x, center_x - lateral_half),
            min(max_x, center_x + lateral_half),
            max(min_z, max_z - band_depth),
            max_z,
        )
    if side == "south":
        return (
            max(min_x, center_x - lateral_half),
            min(max_x, center_x + lateral_half),
            min_z,
            min(max_z, min_z + band_depth),
        )
    if side == "east":
        return (
            max(min_x, max_x - band_depth),
            max_x,
            max(min_z, center_z - lateral_half),
            min(max_z, center_z + lateral_half),
        )
    if side == "west":
        return (
            min_x,
            min(max_x, min_x + band_depth),
            max(min_z, center_z - lateral_half),
            min(max_z, center_z + lateral_half),
        )
    return min_x, max_x, min_z, max_z


def _opening_wall_light_radius(
    opening: dict,
    *,
    min_radius: float,
    max_radius: float,
) -> float:
    width = max(1.0, float(opening.get("width", 48.0)))
    depth = max(1.0, float(opening.get("depth", 64.0)))
    radius = max(width * 1.08, depth * 0.92)
    return max(min_radius, min(max_radius, radius))


def _opening_wall_light_modifier(
    region: dict,
    opening: dict,
) -> dict | None:
    try:
        opening_type = str(opening.get("type", "")).lower()
        x, z = _opening_wall_light_center(region, opening)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None

    if opening_type == "doorway":
        try:
            open_radius = _opening_wall_light_radius(
                opening,
                min_radius=DOORWAY_WALL_LIGHT_MIN_RADIUS,
                max_radius=DOORWAY_WALL_LIGHT_MAX_RADIUS,
            )
            bounds = _opening_wall_light_bounds(region, opening, radius=open_radius)
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return {
            "center": Vector3(x, 0.0, z),
            "radius": 0.0,
            "value": DOORWAY_WALL_SPLASH_VALUE,
            "falloff": DOORWAY_WALL_LIGHT_FALLOFF,
            "bounds": bounds,
            "indoor_only": True,
            "floor_scale": OPENING_WALL_LIGHT_FLOOR_SCALE,
            "opening_type": "doorway",
            "closed_radius": 0.0,
            "open_radius": open_radius,
            "open_value": DOORWAY_WALL_SPLASH_VALUE,
        }

    if opening_type == "window":
        try:
            radius = _opening_wall_light_radius(
                opening,
                min_radius=WINDOW_WALL_LIGHT_MIN_RADIUS,
                max_radius=WINDOW_WALL_LIGHT_MAX_RADIUS,
            )
            bounds = _opening_wall_light_bounds(region, opening, radius=radius)
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return {
            "center": Vector3(x, 0.0, z),
            "radius": radius,
            "value": WINDOW_WALL_SPLASH_VALUE,
            "falloff": WINDOW_WALL_LIGHT_FALLOFF,
            "bounds": bounds,
            "indoor_only": True,
            "floor_scale": OPENING_WALL_LIGHT_FLOOR_SCALE,
            "opening_type": "window",
        }

    return None


def _opening_wall_light_modifiers_for_regions(
    regions,
) -> tuple[list[dict | None], list[dict]]:
    doorway_modifiers: list[dict | None] = []
    window_modifiers: list[dict] = []

    for region in regions or ():
        if not isinstance(region, dict):
            doorway_modifiers.append(None)
            continue

        doorway_modifier = None
        doorway = region.get("doorway")
        if isinstance(doorway, dict):
            doorway_modifier = _opening_wall_light_modifier(region, doorway)
        doorway_modifiers.append(doorway_modifier)

        windows = region.get("windows")
        if isinstance(windows, (list, tuple)):
            for window in windows:
                if not isinstance(window, dict):
                    continue
                modifier = _opening_wall_light_modifier(region, window)
                if modifier is not None:
                    window_modifiers.append(modifier)

    return doorway_modifiers, window_modifiers


def _copy_opening_light_metadata(target: dict, source: dict) -> dict:
    for key in (
        "opening_type",
        "closed_radius",
        "open_radius",
        "open_value",
    ):
        if key in source:
            target[key] = source[key]
    return target


def _install_scene_brightness_modifier(scene, modifier: dict) -> dict | None:
    lighting = getattr(scene, "lighting", None)
    camera = getattr(scene, "camera", None)
    if lighting is not None:
        try:
            return lighting.add_brightness_modifier(modifier, camera=camera)
        except (KeyError, TypeError, ValueError, AttributeError):
            return None

    if not hasattr(scene, "brightness_modifiers") or scene.brightness_modifiers is None:
        scene.brightness_modifiers = []
    scene.brightness_modifiers.append(modifier)
    try:
        Torch.install_brightness_modifier(camera, modifier)
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return modifier


def _install_building_torch_lights(scene) -> None:
    torch_modifiers = Torch.brightness_modifiers_for_building_specs(
        getattr(scene, "building_specs", ()) or ()
    )
    doorway_modifiers_by_region, window_modifiers = (
        _opening_wall_light_modifiers_for_regions(
            getattr(scene, "covered_regions", ()) or ()
        )
    )
    scene.torch_light_modifiers = []
    scene.doorway_light_modifiers_by_region = []
    scene.doorway_light_modifiers = []
    scene.window_light_modifiers = []
    scene.opening_light_modifiers = []

    for modifier in doorway_modifiers_by_region:
        if modifier is None:
            scene.doorway_light_modifiers_by_region.append(None)
            continue
        installed = _install_scene_brightness_modifier(scene, modifier)
        if installed is not None:
            _copy_opening_light_metadata(installed, modifier)
            scene.doorway_light_modifiers_by_region.append(installed)
            scene.doorway_light_modifiers.append(installed)
            scene.opening_light_modifiers.append(installed)
        else:
            scene.doorway_light_modifiers_by_region.append(None)

    for modifier in window_modifiers:
        installed = _install_scene_brightness_modifier(scene, modifier)
        if installed is not None:
            _copy_opening_light_metadata(installed, modifier)
            scene.window_light_modifiers.append(installed)
            scene.opening_light_modifiers.append(installed)

    for modifier in torch_modifiers:
        installed = _install_scene_brightness_modifier(scene, modifier)
        if installed is not None:
            scene.torch_light_modifiers.append(installed)

    lighting = getattr(scene, "lighting", None)
    if lighting is not None:
        sync_aliases = getattr(scene, "_sync_lighting_aliases", None)
        if callable(sync_aliases):
            sync_aliases()
        else:
            scene.brightness_modifiers = lighting.brightness_modifiers
        return


def _build_building_torches(scene) -> None:
    torch_tex = Torch.texture_or_load(getattr(scene, "torch_tex", None))
    scene.torch_tex = torch_tex
    scene.torches = Torch.build_for_brightness_modifiers(
        getattr(scene, "torch_light_modifiers", ()) or (),
        texture=torch_tex,
        camera=scene.camera,
        ground_height_at=scene.ground_height_at,
    )
    scene.sprite_items.extend(scene.torches)


def _build_building_doors(scene) -> None:
    for batch in getattr(scene, "door_batches", ()) or ():
        try:
            batch.dispose()
        except Exception:
            pass
    scene.door_batches = []

    for door in getattr(scene, "doors", ()) or ():
        try:
            if door in scene.entities:
                scene.entities.remove(door)
            for sprite in door.get_sprites() or ():
                if sprite in scene.sprite_items:
                    scene.sprite_items.remove(sprite)
            for mesh in door.get_collision_meshes() or ():
                if mesh in scene.wall_tiles:
                    scene.wall_tiles.remove(mesh)
        except Exception:
            continue

    door_tex = Door.texture_or_load(getattr(scene, "door_tex", None))
    scene.door_tex = door_tex
    scene.doors = []
    add_entity = getattr(scene, "add_entity", None)
    covered_regions = list(getattr(scene, "covered_regions", ()) or ())
    doorway_light_modifiers = list(
        getattr(scene, "doorway_light_modifiers_by_region", ()) or ()
    )
    lighting = getattr(scene, "lighting", None)
    sun_direction = getattr(
        lighting,
        "sun_direction",
        getattr(scene, "sun_direction", None),
    )
    for spec_index, spec in enumerate(getattr(scene, "building_specs", ()) or ()):
        try:
            door = Door.from_building_spec(
                spec,
                texture=door_tex,
                camera=scene.camera,
                ground_height_at=scene.ground_height_at,
                lighting=lighting,
                sun_direction=sun_direction,
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if spec_index < len(covered_regions):
            brightness_modifier = (
                doorway_light_modifiers[spec_index]
                if spec_index < len(doorway_light_modifiers)
                else None
            )
            door.bind_doorway_light(
                covered_regions[spec_index],
                brightness_modifier=brightness_modifier,
            )
        scene.doors.append(door)
        if callable(add_entity):
            add_entity(door)
        else:
            scene.entities.append(door)
            scene.sprite_items.extend(door.get_sprites())
            scene.wall_tiles.extend(door.get_collision_meshes())

    door_batch = build_door_render_batch(scene.doors)
    scene.door_batches = [door_batch] if door_batch is not None else []
    refresh_draw_entities = getattr(scene, "refresh_immediate_entities", None)
    if callable(refresh_draw_entities):
        refresh_draw_entities()


def _normalize_window_specs_for_building(spec: dict) -> list[dict]:
    normalized: list[dict] = []
    try:
        wall_height = float(spec["height"])
    except (KeyError, TypeError, ValueError):
        return normalized

    default_side = _WINDOW_SIDE_BY_DOORWAY.get(
        str(spec.get("doorway_side", "south")).lower(),
        "north",
    )
    max_top = max(4.0, wall_height - 4.0)
    for raw_window in spec.get("windows", ()) or ():
        if not isinstance(raw_window, dict):
            continue

        side = str(raw_window.get("side", default_side)).lower()
        if side not in _SIDE_NORMALS:
            side = default_side

        try:
            sill_height = max(
                0.0,
                float(raw_window.get("sill_height", Window.DEFAULT_SILL_HEIGHT)),
            )
            height_value = raw_window.get("height", None)
            width_value = raw_window.get("width", None)
            if height_value is not None:
                height = max(4.0, float(height_value))
            elif width_value is not None:
                height = max(4.0, float(width_value) / Window.TEXTURE_ASPECT)
            else:
                height = Window.DEFAULT_HEIGHT
            offset = float(raw_window.get("offset", 0.0))
        except (TypeError, ValueError):
            continue

        if sill_height + height > max_top:
            height = max(4.0, max_top - sill_height)
        if sill_height + height > max_top:
            sill_height = max(0.0, max_top - height)
        width = max(4.0, height * Window.TEXTURE_ASPECT)

        try:
            span = float(spec["width"] if side in {"north", "south"} else spec["depth"])
        except (KeyError, TypeError, ValueError):
            span = width
        max_offset = max(0.0, (span - width) * 0.5 - 8.0)
        offset = max(-max_offset, min(max_offset, offset))

        normalized.append(
            {
                "side": side,
                "offset": offset,
                "width": width,
                "height": height,
                "sill_height": sill_height,
            }
        )

    spec["windows"] = normalized
    return normalized


def _build_building_windows(scene) -> None:
    for batch in getattr(scene, "window_batches", ()) or ():
        try:
            batch.dispose()
        except Exception:
            pass
    scene.window_batches = []

    for window in getattr(scene, "windows", ()) or ():
        try:
            if window in scene.entities:
                scene.entities.remove(window)
            for sprite in window.get_sprites() or ():
                if sprite in scene.sprite_items:
                    scene.sprite_items.remove(sprite)
            for mesh in window.get_collision_meshes() or ():
                if mesh in scene.wall_tiles:
                    scene.wall_tiles.remove(mesh)
        except Exception:
            continue

    window_tex = Window.texture_or_load(getattr(scene, "window_tex", None))
    scene.window_tex = window_tex
    scene.windows = []
    add_entity = getattr(scene, "add_entity", None)
    lighting = getattr(scene, "lighting", None)
    sun_direction = getattr(
        lighting,
        "sun_direction",
        getattr(scene, "sun_direction", None),
    )
    for spec in getattr(scene, "building_specs", ()) or ():
        for window_spec in spec.get("windows", ()) or ():
            try:
                window = Window.from_building_spec(
                    spec,
                    window_spec=window_spec,
                    texture=window_tex,
                    backing_texture=getattr(scene, "wall_tex", None),
                    camera=scene.camera,
                    ground_height_at=scene.ground_height_at,
                    lighting=lighting,
                    sun_direction=sun_direction,
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            scene.windows.append(window)
            if callable(add_entity):
                add_entity(window)
            else:
                scene.entities.append(window)
                scene.sprite_items.extend(window.get_sprites())
                scene.wall_tiles.extend(window.get_collision_meshes())

    window_batch = build_window_render_batch(scene.windows)
    scene.window_batches = [window_batch] if window_batch is not None else []
    refresh_draw_entities = getattr(scene, "refresh_immediate_entities", None)
    if callable(refresh_draw_entities):
        refresh_draw_entities()


def create_building_specs(scene, count: int = 10) -> list[dict]:
    rng = random.Random()
    min_x, max_x, min_z, max_z = scene.ground_bounds
    map_width = max_x - min_x
    map_depth = max_z - min_z
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    cell_width = map_width / cols
    cell_depth = map_depth / rows
    cells = [(col, row) for row in range(rows) for col in range(cols)]
    rng.shuffle(cells)

    specs = []
    center_z = (min_z + max_z) * 0.5
    doorway_sides = ("north", "east", "south", "west")

    for col, row in cells[:count]:
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

        windows = _random_window_specs_for_building(
            rng,
            width=width,
            depth=depth,
            doorway_side=doorway_side,
            doorway_width=doorway_width,
        )
        torches = _random_torch_specs_for_building(
            rng,
            width=width,
            depth=depth,
            doorway_side=doorway_side,
            doorway_width=doorway_width,
            windows=windows,
        )

        specs.append(
            {
                "position": Vector3(x, 0, z),
                "width": width,
                "depth": depth,
                "height": BUILDING_HEIGHT,
                "doorway_side": doorway_side,
                "doorway_width": doorway_width,
                "doorway_height": doorway_height,
                "windows": windows,
                "torches": torches,
            }
        )

    return specs


def create_world_objects(
    scene,
    grid_count: int,
    spacing: float,
    half: float,
    grid_tile_size: int,
    grid_gap: int,
    tree_count: int,
    grass_count: int,
    rock_count: int,
) -> None:
    for _label, _finished in create_world_objects_steps(
        scene,
        grid_count,
        spacing,
        half,
        grid_tile_size,
        grid_gap,
        tree_count,
        grass_count,
        rock_count,
    ):
        pass


def create_world_objects_steps(
    scene,
    grid_count: int,
    spacing: float,
    half: float,
    grid_tile_size: int,
    grid_gap: int,
    tree_count: int,
    grass_count: int,
    rock_count: int,
) -> Iterator[tuple[str, bool]]:
    label = "Creating buildings"
    print("Creating buildings...")
    yield (label, False)
    start_time = time.perf_counter()
    scene.buildings: list[Building] = []
    scene.building_specs = create_building_specs(scene, count=10)
    for spec in scene.building_specs:
        _normalize_window_specs_for_building(spec)
    for spec in scene.building_specs:
        scene.buildings.append(Building(position=spec["position"]))
    scene.covered_regions = _building_covered_regions(scene)
    lighting = getattr(scene, "lighting", None)
    if lighting is not None:
        lighting.set_covered_regions(scene.covered_regions)
        sync_aliases = getattr(scene, "_sync_lighting_aliases", None)
        if callable(sync_aliases):
            sync_aliases()
    _install_building_torch_lights(scene)

    scene.builder = TexturedGroundGridBuilder(
        count=grid_count,
        tile_size=grid_tile_size,
        gap=grid_gap,
        texture=scene.ground_tex,
        brightness_modifiers=scene.brightness_modifiers,
        default_brightness=scene.camera.brightness_default,
        lighting=getattr(scene, "lighting", None),
        sun_direction=getattr(scene, "sun_direction", None),
        covered_regions=getattr(scene, "covered_regions", ()),
    )

    scene.log_timing(label, start_time, time.perf_counter())
    yield (label, True)

    label = "Generating ground mesh"
    print("Generating ground mesh...")
    yield (label, False)
    start_time = time.perf_counter()
    _dispose_value(getattr(scene, "ground_mesh", None))
    scene.ground_mesh = scene.builder.build()
    scene._ground_height_sampler = getattr(scene.ground_mesh, "height_sampler", None)
    scene.log_timing(label, start_time, time.perf_counter())
    yield (label, True)

    label = "Building structures"
    yield (label, False)
    _build_buildings(scene)
    yield (label, True)

    label = "Building showcase polygons"
    yield (label, False)
    _build_showcase_polygons(scene)
    rebuild_collision_index = getattr(scene, "rebuild_collision_index", None)
    if callable(rebuild_collision_index):
        rebuild_collision_index()
    yield (label, True)

    yield from _build_roads_and_spawn_sprites_steps(
        scene, tree_count, grass_count, rock_count
    )

    label = "Building fences"
    yield (label, False)
    _build_fences(scene)
    yield (label, True)

    label = "Adding shadows"
    yield (label, False)
    _build_shadow_decals(scene)
    yield (label, True)


def _build_buildings(scene) -> None:
    start_time = time.perf_counter()
    wall_tex = scene.wall_tex or load_texture(WALL1_TEXTURE_PATH)
    scene.wall_tex = wall_tex
    scene.walls = []
    lighting = getattr(scene, "lighting", None)
    sun_direction = getattr(lighting, "sun_direction", getattr(scene, "sun_direction", None))

    for building, spec in zip(scene.buildings, scene.building_specs):
        if building.target_height is None:
            bx, bz = building.position.x, building.position.z
            sampled_y = scene.ground_height_at(bx, bz)
            base_y = sampled_y
        else:
            base_y = None
        wall_thickness = 2.5
        max_doorway_height = max(8.0, float(spec["height"]) - 4.0)
        doorway_height = min(
            float(spec.get("doorway_height", Door.DEFAULT_HEIGHT)),
            max_doorway_height,
        )
        doorway_width = doorway_height * Door.TEXTURE_ASPECT
        spec["base_y"] = (
            float(base_y)
            if base_y is not None
            else float(
                building.target_height
                if building.target_height is not None
                else building.position.y
            )
        )
        spec["doorway_height"] = doorway_height
        spec["doorway_width"] = doorway_width
        spec["wall_thickness"] = wall_thickness
        windows = _normalize_window_specs_for_building(spec)

        pieces = building.create_perimeter_walls(
            wall_height=spec["height"],
            wall_thickness=wall_thickness,
            width=spec["width"],
            depth=spec["depth"],
            max_tile_width=max(spec["width"], spec["depth"]),
            texture=wall_tex,
            uv_repeat=(1.0, 1.0),
            base_y=base_y,
            doorway_side=spec["doorway_side"],
            doorway_width=spec["doorway_width"],
            doorway_height=doorway_height,
            windows=windows,
            roof=True,
            roof_thickness=4.0,
            roof_overhang=BUILDING_ROOF_OVERHANG,
            terrain_height_at=scene.ground_height_at,
            terrain_embed_depth=BUILDING_WALL_TERRAIN_EMBED_DEPTH,
            terrain_sample_spacing=BUILDING_WALL_TERRAIN_SAMPLE_SPACING,
        )
        for piece in pieces:
            piece.sun_direction = sun_direction
            piece.lighting = lighting
        scene.walls.extend(pieces)

    print(f"Built {len(scene.walls)} building pieces.")
    _dispose_values(getattr(scene, "wall_tile_batches", ()))
    scene.wall_tile_batches = build_wall_tile_batches(
        scene.walls,
        camera=scene.camera,
        default_brightness=scene.camera.brightness_default,
        sun_direction=sun_direction,
        lighting=lighting,
    )
    scene.log_timing("Building pieces", start_time, time.perf_counter())
    scene.wall_tiles.extend(scene.walls)
    _build_building_torches(scene)
    _build_building_doors(scene)
    _build_building_windows(scene)


def _build_showcase_polygons(scene) -> None:
    start_time = time.perf_counter()
    for batch in getattr(scene, "polygon_batches", ()) or ():
        try:
            batch.dispose()
        except Exception:
            pass
    scene.polygon_batches = []

    wall_tex = scene.wall_tex
    tri_thickness = 5
    scene.showcase_polygons: list[Polygon] = []
    off_ground = 40

    def regular_polygon(
        cx: float, cy: float, radius: float, sides: int
    ) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for i in range(sides):
            ang = math.radians(90.0 + 360.0 * i / sides)
            x = cx + math.cos(ang) * radius
            y = cy + math.sin(ang) * radius
            pts.append((x, y))
        return pts

    triangle_points = [(0, 0), (60, 0), (30, 50)]
    scene.showcase_polygons.append(
        Polygon(
            position=Vector3(
                scene.world_center.x,
                scene.ground_height_at(scene.world_center.x, scene.world_center.z)
                + off_ground,
                scene.world_center.z,
            ),
            points_2d=triangle_points,
            thickness=tri_thickness,
            texture=wall_tex,
        )
    )

    square_points = [(0, 0), (40, 0), (40, 40), (0, 40)]
    scene.showcase_polygons.append(
        Polygon(
            position=Vector3(
                scene.world_center.x - 100,
                scene.ground_height_at(scene.world_center.x - 100, scene.world_center.z)
                + off_ground,
                scene.world_center.z,
            ),
            points_2d=square_points,
            thickness=tri_thickness,
            texture=wall_tex,
        )
    )

    pent_points = regular_polygon(0.0, 0.0, 30.0, 5)
    scene.showcase_polygons.append(
        Polygon(
            position=Vector3(
                scene.world_center.x + 100,
                scene.ground_height_at(scene.world_center.x + 100, scene.world_center.z)
                + off_ground,
                scene.world_center.z,
            ),
            points_2d=pent_points,
            thickness=tri_thickness,
            texture=wall_tex,
        )
    )

    arrow_points = [
        (0, 10),
        (40, 10),
        (40, -10),
        (60, 20),
        (40, 50),
        (40, 30),
        (0, 30),
    ]
    scene.showcase_polygons.append(
        Polygon(
            position=Vector3(
                scene.world_center.x - 200,
                scene.ground_height_at(scene.world_center.x - 200, scene.world_center.z)
                + off_ground,
                scene.world_center.z - 200,
            ),
            points_2d=arrow_points,
            thickness=tri_thickness,
            texture=wall_tex,
        )
    )

    l_points = [(0, 0), (60, 0), (60, 20), (20, 20), (20, 80), (0, 80)]
    scene.showcase_polygons.append(
        Polygon(
            position=Vector3(
                scene.world_center.x + 230,
                scene.ground_height_at(scene.world_center.x + 230, scene.world_center.z)
                + off_ground,
                scene.world_center.z,
            ),
            points_2d=l_points,
            thickness=tri_thickness,
            texture=wall_tex,
        )
    )

    scene.log_timing("Showcase polygons", start_time, time.perf_counter())
    lighting = getattr(scene, "lighting", None)
    sun_direction = getattr(lighting, "sun_direction", getattr(scene, "sun_direction", None))
    for polygon in scene.showcase_polygons:
        polygon.lighting = lighting
        polygon.sun_direction = sun_direction
    scene.polygons.extend(scene.showcase_polygons)
    polygon_batch = build_polygon_render_batch(scene.showcase_polygons)
    scene.polygon_batches = [polygon_batch] if polygon_batch is not None else []


def _build_roads_and_spawn_sprites(
    scene, tree_count: int, grass_count: int, rock_count: int
) -> None:
    for _label, _finished in _build_roads_and_spawn_sprites_steps(
        scene, tree_count, grass_count, rock_count
    ):
        pass


def _build_roads_and_spawn_sprites_steps(
    scene, tree_count: int, grass_count: int, rock_count: int
) -> Iterator[tuple[str, bool]]:
    label = "Creating roads"
    print("Creating roads...")
    yield (label, False)
    start_time = time.perf_counter()
    center_z = (scene.ground_bounds[2] + scene.ground_bounds[3]) * 0.5
    center_x = (scene.ground_bounds[0] + scene.ground_bounds[1]) * 0.5
    road_y = scene.ground_height_at(center_x, center_z) + 1
    road_points = [
        (scene.ground_bounds[0], center_z),
        (scene.ground_bounds[1], center_z),
    ]
    road_width = 60.0

    scene.camera.position = scene.world_center

    _dispose_value(getattr(scene, "road", None))
    scene.road = Road(
        points=road_points,
        ground_y=road_y,
        width=road_width,
        texture=scene.road_tex,
        px_to_world=1.0,
        v_tiles=1.0,
        height_sampler=scene._ground_height_sampler,
        elevation=3.0,
        segment_length=8.0,
        brightness_modifiers=scene.brightness_modifiers,
        default_brightness=scene.camera.brightness_default,
        lighting=getattr(scene, "lighting", None),
        sun_direction=getattr(scene, "sun_direction", None),
    )

    scene.log_timing("Create road", start_time, time.perf_counter())
    scene.others.append(scene.road)
    scene.roads = [scene.road]
    _dispose_values(getattr(scene, "building_roads", ()))
    scene.building_roads = create_building_access_roads(
        scene,
        road_center_z=center_z,
        road_y=road_y,
        main_road_segment=(road_points[0], road_points[-1]),
    )
    scene.roads.extend(scene.building_roads)
    scene.others.extend(scene.building_roads)
    _build_road_batches(scene)
    segment_count = len(getattr(scene, "building_road_segments", ()))
    print(
        f"Built {len(scene.building_roads)} building access road routes "
        f"({segment_count} segments)."
    )
    yield (label, True)

    label = "Spawning trees"
    yield (label, False)
    _spawn_sprite_layer(
        scene,
        label="trees",
        count=tree_count,
        textures=scene.tree_textures,
        px_to_world=1.2,
        x_off=(scene.ground_bounds[0] + scene.ground_bounds[1]) / 2 + 35,
        z_off=(scene.ground_bounds[2] + scene.ground_bounds[3]) / 2 + 35,
        max_spawn_x=(scene.ground_bounds[1] - scene.ground_bounds[0]) / 2 - 35,
        max_spawn_z=(scene.ground_bounds[3] - scene.ground_bounds[2]) / 2 - 35,
    )
    yield (label, True)

    label = "Spawning goblins"
    yield (label, False)
    _build_goblins(scene)
    yield (label, True)

    label = "Spawning grass"
    yield (label, False)
    _spawn_sprite_layer(
        scene,
        label="grasses",
        count=grass_count,
        textures=scene.grasses_textures,
        px_to_world=1.5,
        x_off=(scene.ground_bounds[0] + scene.ground_bounds[1]) / 2,
        z_off=(scene.ground_bounds[2] + scene.ground_bounds[3]) / 2,
        max_spawn_x=(scene.ground_bounds[1] - scene.ground_bounds[0]) / 2,
        max_spawn_z=(scene.ground_bounds[3] - scene.ground_bounds[2]) / 2,
    )
    yield (label, True)

    label = "Spawning rocks"
    yield (label, False)
    _spawn_sprite_layer(
        scene,
        label="rocks",
        count=rock_count,
        textures=scene.rock_textures,
        px_to_world=1.0,
        x_off=(scene.ground_bounds[0] + scene.ground_bounds[1]) / 2,
        z_off=(scene.ground_bounds[2] + scene.ground_bounds[3]) / 2,
        max_spawn_x=(scene.ground_bounds[1] - scene.ground_bounds[0]) / 2,
        max_spawn_z=(scene.ground_bounds[3] - scene.ground_bounds[2]) / 2,
    )
    yield (label, True)


def _goblin_position_blocker(scene):
    min_x, max_x, min_z, max_z = scene.ground_bounds
    buildings = list(getattr(scene, "buildings", ()) or ())

    def blocked(x: float, z: float, margin: float = 0.0) -> bool:
        clearance = max(0.0, float(margin))
        if (
            x < min_x + clearance
            or x > max_x - clearance
            or z < min_z + clearance
            or z > max_z - clearance
        ):
            return True

        if scene.is_on_road(x, z, margin=clearance):
            return True

        for building in buildings:
            contains_point = getattr(building, "contains_point", None)
            if not callable(contains_point):
                continue
            if contains_point(x, z, margin=clearance):
                return True

        return False

    return blocked


def _random_goblin_spawn_near_tree(scene, rng: random.Random):
    trees = list(getattr(scene, "trees", ()) or ())
    min_x, max_x, min_z, max_z = scene.ground_bounds

    if not trees:
        x = rng.uniform(min_x, max_x)
        z = rng.uniform(min_z, max_z)
        return x, z

    anchor = rng.choice(trees).position
    angle = rng.uniform(0.0, math.tau)
    radius = max(0.0, float(GOBLIN_SPAWN_TREE_RADIUS)) * math.sqrt(rng.random())
    return (
        float(anchor.x) + math.cos(angle) * radius,
        float(anchor.z) + math.sin(angle) * radius,
    )


def _build_goblins(scene) -> None:
    for goblin in getattr(scene, "goblins", ()) or ():
        try:
            if goblin in scene.entities:
                scene.entities.remove(goblin)
            for sprite in goblin.get_sprites() or ():
                if sprite in scene.sprite_items:
                    scene.sprite_items.remove(sprite)
        except Exception:
            continue

    goblin_tex = Goblin.texture_or_load(getattr(scene, "goblin_tex", None))
    scene.goblin_tex = goblin_tex
    front_frames = goblin_tex.get("front", ())
    scene.goblins = []
    count = max(0, int(getattr(scene, "goblin_count", GOBLIN_COUNT)))
    if count <= 0 or not front_frames:
        print("Spawned 0 goblins.")
        return

    shadow_texture = create_shadow_texture(
        width_px=96,
        height_px=96,
        max_alpha=0.24,
        inner_ratio=0.14,
        outer_ratio=0.92,
        falloff_exp=1.8,
        pixelated=False,
    )
    rng = random.Random()
    blocked = _goblin_position_blocker(scene)
    clearance = max(
        float(GOBLIN_SPAWN_CLEARANCE),
        float(getattr(Goblin, "DEFAULT_HEIGHT", 0.0)) * 0.3,
    )
    min_separation_sq = max(0.0, float(GOBLIN_MIN_SEPARATION)) ** 2
    max_attempts = max(1, int(GOBLIN_SPAWN_ATTEMPTS))
    add_entity = getattr(scene, "add_entity", None)

    for _ in range(count):
        spawn = None
        for _attempt in range(max_attempts):
            x, z = _random_goblin_spawn_near_tree(scene, rng)
            if blocked(x, z, clearance):
                continue

            too_close = False
            for other in scene.goblins:
                dx = other.spawn_position.x - x
                dz = other.spawn_position.z - z
                if dx * dx + dz * dz < min_separation_sq:
                    too_close = True
                    break
            if too_close:
                continue

            spawn = Vector3(x, scene.ground_height_at(x, z), z)
            break

        if spawn is None:
            continue

        try:
            goblin = Goblin(
                spawn,
                texture=goblin_tex,
                camera=scene.camera,
                ground_height_at=scene.ground_height_at,
                position_blocked=blocked,
                shadow_texture=shadow_texture,
                rng=random.Random(rng.randrange(1 << 30)),
            )
        except (TypeError, ValueError, AttributeError):
            continue

        scene.goblins.append(goblin)
        if callable(add_entity):
            add_entity(goblin)
        else:
            scene.entities.append(goblin)
            scene.sprite_items.extend(goblin.get_sprites())

    print(f"Spawned {len(scene.goblins)} goblins.")
    refresh_draw_entities = getattr(scene, "refresh_immediate_entities", None)
    if callable(refresh_draw_entities):
        refresh_draw_entities()


def _spawn_sprite_layer(
    scene,
    *,
    label: str,
    count: int,
    textures: list,
    px_to_world: float,
    x_off: float,
    z_off: float,
    max_spawn_x: float,
    max_spawn_z: float,
) -> None:
    start_time = time.perf_counter()
    sprites = spawn_world_sprites(
        scene,
        count=count,
        textures=textures,
        px_to_world=px_to_world,
        camera=scene.camera,
        x_off=x_off,
        z_off=z_off,
        max_spawn_x=max_spawn_x,
        max_spawn_z=max_spawn_z,
        avoid_roads=scene.roads,
        avoid_areas=scene.buildings,
    )
    setattr(scene, label, sprites)
    print(f"Spawned {len(sprites)} {label}.")
    scene.log_timing(f"Spawn {label}", start_time, time.perf_counter())
    scene.sprite_items.extend(sprites)


def _build_fences(scene) -> None:
    start_time = time.perf_counter()
    _dispose_values(getattr(scene, "fence_meshes", ()))
    scene.fence_meshes = build_textured_fence_ring(
        min_x=scene.ground_bounds[0],
        max_x=scene.ground_bounds[1],
        min_z=scene.ground_bounds[2],
        max_z=scene.ground_bounds[3],
        ground_y=scene.ground_height_at(0, 0),
        height_sampler=getattr(scene.ground_mesh, "height_sampler", None),
        textures=[t for t in scene.fence_textures if t is not None],
        px_to_world=1.0,
        wave_amp=0.5,
        wave_freq=0.02,
        wave_phase=0.3,
        brightness_modifiers=scene.brightness_modifiers,
        default_brightness=scene.camera.brightness_default,
        lighting=getattr(scene, "lighting", None),
        sun_direction=getattr(scene, "sun_direction", None),
    )
    print(f"Built {len(scene.fence_meshes)} fence segments.")
    scene.log_timing("Build fences", start_time, time.perf_counter())

    start_time = time.perf_counter()
    scene.log_timing("Assemble static meshes", start_time, time.perf_counter())


def _build_shadow_decals(scene) -> None:
    tree_shadow_subdiv = 12
    start_time = time.perf_counter()
    tree_shadow_texture = create_tree_shadow_texture(
        width_px=256,
        height_px=256,
        max_alpha=0.48,
        variant_seed=0,
        pixelated=False,
    )
    contact_shadow_texture = create_shadow_texture(
        width_px=128,
        height_px=128,
        max_alpha=0.34,
        inner_ratio=0.18,
        outer_ratio=0.98,
        falloff_exp=1.55,
        pixelated=False,
    )
    print("Created shadow textures.")
    scene.log_timing("Create shadow textures", start_time, time.perf_counter())

    decals: list[Decal] = []
    rng = random.Random()
    shadow_receiver = _outside_building_shadow_receiver(
        _building_shadow_clip_bounds(scene)
    )

    def make_tree_decal_for_sprite(sprite: WorldSprite) -> Decal:
        w, h = sprite.size
        size_w = w * rng.uniform(0.45, 0.75)
        size_h = h * rng.uniform(0.45, 0.75)

        lighting = getattr(scene, "lighting", None)
        sun = getattr(lighting, "sun_direction", getattr(scene, "sun_direction", None))
        final_w, final_h = size_w, size_h
        offset_x, offset_z = 0.0, 0.0
        rot = 0.0

        if sun is not None:
            proj_x = float(sun.x)
            proj_z = float(sun.z)
            proj_len = math.hypot(proj_x, proj_z)

            if proj_len >= 1e-6:
                vert = abs(float(sun.y))
                elong = 1.0 / max(0.05, vert)
                elong = max(1.0, min(elong, 12.0))

                seed = max(size_w, size_h)
                major = max(14.0, min(400.0, seed * (0.9 + elong * 0.6)))
                minor = max(8.0, min(200.0, min(size_w, size_h) * 0.9))
                final_w, final_h = major, minor

                dir_x = -proj_x / proj_len
                dir_z = -proj_z / proj_len

                offset_x = -dir_x * (major * 0.45)
                offset_z = -dir_z * (major * 0.5)

                angle_rad = math.atan2(-proj_x, -proj_z)
                angle_deg = math.degrees(angle_rad)
                rot = (angle_deg + 90.0) % 360.0

        center_y = scene.ground_height_at(
            sprite.position.x + offset_x,
            sprite.position.z + offset_z,
        )

        return Decal(
            center=Vector3(
                sprite.position.x + offset_x,
                center_y,
                sprite.position.z + offset_z,
            ),
            size=(final_w, final_h),
            texture=tree_shadow_texture,
            rotation_deg=rot,
            subdiv_u=tree_shadow_subdiv,
            subdiv_v=tree_shadow_subdiv,
            height_fn=scene.ground_height_at,
            elevation=1,
            uv_repeat=(1.0, 1.0),
            color=(1.0, 1.0, 1.0),
            receiver_fn=shadow_receiver,
            # DecalBatch owns the final VBO; avoid building throwaway per-shadow VBOs.
            build_vbo=False,
        )

    def make_contact_decal_for_sprite(
        sprite: WorldSprite,
        *,
        scale_min: float,
        scale_max: float,
        min_size: float,
        max_size: float,
    ) -> Decal:
        w, h = sprite.size
        footprint = max(1.0, max(w, h))
        diameter = max(
            min_size,
            min(max_size, footprint * rng.uniform(scale_min, scale_max)),
        )
        center_y = scene.ground_height_at(sprite.position.x, sprite.position.z)

        return Decal(
            center=Vector3(sprite.position.x, center_y, sprite.position.z),
            size=(diameter, diameter),
            texture=contact_shadow_texture,
            rotation_deg=rng.uniform(0.0, 360.0),
            subdiv_u=2,
            subdiv_v=2,
            height_fn=scene.ground_height_at,
            elevation=0.75,
            uv_repeat=(1.0, 1.0),
            color=(1.0, 1.0, 1.0),
            receiver_fn=shadow_receiver,
            build_vbo=False,
        )

    start_time = time.perf_counter()
    for sprite in scene.trees:
        decals.append(make_tree_decal_for_sprite(sprite))
    for sprite in getattr(scene, "grasses", ()):
        decals.append(
            make_contact_decal_for_sprite(
                sprite,
                scale_min=.8,
                scale_max=1,
                min_size=8.0,
                max_size=28.0,
            )
        )
    for sprite in getattr(scene, "rocks", ()):
        decals.append(
            make_contact_decal_for_sprite(
                sprite,
                scale_min=1,
                scale_max=1.05,
                min_size=10.0,
                max_size=42.0,
            )
        )

    print(
        f"Created {len(scene.trees)} tree, "
        f"{len(getattr(scene, 'grasses', ()))} grass, "
        f"and {len(getattr(scene, 'rocks', ()))} rock shadow decals."
    )
    scene.log_timing("Create decals", start_time, time.perf_counter())
    _dispose_value(getattr(scene, "decal_batch", None))
    _dispose_values(getattr(scene, "decal_batches", ()))
    scene.decal_batches = []
    scene.decal_batch = DecalBatch.build(decals)
    scene.decal_batches.append(scene.decal_batch)
    start_time = time.perf_counter()
    scene.log_timing("Build decal batch", start_time, time.perf_counter())
