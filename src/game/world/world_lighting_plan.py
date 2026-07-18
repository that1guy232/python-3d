"""Lighting authoring derived from high-level world content."""

from __future__ import annotations

from typing import Any

from pygame.math import Vector3

from engine.rendering.lighting import INDOOR_LIGHT_FACTOR
from game.world.objects import Torch, Window

WINDOW_LIGHT_EDGE_FACTOR = 0.86
DOORWAY_WALL_SPLASH_VALUE = 1.36
DOORWAY_WALL_LIGHT_FALLOFF = 1.65
DOORWAY_WALL_LIGHT_MIN_RADIUS = 54.0
DOORWAY_WALL_LIGHT_MAX_RADIUS = 118.0
WINDOW_WALL_SPLASH_VALUE = 1.26
WINDOW_WALL_LIGHT_FALLOFF = 1.9
WINDOW_WALL_LIGHT_MIN_RADIUS = 42.0
WINDOW_WALL_LIGHT_MAX_RADIUS = 96.0
OPENING_WALL_LIGHT_INSET = 10.0
OPENING_WALL_LIGHT_BOUNDS_INSET = 2.0
OPENING_WALL_LIGHT_BAND_DEPTH = 18.0
OPENING_WALL_LIGHT_LATERAL_PAD = 28.0
OPENING_WALL_LIGHT_FLOOR_SCALE = 0.0


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


def building_covered_regions(building_specs) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for spec in building_specs or ():
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


def opening_wall_light_modifiers_for_regions(
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


def install_building_lights(scene, building_specs=None) -> None:
    torch_modifiers = Torch.brightness_modifiers_for_building_specs(
        building_specs
        if building_specs is not None
        else getattr(scene, "building_specs", ()) or ()
    )
    doorway_modifiers_by_region, window_modifiers = (
        opening_wall_light_modifiers_for_regions(
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
        lighting_controller = getattr(scene, "lighting_controller", None)
        if lighting_controller is not None:
            lighting_controller.sync_aliases()
        else:
            scene.brightness_modifiers = lighting.brightness_modifiers


def apply_building_lighting(scene, building_specs=None) -> list[dict[str, Any]]:
    specs = (
        building_specs
        if building_specs is not None
        else getattr(scene, "building_specs", ())
    )
    regions = building_covered_regions(specs)
    scene.covered_regions = regions
    lighting = getattr(scene, "lighting", None)
    if lighting is not None:
        lighting.set_covered_regions(regions)
        lighting_controller = getattr(scene, "lighting_controller", None)
        if lighting_controller is not None:
            lighting_controller.sync_aliases()
    install_building_lights(scene, specs)
    return regions
