"""Lighting authoring derived from high-level world content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.rendering.lighting import INDOOR_LIGHT_FACTOR
from engine.rendering.lighting_state import LocalBrightnessLight
from game.world.environment import EnvironmentPortal, EnvironmentVolume
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


@dataclass(frozen=True, slots=True)
class AuthoredOpeningLight:
    """Typed local light plus doorway/window transition metadata."""

    light: LocalBrightnessLight
    opening_type: str
    closed_radius: float | None = None
    open_radius: float | None = None
    open_value: float | None = None

    def to_legacy_dict(self) -> dict[str, object]:
        result = self.light.to_legacy_dict()
        result["opening_type"] = self.opening_type
        if self.closed_radius is not None:
            result["closed_radius"] = self.closed_radius
        if self.open_radius is not None:
            result["open_radius"] = self.open_radius
        if self.open_value is not None:
            result["open_value"] = self.open_value
        return result


_MISSING = object()


def _field(source: object, name: str, default: object = _MISSING):
    if isinstance(source, dict):
        if default is _MISSING:
            return source[name]
        return source.get(name, default)
    if default is _MISSING:
        return getattr(source, name)
    return getattr(source, name, default)


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


def _window_environment_portal(
    spec: dict,
    window: dict,
    *,
    volume_id: str,
    window_index: int,
) -> EnvironmentPortal | None:
    try:
        side = str(window.get("side", "north")).lower()
        width = max(1.0, float(window.get("width", Window.DEFAULT_WIDTH)))
        offset = float(window.get("offset", 0.0))
        half_x = float(spec["width"]) * 0.5
        half_z = float(spec["depth"]) * 0.5
    except (KeyError, TypeError, ValueError, AttributeError):
        return None

    if side not in {"north", "east", "south", "west"}:
        return None

    center_x, center_z = _opening_light_center(spec, side, offset)
    depth = max(36.0, min(86.0, min(half_x, half_z) * 0.58))
    return EnvironmentPortal(
        portal_id=f"{volume_id}:window:{window_index}",
        kind="window",
        side=side,
        center_x=center_x,
        center_z=center_z,
        width=max(width * 1.4, width + 10.0),
        depth=depth,
        side_fade=max(8.0, width * 0.42),
        closed_factor=WINDOW_LIGHT_EDGE_FACTOR,
        open_factor=WINDOW_LIGHT_EDGE_FACTOR,
        openness=1.0,
    )


def building_environment_volumes(building_specs) -> list[EnvironmentVolume]:
    """Build typed indoor volumes before projecting legacy lighting regions."""

    volumes: list[EnvironmentVolume] = []
    for spec_index, spec in enumerate(building_specs or ()):
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
        if side not in {"north", "east", "south", "west"}:
            side = "south"
        volume_id = f"building:{spec_index}"
        doorway = EnvironmentPortal(
            portal_id=f"{volume_id}:doorway",
            kind="doorway",
            side=side,
            center_x=x,
            center_z=z,
            width=max(doorway_width * 1.16, doorway_width + 8.0),
            depth=doorway_depth,
            side_fade=max(10.0, doorway_width * 0.26),
            closed_factor=indoor_factor,
            open_factor=1.0,
        )
        windows = [
            portal
            for portal in (
                _window_environment_portal(
                    spec,
                    window,
                    volume_id=volume_id,
                    window_index=window_index,
                )
                for window_index, window in enumerate(spec.get("windows", ()) or ())
            )
            if portal is not None
        ]
        volumes.append(
            EnvironmentVolume(
                volume_id=volume_id,
                min_x=min_x,
                max_x=max_x,
                min_z=min_z,
                max_z=max_z,
                indoor_factor=indoor_factor,
                portals=(doorway, *windows),
            )
        )
    return volumes


def building_covered_regions(building_specs) -> list[dict[str, Any]]:
    """Compatibility projection for current CPU and shader lighting paths."""

    return [
        volume.to_legacy_dict()
        for volume in building_environment_volumes(building_specs)
    ]


def _opening_wall_light_center(
    region: object,
    opening: object,
) -> tuple[float, float]:
    side = str(_field(opening, "side", "")).lower()
    min_x = float(_field(region, "min_x"))
    max_x = float(_field(region, "max_x"))
    min_z = float(_field(region, "min_z"))
    max_z = float(_field(region, "max_z"))
    center_x = float(_field(opening, "center_x", (min_x + max_x) * 0.5))
    center_z = float(_field(opening, "center_z", (min_z + max_z) * 0.5))
    inset = max(
        0.0,
        float(_field(opening, "wall_light_inset", OPENING_WALL_LIGHT_INSET)),
    )

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
    region: object,
    opening: object,
    *,
    radius: float | None = None,
) -> tuple[float, float, float, float]:
    inset = OPENING_WALL_LIGHT_BOUNDS_INSET
    min_x = float(_field(region, "min_x")) + inset
    max_x = float(_field(region, "max_x")) - inset
    min_z = float(_field(region, "min_z")) + inset
    max_z = float(_field(region, "max_z")) - inset
    if max_x < min_x:
        min_x, max_x = max_x, min_x
    if max_z < min_z:
        min_z, max_z = max_z, min_z

    side = str(_field(opening, "side", "")).lower()
    center_x = float(_field(opening, "center_x", (min_x + max_x) * 0.5))
    center_z = float(_field(opening, "center_z", (min_z + max_z) * 0.5))
    width = max(1.0, float(_field(opening, "width", 48.0)))
    side_fade = max(1.0, float(_field(opening, "side_fade", width * 0.25)))
    influence_radius = max(0.0, float(radius)) if radius is not None else 0.0
    wall_light_inset = max(
        0.0,
        float(_field(opening, "wall_light_inset", OPENING_WALL_LIGHT_INSET)),
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
    opening: object,
    *,
    min_radius: float,
    max_radius: float,
) -> float:
    width = max(1.0, float(_field(opening, "width", 48.0)))
    depth = max(1.0, float(_field(opening, "depth", 64.0)))
    radius = max(width * 1.08, depth * 0.92)
    return max(min_radius, min(max_radius, radius))


def _opening_wall_light(
    region: object,
    opening: object,
) -> AuthoredOpeningLight | None:
    try:
        opening_type = str(
            _field(opening, "kind", _field(opening, "type", ""))
        ).lower()
        portal_id = str(_field(opening, "portal_id", opening_type or "opening"))
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
        return AuthoredOpeningLight(
            light=LocalBrightnessLight(
                light_id=f"{portal_id}:wall-splash",
                center=(x, 0.0, z),
                radius=0.0,
                value=DOORWAY_WALL_SPLASH_VALUE,
                falloff=DOORWAY_WALL_LIGHT_FALLOFF,
                bounds=bounds,
                indoor_only=True,
                floor_scale=OPENING_WALL_LIGHT_FLOOR_SCALE,
            ),
            opening_type="doorway",
            closed_radius=0.0,
            open_radius=open_radius,
            open_value=DOORWAY_WALL_SPLASH_VALUE,
        )

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
        return AuthoredOpeningLight(
            light=LocalBrightnessLight(
                light_id=f"{portal_id}:wall-splash",
                center=(x, 0.0, z),
                radius=radius,
                value=WINDOW_WALL_SPLASH_VALUE,
                falloff=WINDOW_WALL_LIGHT_FALLOFF,
                bounds=bounds,
                indoor_only=True,
                floor_scale=OPENING_WALL_LIGHT_FLOOR_SCALE,
            ),
            opening_type="window",
        )

    return None


def opening_wall_lights_for_volumes(
    volumes,
) -> tuple[list[AuthoredOpeningLight | None], list[AuthoredOpeningLight]]:
    """Author typed opening lights directly from typed environment state."""

    doorway_lights: list[AuthoredOpeningLight | None] = []
    window_lights: list[AuthoredOpeningLight] = []
    for volume in volumes or ():
        if not isinstance(volume, EnvironmentVolume):
            doorway_lights.append(None)
            continue
        doorway = volume.doorway
        doorway_lights.append(
            _opening_wall_light(volume, doorway) if doorway is not None else None
        )
        for portal in volume.portals:
            if portal.kind != "window":
                continue
            light = _opening_wall_light(volume, portal)
            if light is not None:
                window_lights.append(light)
    return doorway_lights, window_lights


def _opening_wall_light_modifier(
    region: dict,
    opening: dict,
) -> dict | None:
    """Compatibility wrapper for legacy region callers."""

    authored = _opening_wall_light(region, opening)
    return authored.to_legacy_dict() if authored is not None else None


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


def _copy_opening_light_metadata(target: dict, source: object) -> dict:
    if isinstance(source, AuthoredOpeningLight):
        source = source.to_legacy_dict()
    if not isinstance(source, dict):
        return target
    for key in (
        "opening_type",
        "closed_radius",
        "open_radius",
        "open_value",
    ):
        if key in source:
            target[key] = source[key]
    return target


def _legacy_light_dict(source: object) -> dict | None:
    if isinstance(source, AuthoredOpeningLight):
        return source.to_legacy_dict()
    if isinstance(source, LocalBrightnessLight):
        return source.to_legacy_dict()
    if isinstance(source, dict):
        return source
    return None


def _install_scene_brightness_modifier(scene, source: object) -> object | None:
    modifier = _legacy_light_dict(source)
    if modifier is None:
        return None
    lighting = getattr(scene, "lighting", None)
    camera = getattr(scene, "camera", None)
    packet_backend = getattr(scene, "lighting_backend", "legacy") == "packet"
    if lighting is not None:
        try:
            typed_light = (
                source.light if isinstance(source, AuthoredOpeningLight) else source
            )
            installed = lighting.add_local_light(
                typed_light,
                camera=camera,
            )
            if getattr(scene, "lighting_backend", "legacy") == "packet":
                return source
            return _copy_opening_light_metadata(installed.to_legacy_dict(), source)
        except (KeyError, TypeError, ValueError, AttributeError):
            return None

    if not hasattr(scene, "brightness_modifiers") or scene.brightness_modifiers is None:
        scene.brightness_modifiers = []
    scene.brightness_modifiers.append(modifier)
    try:
        Torch.install_local_light(camera, modifier)
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return modifier


def install_building_lights(
    scene,
    building_specs=None,
    *,
    environment_volumes=None,
) -> None:
    specs = (
        building_specs
        if building_specs is not None
        else getattr(scene, "building_specs", ()) or ()
    )
    torch_lights = Torch.local_lights_for_building_specs(specs)
    torch_point_lights = Torch.point_lights_for_building_specs(specs)
    volumes = (
        environment_volumes
        if environment_volumes is not None
        else getattr(scene, "environment_volumes", None)
    )
    if volumes is not None:
        doorway_lights_by_region, window_lights = opening_wall_lights_for_volumes(
            volumes
        )
    else:
        doorway_lights_by_region, window_lights = (
            opening_wall_light_modifiers_for_regions(
                getattr(scene, "covered_regions", ()) or ()
            )
        )
    lighting = getattr(scene, "lighting", None)
    camera = getattr(scene, "camera", None)
    packet_backend = getattr(scene, "lighting_backend", "legacy") == "packet"
    if lighting is not None:
        remove_lights = getattr(lighting, "remove_local_lights", None)
        if callable(remove_lights):
            remove_lights(id_prefix="building:", camera=camera)
        remove_point_lights = getattr(lighting, "remove_point_lights", None)
        if callable(remove_point_lights):
            remove_point_lights(id_prefix="building:")
        extend_point_lights = getattr(lighting, "extend_point_lights", None)
        if callable(extend_point_lights):
            extend_point_lights(torch_point_lights)
    else:
        retained = [
            modifier
            for modifier in getattr(scene, "brightness_modifiers", ()) or ()
            if not str(
                modifier.get("light_id", "")
                if isinstance(modifier, dict)
                else ""
            ).startswith("building:")
        ]
        scene.brightness_modifiers = retained
        clear_areas = getattr(camera, "clear_brightness_query_lights", None)
        if callable(clear_areas):
            clear_areas()
            for modifier in retained:
                try:
                    Torch.install_local_light(camera, modifier)
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
    scene.torch_light_modifiers = []
    scene.torch_point_lights = list(torch_point_lights) if packet_backend else []
    scene.doorway_light_modifiers_by_region = []
    scene.doorway_light_modifiers = []
    scene.window_light_modifiers = []
    scene.opening_light_modifiers = []

    # Packet rendering gets illumination from geometry, sun shadows, and the
    # typed torch PointLight records above.  Do not install the legacy scalar
    # doorway/window/torch splashes into its authoritative scene state.
    if packet_backend:
        replace_query_lights = getattr(
            camera,
            "replace_brightness_query_lights",
            None,
        )
        if callable(replace_query_lights) and lighting is not None:
            replace_query_lights((), source_revision=lighting.revision)
        return

    for light in doorway_lights_by_region:
        if light is None:
            scene.doorway_light_modifiers_by_region.append(None)
            continue
        installed = _install_scene_brightness_modifier(scene, light)
        if installed is not None:
            scene.doorway_light_modifiers_by_region.append(installed)
            scene.doorway_light_modifiers.append(installed)
            scene.opening_light_modifiers.append(installed)
        else:
            scene.doorway_light_modifiers_by_region.append(None)

    for light in window_lights:
        installed = _install_scene_brightness_modifier(scene, light)
        if installed is not None:
            scene.window_light_modifiers.append(installed)
            scene.opening_light_modifiers.append(installed)

    for light in torch_lights:
        installed = _install_scene_brightness_modifier(scene, light)
        if installed is not None:
            scene.torch_light_modifiers.append(installed)

    if lighting is not None:
        lighting_controller = getattr(scene, "lighting_controller", None)
        packet_backend = getattr(scene, "lighting_backend", "legacy") == "packet"
        if lighting_controller is not None and not packet_backend:
            lighting_controller.sync_aliases()
        elif not packet_backend:
            scene.brightness_modifiers = [
                light.to_legacy_dict()
                for light in getattr(lighting, "local_lights", ()) or ()
            ]


def apply_building_lighting(scene, building_specs=None) -> list[dict[str, Any]]:
    specs = (
        building_specs
        if building_specs is not None
        else getattr(scene, "building_specs", ())
    )
    environment_volumes = building_environment_volumes(specs)
    scene.environment_volumes = environment_volumes
    packet_backend = getattr(scene, "lighting_backend", "legacy") == "packet"
    regions = (
        []
        if packet_backend
        else [volume.to_legacy_dict() for volume in environment_volumes]
    )
    if not packet_backend:
        scene.covered_regions = regions
    lighting = getattr(scene, "lighting", None)
    if lighting is not None and not packet_backend:
        lighting_controller = getattr(scene, "lighting_controller", None)
        if lighting_controller is not None:
            lighting_controller.set_legacy_covered_regions(regions)
            lighting_controller.sync_aliases()
    install_building_lights(
        scene,
        specs,
        environment_volumes=environment_volumes,
    )
    return regions
