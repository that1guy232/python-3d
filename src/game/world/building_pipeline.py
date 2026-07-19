"""Building and showcase construction pipeline for world scenes."""

from __future__ import annotations


import math
import time


from pygame.math import Vector3


from engine.textures.texture_utils import load_texture

from game.resources.paths import WALL1_TEXTURE_PATH

from game.world.builder_support import _dispose_value, _dispose_values

from game.world.interior_layout import create_building_interior_layout

from game.world.objects import Chest, Door, Torch, Window

from game.world.objects.building import Building

from game.world.objects.door import build_door_render_batch

from game.world.objects.ground import TexturedGroundGridBuilder

from game.world.objects.polygon import Polygon, build_polygon_render_batch

from game.world.objects.wall_tile import build_wall_tile_batches

from game.world.objects.window import build_window_render_batch

from game.world.world_content import create_world_content, resolve_world_content

from game.world.world_lighting_plan import apply_building_lighting

BUILDING_ROOF_OVERHANG = 6.0

BUILDING_WALL_TERRAIN_EMBED_DEPTH = 8.0

BUILDING_WALL_TERRAIN_SAMPLE_SPACING = 18.0


_SIDE_NORMALS = {
    "north": (0.0, 1.0),
    "east": (1.0, 0.0),
    "south": (0.0, -1.0),
    "west": (-1.0, 0.0),
}

_WINDOW_SIDE_BY_DOORWAY = {
    "north": "east",
    "east": "south",
    "south": "west",
    "west": "north",
}


def _ensure_state_owners(scene) -> None:
    """Allow focused legacy fakes while production uses explicit owners."""
    if not hasattr(scene, "build_state"):
        scene.build_state = scene
    if not hasattr(scene, "render_resources"):
        scene.render_resources = scene


def _build_building_torches(scene) -> None:

    _ensure_state_owners(scene)

    # Building base_y values are terrain-resolved during structure creation,
    # after the early environment pass. Re-author torch lights here so visible
    # sprites and point lights share the final wall-relative height.
    apply_building_lighting(scene, getattr(scene, "building_specs", ()) or ())

    torch_tex = Torch.texture_or_load(getattr(scene, "torch_tex", None))

    scene.render_resources.torch_tex = torch_tex

    if getattr(scene, "lighting_backend", "legacy") == "packet":
        scene.build_state.torches = Torch.build_for_point_lights(
            getattr(scene, "torch_point_lights", ()) or (),
            texture=torch_tex,
            camera=scene.camera,
        )
    else:
        scene.build_state.torches = Torch.build_for_brightness_modifiers(
            getattr(scene, "torch_light_modifiers", ()) or (),
            texture=torch_tex,
            camera=scene.camera,
            ground_height_at=scene.ground_height_at,
        )

    scene.render_resources.sprite_items.extend(scene.build_state.torches)


def _interior_door_from_spec(
    building_spec: dict,
    door_spec: dict,
    *,
    texture,
    camera,
    ground_height_at,
    lighting=None,
    sun_direction=None,
) -> Door:

    center = building_spec["position"]

    local_x = float(door_spec.get("x", 0.0))

    local_z = float(door_spec.get("z", 0.0))

    x = float(center.x) + local_x

    z = float(center.z) + local_z

    side = str(door_spec.get("side", "south")).lower()

    doorway_height = max(
        8.0,
        float(
            door_spec.get(
                "height",
                building_spec.get("doorway_height", Door.DEFAULT_HEIGHT),
            )
        ),
    )

    doorway_width = max(
        8.0,
        float(
            door_spec.get(
                "width",
                building_spec.get(
                    "doorway_width",
                    doorway_height * Door.TEXTURE_ASPECT,
                ),
            )
        ),
    )

    visual_width = max(8.0, doorway_width + 1.0)

    visual_height = max(8.0, visual_width / Door.TEXTURE_ASPECT)

    base_y = building_spec.get("base_y", None)

    if base_y is None:

        base_y = ground_height_at(float(center.x), float(center.z))

    wall_thickness = max(4.0, float(building_spec.get("wall_thickness", 2.5)))

    return Door(
        Vector3(x, float(base_y) + visual_height * 0.5, z),
        camera=camera,
        texture=texture,
        lighting=lighting,
        sun_direction=sun_direction,
        width=visual_width,
        height=visual_height,
        side=side,
        thickness=wall_thickness,
        collision_width=visual_width,
        collision_height=visual_height,
        collision_thickness=wall_thickness,
        interior=True,
    )


def _build_building_doors(scene) -> None:

    _ensure_state_owners(scene)

    for batch in getattr(scene, "door_batches", ()) or ():

        try:

            batch.dispose()

        except Exception:

            pass

    scene.render_resources.door_batches = []

    for door in getattr(scene, "doors", ()) or ():

        try:

            if door in scene.render_resources.entities:

                scene.render_resources.entities.remove(door)

            for sprite in door.get_sprites() or ():

                if sprite in scene.render_resources.sprite_items:

                    scene.render_resources.sprite_items.remove(sprite)

            for mesh in door.get_collision_meshes() or ():

                if mesh in scene.render_resources.wall_tiles:

                    scene.render_resources.wall_tiles.remove(mesh)

        except Exception:

            continue

    door_tex = Door.texture_or_load(getattr(scene, "door_tex", None))

    scene.render_resources.door_tex = door_tex

    scene.build_state.doors = []

    add_entity = getattr(scene, "add_entity", None)

    packet_backend = getattr(scene, "lighting_backend", "legacy") == "packet"

    covered_regions = (
        [] if packet_backend else list(getattr(scene, "covered_regions", ()) or ())
    )

    environment_volumes = list(getattr(scene, "environment_volumes", ()) or ())

    doorway_light_modifiers = list(
        getattr(scene, "doorway_light_modifiers_by_region", ()) or ()
    )

    lighting = getattr(scene, "lighting", None)

    sun_direction = (
        None if lighting is not None else getattr(scene, "sun_direction", None)
    )

    def add_door(door: Door) -> None:

        scene.build_state.doors.append(door)

        if callable(add_entity):

            add_entity(door)

        else:

            scene.render_resources.entities.append(door)

            scene.render_resources.sprite_items.extend(door.get_sprites())

            scene.render_resources.wall_tiles.extend(door.get_collision_meshes())

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

        portal = (
            environment_volumes[spec_index].doorway
            if spec_index < len(environment_volumes)
            else None
        )

        if spec_index < len(covered_regions) or portal is not None:

            brightness_modifier = (
                doorway_light_modifiers[spec_index]
                if spec_index < len(doorway_light_modifiers)
                else None
            )

            door.bind_doorway_light(
                (
                    covered_regions[spec_index]
                    if spec_index < len(covered_regions)
                    else None
                ),
                brightness_modifier=brightness_modifier,
                portal=portal,
            )

        add_door(door)

        interior = spec.get("interior", {})

        interior_door_specs = (
            interior.get("doors", ()) if isinstance(interior, dict) else ()
        )

        for interior_door_spec in interior_door_specs:

            try:

                interior_door = _interior_door_from_spec(
                    spec,
                    interior_door_spec,
                    texture=door_tex,
                    camera=scene.camera,
                    ground_height_at=scene.ground_height_at,
                    lighting=lighting,
                    sun_direction=sun_direction,
                )

            except (KeyError, TypeError, ValueError, AttributeError):

                continue

            add_door(interior_door)

    door_batch = build_door_render_batch(scene.build_state.doors)

    scene.render_resources.door_batches = (
        [door_batch] if door_batch is not None else []
    )

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

    _ensure_state_owners(scene)

    for batch in getattr(scene, "window_batches", ()) or ():

        try:

            batch.dispose()

        except Exception:

            pass

    scene.render_resources.window_batches = []

    for window in getattr(scene, "windows", ()) or ():

        try:

            if window in scene.render_resources.entities:

                scene.render_resources.entities.remove(window)

            for sprite in window.get_sprites() or ():

                if sprite in scene.render_resources.sprite_items:

                    scene.render_resources.sprite_items.remove(sprite)

            for mesh in window.get_collision_meshes() or ():

                if mesh in scene.render_resources.wall_tiles:

                    scene.render_resources.wall_tiles.remove(mesh)

        except Exception:

            continue

    window_tex = Window.texture_or_load(getattr(scene, "window_tex", None))

    scene.render_resources.window_tex = window_tex

    scene.build_state.windows = []

    add_entity = getattr(scene, "add_entity", None)

    lighting = getattr(scene, "lighting", None)

    sun_direction = (
        None if lighting is not None else getattr(scene, "sun_direction", None)
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

            scene.build_state.windows.append(window)

            if callable(add_entity):

                add_entity(window)

            else:

                scene.render_resources.entities.append(window)

                scene.render_resources.sprite_items.extend(window.get_sprites())

                scene.render_resources.wall_tiles.extend(window.get_collision_meshes())

    window_batch = build_window_render_batch(scene.build_state.windows)

    scene.render_resources.window_batches = (
        [window_batch] if window_batch is not None else []
    )

    refresh_draw_entities = getattr(scene, "refresh_immediate_entities", None)

    if callable(refresh_draw_entities):

        refresh_draw_entities()


def create_building_specs(scene, count: int = 10) -> list[dict]:

    return create_world_content(scene, building_count=count).to_building_specs()


def _prepare_buildings(
    scene,
    grid_count: int,
    grid_tile_size: int,
    grid_gap: int,
) -> None:

    _ensure_state_owners(scene)

    scene.build_state.buildings = []

    content = resolve_world_content(
        scene,
        building_count=int(getattr(scene, "building_count", 10)),
    )

    scene.build_state.building_specs = content.to_building_specs()

    for spec in scene.building_specs:

        _normalize_window_specs_for_building(spec)

    for spec in scene.building_specs:

        scene.build_state.buildings.append(Building(position=spec["position"]))

    apply_building_lighting(scene, scene.building_specs)

    packet_backend = getattr(scene, "lighting_backend", "legacy") == "packet"

    scene.build_state.builder = TexturedGroundGridBuilder(
        count=grid_count,
        tile_size=grid_tile_size,
        gap=grid_gap,
        texture=scene.ground_tex,
        brightness_modifiers=(
            () if packet_backend else getattr(scene, "brightness_modifiers", ())
        ),
        default_brightness=scene.camera.brightness_default,
        lighting=getattr(scene, "lighting", None),
        sun_direction=(
            None
            if getattr(scene, "lighting", None) is not None
            else getattr(scene, "sun_direction", None)
        ),
        covered_regions=(
            () if packet_backend else getattr(scene, "covered_regions", ())
        ),
        environment_volumes=getattr(scene, "environment_volumes", ()),
        dynamic_lighting=getattr(scene, "lighting_backend", "legacy") == "packet",
    )


def _build_showcase_polygons_and_collision(scene) -> None:

    _build_showcase_polygons(scene)

    rebuild_collision_index = getattr(scene, "rebuild_collision_index", None)

    if callable(rebuild_collision_index):

        rebuild_collision_index()


def _build_buildings(scene) -> None:

    _ensure_state_owners(scene)

    start_time = time.perf_counter()

    wall_tex = scene.wall_tex or load_texture(WALL1_TEXTURE_PATH)

    scene.render_resources.wall_tex = wall_tex

    scene.build_state.walls = []

    lighting = getattr(scene, "lighting", None)

    sun_direction = (
        None if lighting is not None else getattr(scene, "sun_direction", None)
    )

    for building, spec in zip(
        scene.build_state.buildings,
        scene.build_state.building_specs,
    ):

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

        interior = create_building_interior_layout(spec)

        spec["interior"] = interior

        pieces.extend(
            building.create_interior_walls(
                interior.get("partitions", ()),
                wall_height=spec["height"],
                wall_thickness=wall_thickness,
                texture=wall_tex,
                uv_repeat=(1.0, 1.0),
                base_y=base_y,
                max_tile_width=max(spec["width"], spec["depth"]),
            )
        )

        for piece in pieces:

            piece.sun_direction = sun_direction

            piece.lighting = lighting

        scene.build_state.walls.extend(pieces)

    print(f"Built {len(scene.build_state.walls)} building pieces.")

    _dispose_values(getattr(scene, "wall_tile_batches", ()))

    scene.render_resources.wall_tile_batches = build_wall_tile_batches(
        scene.build_state.walls,
        camera=scene.camera,
        default_brightness=scene.camera.brightness_default,
        sun_direction=sun_direction,
        lighting=lighting,
        dynamic_lighting=getattr(scene, "lighting_backend", "legacy") == "packet",
    )

    scene.log_timing("Building pieces", start_time, time.perf_counter())

    scene.render_resources.wall_tiles.extend(scene.build_state.walls)

    _build_building_torches(scene)

    _build_building_doors(scene)

    _build_building_windows(scene)


def _build_showcase_polygons(scene) -> None:

    _ensure_state_owners(scene)

    start_time = time.perf_counter()

    for batch in getattr(scene, "polygon_batches", ()) or ():

        try:

            batch.dispose()

        except Exception:

            pass

    scene.render_resources.polygon_batches = []

    _remove_showcase_chests(scene)

    wall_tex = scene.wall_tex or load_texture(WALL1_TEXTURE_PATH)

    scene.render_resources.wall_tex = wall_tex

    tri_thickness = 5

    scene.build_state.showcase_polygons = []

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

    scene.build_state.showcase_polygons.append(
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

    scene.build_state.showcase_polygons.append(
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

    scene.build_state.showcase_polygons.append(
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

    scene.build_state.showcase_polygons.append(
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

    scene.build_state.showcase_polygons.append(
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

    sun_direction = (
        None if lighting is not None else getattr(scene, "sun_direction", None)
    )

    for polygon in scene.build_state.showcase_polygons:

        polygon.lighting = lighting

        polygon.sun_direction = sun_direction

    scene.render_resources.polygons.extend(scene.build_state.showcase_polygons)

    polygon_batch = build_polygon_render_batch(scene.build_state.showcase_polygons)

    scene.render_resources.polygon_batches = (
        [polygon_batch] if polygon_batch is not None else []
    )

    _build_showcase_chest(scene, wall_tex)


def _remove_showcase_chests(scene) -> None:

    _ensure_state_owners(scene)

    for chest in getattr(scene, "showcase_chests", ()) or ():

        _dispose_value(chest)

        for attr_name in ("entities", "immediate_entities", "wall_tiles", "chests"):

            values = getattr(scene, attr_name, None)

            if not isinstance(values, list):

                continue

            while chest in values:

                values.remove(chest)

    scene.build_state.showcase_chests = []


def _build_showcase_chest(scene, texture) -> None:

    _ensure_state_owners(scene)

    lighting = getattr(scene, "lighting", None)

    sun_direction = (
        None if lighting is not None else getattr(scene, "sun_direction", None)
    )

    x = float(scene.world_center.x)

    z = float(scene.world_center.z + 120.0)

    chest = Chest(
        Vector3(x, scene.ground_height_at(x, z), z),
        texture=Chest.texture_or_load(texture),
        lighting=lighting,
        sun_direction=sun_direction,
        side="south",
    )

    scene.build_state.showcase_chests = [chest]

    if not isinstance(getattr(scene, "chests", None), list):

        scene.build_state.chests = []

    scene.build_state.chests.append(chest)

    add_entity = getattr(scene, "add_entity", None)

    if callable(add_entity):

        add_entity(chest)

        return

    scene.render_resources.entities.append(chest)

    scene.render_resources.immediate_entities.append(chest)

    scene.render_resources.wall_tiles.extend(chest.get_collision_meshes())
