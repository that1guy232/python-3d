"""World object construction for WorldScene."""

from __future__ import annotations

import math
import random
import time
from typing import Iterator

from pygame.math import Vector3

from game.config import (
    GOBLIN_CHASE_GIVE_UP_RADIUS,
    GOBLIN_CHASE_RADIUS,
    GOBLIN_COUNT,
    GOBLIN_MIN_SEPARATION,
    GOBLIN_SPAWN_ATTEMPTS,
    GOBLIN_SPAWN_CLEARANCE,
    GOBLIN_SPAWN_TREE_RADIUS,
)
from engine.rendering.decal import Decal
from engine.rendering.decal_batch import DecalBatch
from engine.rendering.sprite import WorldSprite
from game.resources.paths import WALL1_TEXTURE_PATH
from engine.textures.texture_utils import (
    create_forest_floor_texture,
    create_shadow_texture,
    load_texture,
)
from game.world.objects import Door, Goblin, Road, Torch, Window
from game.world.objects.building import Building
from game.world.objects.door import build_door_render_batch
from game.world.objects.fence import build_textured_fence_ring
from game.world.objects.ground import TexturedGroundGridBuilder
from game.world.objects.polygon import Polygon, build_polygon_render_batch
from game.world.objects.road import build_road_render_batch
from game.world.objects.window import build_window_render_batch
from game.world.world_content import create_world_content, resolve_world_content
from game.world.world_lighting_plan import apply_building_lighting
from game.world.objects.wall_tile import build_wall_tile_batches
from game.world.world_road_planner import create_building_access_roads
from game.world.world_spawner import spawn_world_sprites


CREATE_WORLD_OBJECT_STEPS = 11
BUILDING_ROOF_OVERHANG = 6.0
BUILDING_WALL_TERRAIN_EMBED_DEPTH = 8.0
BUILDING_WALL_TERRAIN_SAMPLE_SPACING = 18.0
SHADOW_BUILDING_CLIP_MARGIN = 2.0

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
    return create_world_content(scene, building_count=count).to_building_specs()


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
    content = resolve_world_content(
        scene,
        building_count=int(getattr(scene, "building_count", 10)),
    )
    scene.building_specs = content.to_building_specs()
    for spec in scene.building_specs:
        _normalize_window_specs_for_building(spec)
    for spec in scene.building_specs:
        scene.buildings.append(Building(position=spec["position"]))
    apply_building_lighting(scene, scene.building_specs)

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

    label = "Adding ground details"
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


def _goblin_position_blocker(scene, *, block_roads: bool = True):
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

        if block_roads and scene.is_on_road(x, z, margin=clearance):
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


def _player_in_building_detector(scene):
    def player_in_building() -> bool:
        camera = getattr(scene, "camera", None)
        position = getattr(camera, "position", None)
        if position is None:
            return False

        x = float(position.x)
        z = float(position.z)
        for building in getattr(scene, "buildings", ()) or ():
            contains_point = getattr(building, "contains_point", None)
            if callable(contains_point) and contains_point(x, z, margin=0.0):
                return True
        return False

    return player_in_building


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
    spawn_blocked = _goblin_position_blocker(scene, block_roads=True)
    movement_blocked = _goblin_position_blocker(scene, block_roads=False)
    clearance = max(
        float(GOBLIN_SPAWN_CLEARANCE),
        float(getattr(Goblin, "DEFAULT_HEIGHT", 0.0)) * 0.3,
    )
    min_separation_sq = max(0.0, float(GOBLIN_MIN_SEPARATION)) ** 2
    max_attempts = max(1, int(GOBLIN_SPAWN_ATTEMPTS))
    add_entity = getattr(scene, "add_entity", None)
    player_in_building = _player_in_building_detector(scene)
    chase_radius = float(getattr(scene, "goblin_chase_radius", GOBLIN_CHASE_RADIUS))
    chase_give_up_radius = float(
        getattr(scene, "goblin_chase_give_up_radius", GOBLIN_CHASE_GIVE_UP_RADIUS)
    )

    for _ in range(count):
        spawn = None
        for _attempt in range(max_attempts):
            x, z = _random_goblin_spawn_near_tree(scene, rng)
            if spawn_blocked(x, z, clearance):
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
                position_blocked=movement_blocked,
                player_in_building=player_in_building,
                chase_radius=chase_radius,
                chase_give_up_radius=chase_give_up_radius,
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
    tree_detail_subdiv = 4
    start_time = time.perf_counter()
    tree_shadow_texture = create_shadow_texture(
        width_px=160,
        height_px=128,
        max_alpha=0.46,
        inner_ratio=0.16,
        outer_ratio=0.96,
        falloff_exp=1.55,
        pixelated=False,
    )
    tree_detail_textures = [
        create_forest_floor_texture(
            width_px=192,
            height_px=192,
            variant_seed=seed,
            alpha_scale=0.86,
            pixelated=False,
        )
        for seed in range(4)
    ]
    contact_shadow_texture = create_shadow_texture(
        width_px=128,
        height_px=128,
        max_alpha=0.34,
        inner_ratio=0.18,
        outer_ratio=0.98,
        falloff_exp=1.55,
        pixelated=False,
    )
    print("Created ground detail and contact shadow textures.")
    scene.log_timing(
        "Create ground detail textures",
        start_time,
        time.perf_counter(),
    )

    decals: list[Decal] = []
    rng = random.Random()
    shadow_receiver = _outside_building_shadow_receiver(
        _building_shadow_clip_bounds(scene)
    )

    def make_tree_detail_decal_for_sprite(sprite: WorldSprite) -> Decal:
        w, h = sprite.size
        footprint = max(12.0, max(w * 0.58, h * 0.28))
        final_w = max(18.0, min(72.0, footprint * rng.uniform(0.82, 1.28)))
        final_h = max(14.0, min(58.0, footprint * rng.uniform(0.62, 1.04)))
        rot = rng.uniform(0.0, 360.0)
        offset_radius = min(final_w, final_h) * rng.uniform(0.0, 0.14)
        offset_angle = rng.uniform(0.0, math.tau)
        offset_x = math.cos(offset_angle) * offset_radius
        offset_z = math.sin(offset_angle) * offset_radius

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
            texture=rng.choice(tree_detail_textures),
            rotation_deg=rot,
            subdiv_u=tree_detail_subdiv,
            subdiv_v=tree_detail_subdiv,
            height_fn=scene.ground_height_at,
            elevation=0.82,
            uv_repeat=(1.0, 1.0),
            color=(1.0, 1.0, 1.0),
            receiver_fn=shadow_receiver,
            # DecalBatch owns the final VBO; avoid building throwaway per-decal VBOs.
            build_vbo=False,
        )

    def make_tree_shadow_decal_for_sprite(sprite: WorldSprite) -> Decal:
        w, h = sprite.size
        base_width = max(20.0, min(78.0, w * rng.uniform(0.55, 0.82)))
        base_depth = max(14.0, min(54.0, h * rng.uniform(0.18, 0.32)))
        center_y = scene.ground_height_at(sprite.position.x, sprite.position.z)

        return Decal(
            center=Vector3(sprite.position.x, center_y, sprite.position.z),
            size=(base_width, base_depth),
            texture=tree_shadow_texture,
            rotation_deg=rng.uniform(0.0, 360.0),
            subdiv_u=3,
            subdiv_v=3,
            height_fn=scene.ground_height_at,
            elevation=0.68,
            uv_repeat=(1.0, 1.0),
            color=(1.0, 1.0, 1.0),
            receiver_fn=shadow_receiver,
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
        decals.append(make_tree_shadow_decal_for_sprite(sprite))
        decals.append(make_tree_detail_decal_for_sprite(sprite))
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
        f"Created {len(scene.trees)} tree shadow/detail pairs, "
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
