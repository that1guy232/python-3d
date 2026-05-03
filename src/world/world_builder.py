"""World object construction for WorldScene."""

from __future__ import annotations

import math
import random
import time
from typing import Iterator

from pygame.math import Vector3

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
from world.objects import Road, Torch
from world.objects.building import Building
from world.objects.fence import build_textured_fence_ring
from world.objects.ground import TexturedGroundGridBuilder
from world.objects.polygon import Polygon
from world.objects.wall_tile import build_wall_tile_batches
from world.world_road_planner import create_building_access_roads
from world.world_spawner import spawn_world_sprites


CREATE_WORLD_OBJECT_STEPS = 10
BUILDING_ROOF_OVERHANG = 6.0
SHADOW_BUILDING_CLIP_MARGIN = 2.0

_SIDE_NORMALS = {
    "north": (0.0, 1.0),
    "east": (1.0, 0.0),
    "south": (0.0, -1.0),
    "west": (-1.0, 0.0),
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
        regions.append(
            {
                "min_x": min_x,
                "max_x": max_x,
                "min_z": min_z,
                "max_z": max_z,
                "factor": INDOOR_LIGHT_FACTOR,
                "doorway": {
                    "side": side,
                    "center_x": x,
                    "center_z": z,
                    "width": max(doorway_width * 1.16, doorway_width + 8.0),
                    "depth": doorway_depth,
                    "side_fade": max(10.0, doorway_width * 0.26),
                    "edge_factor": 1.0,
                },
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


def _install_building_torch_lights(scene) -> None:
    modifiers = Torch.brightness_modifiers_for_building_specs(
        getattr(scene, "building_specs", ()) or ()
    )
    scene.torch_light_modifiers = modifiers
    scene.doorway_light_modifiers = []
    if not hasattr(scene, "brightness_modifiers") or scene.brightness_modifiers is None:
        scene.brightness_modifiers = []
    scene.brightness_modifiers.extend(modifiers)

    camera = getattr(scene, "camera", None)
    for modifier in modifiers:
        Torch.install_brightness_modifier(camera, modifier)


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

        doorway_span = width if doorway_side in {"north", "south"} else depth
        specs.append(
            {
                "position": Vector3(x, 0, z),
                "width": width,
                "depth": depth,
                "height": 50,
                "doorway_side": doorway_side,
                "doorway_width": min(90.0, max(36.0, doorway_span * 0.22)),
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
        scene.buildings.append(Building(position=spec["position"]))
    scene.covered_regions = _building_covered_regions(scene)
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

        pieces = building.create_perimeter_walls(
            wall_height=spec["height"],
            wall_thickness=2.5,
            width=spec["width"],
            depth=spec["depth"],
            max_tile_width=max(spec["width"], spec["depth"]),
            texture=wall_tex,
            uv_repeat=(1.0, 1.0),
            base_y=base_y,
            doorway_side=spec["doorway_side"],
            doorway_width=spec["doorway_width"],
            doorway_height=spec["height"] * 0.68,
            roof=True,
            roof_thickness=4.0,
            roof_overhang=BUILDING_ROOF_OVERHANG,
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


def _build_showcase_polygons(scene) -> None:
    start_time = time.perf_counter()
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

        sun = getattr(scene, "sun_direction", None)
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
