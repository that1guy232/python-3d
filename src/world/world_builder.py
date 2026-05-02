"""World object construction for WorldScene."""

from __future__ import annotations

import math
import random
import time
from typing import Iterator

from pygame.math import Vector3

from textures.resource_path import WALL1_TEXTURE_PATH
from textures.texture_utils import create_shadow_texture, load_texture
from world.decal import Decal
from world.decal_batch import DecalBatch
from world.objects import Road
from world.objects.building import Building
from world.objects.fence import build_textured_fence_ring
from world.objects.ground import TexturedGroundGridBuilder
from world.objects.polygon import Polygon
from world.objects.wall_tile import build_wall_tile_batches
from world.sprite import WorldSprite
from world.world_road_planner import create_building_access_roads
from world.world_spawner import spawn_world_sprites


CREATE_WORLD_OBJECT_STEPS = 10


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

    scene.builder = TexturedGroundGridBuilder(
        count=grid_count,
        tile_size=grid_tile_size,
        gap=grid_gap,
        texture=scene.ground_tex,
        brightness_modifiers=scene.brightness_modifiers,
        default_brightness=scene.camera.brightness_default,
    )

    scene.log_timing(label, start_time, time.perf_counter())
    yield (label, True)

    label = "Generating ground mesh"
    print("Generating ground mesh...")
    yield (label, False)
    start_time = time.perf_counter()
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
    sun_direction = getattr(scene, "sun_direction", None)

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
            roof_overhang=6.0,
        )
        for piece in pieces:
            piece.sun_direction = sun_direction
        scene.walls.extend(pieces)

    print(f"Built {len(scene.walls)} building pieces.")
    scene.wall_tile_batches = build_wall_tile_batches(
        scene.walls,
        camera=scene.camera,
        default_brightness=scene.camera.brightness_default,
        sun_direction=sun_direction,
    )
    scene.log_timing("Building pieces", start_time, time.perf_counter())
    scene.wall_tiles.extend(scene.walls)


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
    print("Spawning world objects...")
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

    if len(road_points) >= 2:
        x0, z0 = road_points[0]
        x1, z1 = road_points[1]
        t = 0.15
        sx = x0 + (x1 - x0) * t
        sz = z0 + (z1 - z0) * t
    else:
        sx, sz = road_points[0]

    min_x, max_x, min_z, max_z = scene.ground_bounds
    margin = 1.0
    sx = max(min_x + margin, min(max_x - margin, sx))
    sz = max(min_z + margin, min(max_z - margin, sz))

    scene.camera.position = scene.world_center

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
    )

    scene.log_timing("Create road", start_time, time.perf_counter())
    scene.others.append(scene.road)
    scene.roads = [scene.road]
    scene.building_roads = create_building_access_roads(
        scene,
        road_center_z=center_z,
        road_y=road_y,
        main_road_segment=(road_points[0], road_points[-1]),
    )
    scene.roads.extend(scene.building_roads)
    scene.others.extend(scene.building_roads)
    print(f"Built {len(scene.building_roads)} building access road segments.")
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
    )
    print(f"Built {len(scene.fence_meshes)} fence segments.")
    scene.log_timing("Build fences", start_time, time.perf_counter())

    start_time = time.perf_counter()
    scene.log_timing("Assemble static meshes", start_time, time.perf_counter())


def _build_shadow_decals(scene) -> None:
    shadow_subdiv = 4
    start_time = time.perf_counter()
    tree_shadow_texture = create_shadow_texture(
        width_px=256,
        height_px=256,
        max_alpha=0.8,
        inner_ratio=0.02,
        outer_ratio=1,
        falloff_exp=0.55,
        pixelated=True,
        pixel_scale=16,
    )
    building_shadow_texture = tree_shadow_texture
    print("Created shadow texture.")
    scene.log_timing("Create shadow texture", start_time, time.perf_counter())

    decals: list[Decal] = []
    rng = random.Random()

    def make_decal_for_sprite(sprite: WorldSprite) -> Decal:
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
            subdiv_u=shadow_subdiv,
            subdiv_v=shadow_subdiv,
            height_fn=scene.ground_height_at,
            elevation=1,
            uv_repeat=(1.0, 1.0),
            color=(1.0, 1.0, 1.0),
            # DecalBatch owns the final VBO; avoid building throwaway per-shadow VBOs.
            build_vbo=False,
        )

    def make_decal_for_building(building: Building, spec: dict) -> Decal | None:
        sun = getattr(scene, "sun_direction", None)
        if sun is None:
            return None

        proj_x = float(sun.x)
        proj_z = float(sun.z)
        proj_len = math.hypot(proj_x, proj_z)
        if proj_len < 1e-6:
            return None

        dir_x = proj_x / proj_len
        dir_z = proj_z / proj_len
        perp_x = -dir_z
        perp_z = dir_x

        min_x, max_x, min_z, max_z = building.bounds
        footprint_w = max(1.0, float(max_x - min_x))
        footprint_d = max(1.0, float(max_z - min_z))
        building_height = max(
            1.0,
            float(getattr(building, "height", spec.get("height", 50.0))),
        )
        vertical = max(0.05, abs(float(sun.y)))
        offset_len = min(700.0, building_height * proj_len / vertical)

        along = abs(footprint_w * dir_x) + abs(footprint_d * dir_z)
        across = abs(footprint_w * perp_x) + abs(footprint_d * perp_z)
        final_w = max(18.0, (along + offset_len) * 1.05)
        final_h = max(18.0, across * 1.15)

        center_x = float(building.position.x) + dir_x * offset_len * 0.5
        center_z = float(building.position.z) + dir_z * offset_len * 0.5
        center_y = scene.ground_height_at(center_x, center_z)
        rotation_deg = math.degrees(math.atan2(-dir_z, dir_x))
        building_shadow_subdiv = 2

        return Decal(
            center=Vector3(center_x, center_y, center_z),
            size=(final_w, final_h),
            texture=building_shadow_texture,
            rotation_deg=rotation_deg,
            subdiv_u=building_shadow_subdiv,
            subdiv_v=building_shadow_subdiv,
            height_fn=scene.ground_height_at,
            elevation=1.2,
            uv_repeat=(1.0, 1.0),
            color=(1.0, 1.0, 1.0),
            build_vbo=False,
        )

    start_time = time.perf_counter()
    for sprite in scene.trees:
        decals.append(make_decal_for_sprite(sprite))

    building_shadow_count = 0
    for building, spec in zip(
        getattr(scene, "buildings", ()),
        getattr(scene, "building_specs", ()),
    ):
        decal = make_decal_for_building(building, spec)
        if decal is not None:
            decals.append(decal)
            building_shadow_count += 1

    print(
        f"Created {len(scene.trees)} tree shadow decals and "
        f"{building_shadow_count} building shadow decals."
    )
    scene.log_timing("Create decals", start_time, time.perf_counter())
    scene.decal_batch = DecalBatch.build(decals)
    scene.decal_batches.append(scene.decal_batch)
    start_time = time.perf_counter()
    scene.log_timing("Build decal batch", start_time, time.perf_counter())
