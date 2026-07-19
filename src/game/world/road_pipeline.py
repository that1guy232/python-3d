"""Road construction pipeline for world scenes."""

from __future__ import annotations


import time


from game.world.builder_support import _dispose_value, _dispose_values

from game.world.objects import Road

from game.world.objects.road import build_road_render_batch

from game.world.world_road_planner import create_building_access_roads


def _build_road_batches(scene) -> None:

    _dispose_values(getattr(scene, "road_batches", ()))

    roads = getattr(scene, "roads", ()) or ()

    for road in roads:

        setattr(road, "render_batched", False)

    road_batch = build_road_render_batch(roads)

    scene.road_batches = [road_batch] if road_batch is not None else []


def _build_roads(scene) -> None:

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

    packet_backend = getattr(scene, "lighting_backend", "legacy") == "packet"
    scene.road = Road(
        points=road_points,
        ground_y=road_y,
        width=road_width,
        texture=scene.road_tex,
        v_tiles=1.0,
        height_sampler=scene._ground_height_sampler,
        elevation=3.0,
        segment_length=8.0,
        brightness_modifiers=(
            () if packet_backend else getattr(scene, "brightness_modifiers", ())
        ),
        default_brightness=scene.camera.brightness_default,
        lighting=getattr(scene, "lighting", None),
        sun_direction=(None if packet_backend else getattr(scene, "sun_direction", None)),
        dynamic_lighting=packet_backend,
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
