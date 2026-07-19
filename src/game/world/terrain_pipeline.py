"""Terrain and boundary construction pipeline for world scenes."""

from __future__ import annotations


import time


from game.world.builder_support import _dispose_value, _dispose_values

from game.world.objects.fence import build_textured_fence_ring


def _generate_ground_mesh(scene) -> None:

    _dispose_value(getattr(scene, "ground_mesh", None))

    scene.ground_mesh = scene.builder.build()

    scene._ground_height_sampler = getattr(scene.ground_mesh, "height_sampler", None)


def _build_fences(scene) -> None:

    start_time = time.perf_counter()

    _dispose_values(getattr(scene, "fence_meshes", ()))

    packet_backend = getattr(scene, "lighting_backend", "legacy") == "packet"
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
        brightness_modifiers=(
            () if packet_backend else getattr(scene, "brightness_modifiers", ())
        ),
        default_brightness=scene.camera.brightness_default,
        lighting=getattr(scene, "lighting", None),
        sun_direction=(None if packet_backend else getattr(scene, "sun_direction", None)),
        dynamic_lighting=packet_backend,
    )

    print(f"Built {len(scene.fence_meshes)} fence segments.")

    scene.log_timing("Build fences", start_time, time.perf_counter())

    start_time = time.perf_counter()

    scene.log_timing("Assemble static meshes", start_time, time.perf_counter())
