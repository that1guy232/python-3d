"""World object construction orchestration for WorldScene."""

from __future__ import annotations

import time
from typing import Iterator

from game.world.builder_support import WorldObjectBuildStep
from game.world.building_pipeline import (
    _build_buildings,
    _build_showcase_polygons_and_collision,
    _prepare_buildings,
    create_building_specs as create_building_specs,
)
from game.world.detail_pipeline import _build_shadow_decals
from game.world.road_pipeline import (
    _build_road_batches as _build_road_batches,
    _build_roads,
)
from game.world.spawn_pipeline import (
    _build_goblins,
    _spawn_grass,
    _spawn_rocks,
    _spawn_trees,
)
from game.world.terrain_pipeline import (
    _build_fences as _build_fences,
    _generate_ground_mesh,
)


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
    for _label, _progress in create_world_objects_steps(
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
) -> Iterator[tuple[str, float]]:
    """Run world build phases and expose fractional progress within a phase."""
    for step in create_world_object_step_specs(
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
        if step.message:
            print(step.message)

        yield (step.label, 0.0)
        start_time = time.perf_counter()
        incremental_progress = step.action()

        if incremental_progress is not None:
            for fraction in incremental_progress:
                yield (step.label, max(0.0, min(1.0, float(fraction))))

        scene.log_timing(step.label, start_time, time.perf_counter())
        yield (step.label, 1.0)


def create_world_object_step_count(
    scene,
    grid_count: int,
    spacing: float,
    half: float,
    grid_tile_size: int,
    grid_gap: int,
    tree_count: int,
    grass_count: int,
    rock_count: int,
) -> int:
    return len(
        create_world_object_step_specs(
            scene,
            grid_count,
            spacing,
            half,
            grid_tile_size,
            grid_gap,
            tree_count,
            grass_count,
            rock_count,
        )
    )


def create_world_object_step_specs(
    scene,
    grid_count: int,
    spacing: float,
    half: float,
    grid_tile_size: int,
    grid_gap: int,
    tree_count: int,
    grass_count: int,
    rock_count: int,
) -> tuple[WorldObjectBuildStep, ...]:
    del spacing, half

    return (
        WorldObjectBuildStep(
            "Creating buildings",
            lambda: _prepare_buildings(scene, grid_count, grid_tile_size, grid_gap),
            "Creating buildings...",
        ),
        WorldObjectBuildStep(
            "Generating ground mesh",
            lambda: _generate_ground_mesh(scene),
            "Generating ground mesh...",
        ),
        WorldObjectBuildStep("Building structures", lambda: _build_buildings(scene)),
        WorldObjectBuildStep(
            "Building showcase polygons",
            lambda: _build_showcase_polygons_and_collision(scene),
        ),
        WorldObjectBuildStep(
            "Creating roads",
            lambda: _build_roads(scene),
            "Creating roads...",
        ),
        WorldObjectBuildStep("Spawning trees", lambda: _spawn_trees(scene, tree_count)),
        WorldObjectBuildStep("Spawning goblins", lambda: _build_goblins(scene)),
        WorldObjectBuildStep(
            "Spawning grass", lambda: _spawn_grass(scene, grass_count)
        ),
        WorldObjectBuildStep("Spawning rocks", lambda: _spawn_rocks(scene, rock_count)),
        WorldObjectBuildStep("Building fences", lambda: _build_fences(scene)),
        WorldObjectBuildStep(
            "Adding ground details", lambda: _build_shadow_decals(scene)
        ),
    )
