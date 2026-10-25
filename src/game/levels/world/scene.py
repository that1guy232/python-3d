"""The outdoor world level; shared gameplay lives in game.gameplay."""
from __future__ import annotations

import time
from typing import Callable, Iterator, Optional
from pygame.math import Vector3
from engine.camera import Camera
from game.config import SKELETON_COUNT
from game.gameplay.scene import PlayableScene
from game.gameplay import runtime as world_runtime, setup as world_setup
from game.levels.world.content import WorldContent
from game.levels.world.generation import builder as world_builder
from game.levels.world.generation.road_planner import create_building_access_roads


class WorldScene(PlayableScene):
    def __init__(
        self,
        camera: Optional[Camera] = None,
        *,
        grid_count: int = 200,
        grid_tile_size: int = 25,
        grid_gap: int = 0,
        tree_count: int = 1000,
        grass_count: int = 2000,
        rock_count: int = 750,
        building_count: int = 10,
        skeleton_count: int = SKELETON_COUNT,
        world_content: WorldContent | None = None,
        world_random_seed: int | None = None,
        defer_setup: bool = False,
        lighting_backend: str = "packet",
    ) -> None:
        super().__init__(camera, lighting_backend=lighting_backend, world_random_seed=world_random_seed)
        self.skeleton_count = max(0, int(skeleton_count))
        self._grid_count = grid_count
        self._grid_tile_size = grid_tile_size
        self._grid_gap = grid_gap
        self._tree_count = tree_count
        self._grass_count = grass_count
        self._rock_count = rock_count
        self.building_count = building_count
        self.world_content = world_content
        self.world_random_seed = world_random_seed

        spacing = grid_tile_size + grid_gap
        half = grid_tile_size / 2.0
        self._grid_spacing = spacing
        self._grid_half = half
        terrain_maximum = (max(1, int(grid_count)) - 1) * spacing + half
        self.terrain_bounds = (
            -half,
            terrain_maximum,
            -half,
            terrain_maximum,
        )
        terrain_center = (-half + terrain_maximum) * 0.5
        self.world_center = Vector3(
            terrain_center,
            0,
            terrain_center,
        )

        print("World Scene Initialized")

        self.ground_bounds = (
            0.0,
            max(0.0, (grid_count - 1) * spacing),
            0.0,
            max(0.0, (grid_count - 1) * spacing),
        )

        if not defer_setup:
            with self._resources.rollback_on_error():
                for _label, _progress in self.initialize_steps():
                    pass


    def initialize_steps(self) -> Iterator[tuple[str, float]]:
        """Initialize the scene incrementally for loading-screen rendering."""
        if self._initialized:
            yield ("Ready", 1.0)
            return

        setup_steps: list[tuple[str, Callable[[], None]]] = [
            (
                "Setting up local lights",
                lambda: self._setup_local_lights(
                    self._grid_count, self._grid_spacing, self._grid_half
                ),
            ),
            ("Setting up controllers", self._setup_controllers),
            ("Setting up graphics", self._setup_graphics),
            ("Loading assets", self._load_assets),
        ]
        world_object_steps = world_builder.create_world_object_step_count(
            self,
            self._grid_count,
            self._grid_spacing,
            self._grid_half,
            self._grid_tile_size,
            self._grid_gap,
            self._tree_count,
            self._grass_count,
            self._rock_count,
        )
        total_steps = len(setup_steps) + world_object_steps
        completed_steps = 0

        for label, step in setup_steps:
            yield (label, completed_steps / total_steps)
            start_time = time.perf_counter()
            step()
            self.log_timing(label, start_time, time.perf_counter())
            completed_steps += 1
            yield (label, completed_steps / total_steps)

        start_time = time.perf_counter()
        object_steps = self._create_world_objects_steps(
            self._grid_count,
            self._grid_spacing,
            self._grid_half,
            self._grid_tile_size,
            self._grid_gap,
            self._tree_count,
            self._grass_count,
            self._rock_count,
        )
        for label, step_progress in object_steps:
            step_progress = max(0.0, min(1.0, float(step_progress)))
            yield (
                label,
                (completed_steps + step_progress) / total_steps,
            )
            if step_progress >= 1.0:
                completed_steps += 1

        self.log_timing("Creating world objects", start_time, time.perf_counter())
        self._initialized = True
        world_runtime.initialize_player_spawn_height(self)
        self._last_static_lighting_brightness = float(
            getattr(self.camera, "brightness_default", 1.0)
        )
        self.lighting_controller.sync_uniforms(compile_shader=False)
        print("World scene initialization complete.")
        yield ("Ready", 1.0)


    def _setup_local_lights(
        self, grid_count: int, spacing: float, half: float
    ) -> None:
        return world_setup.setup_local_lights(self, grid_count, spacing, half)


    def _create_building_specs(self, count: int = 10) -> list[dict]:
        return world_builder.create_building_specs(self, count=count)


    def _create_building_access_roads(
        self,
        *,
        road_center_z: float,
        road_y: float,
        main_road_segment: tuple[tuple[float, float], tuple[float, float]],
    ):
        return create_building_access_roads(
            self,
            road_center_z=road_center_z,
            road_y=road_y,
            main_road_segment=main_road_segment,
        )


    def _create_world_objects(
        self,
        grid_count: int,
        spacing: float,
        half: float,
        grid_tile_size: int,
        grid_gap: int,
        tree_count: int,
        grass_count: int,
        rock_count: int,
    ) -> None:
        return world_builder.create_world_objects(
            self,
            grid_count,
            spacing,
            half,
            grid_tile_size,
            grid_gap,
            tree_count,
            grass_count,
            rock_count,
        )


    def _create_world_objects_steps(
        self,
        grid_count: int,
        spacing: float,
        half: float,
        grid_tile_size: int,
        grid_gap: int,
        tree_count: int,
        grass_count: int,
        rock_count: int,
    ) -> Iterator[tuple[str, bool]]:
        return world_builder.create_world_objects_steps(
            self,
            grid_count,
            spacing,
            half,
            grid_tile_size,
            grid_gap,
            tree_count,
            grass_count,
            rock_count,
        )


    def rebuild_legacy_road_batches(self):
        world_builder._build_road_batches(self)

    def rebuild_legacy_fences(self):
        world_builder._build_fences(self)
