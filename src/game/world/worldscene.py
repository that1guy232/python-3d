"""World scene orchestration.

The heavy lifting is split into focused modules:
- world_content.py: game-facing scene declarations
- world_lighting_plan.py: lighting derived from those declarations
- world_setup.py: rendering/input bootstrap and asset loading
- world_builder.py: ground, buildings, roads, sprites, fences, and decals
- world_runtime.py: update/input/height-query helpers
- world_road_planner.py: building driveway route planning
"""

from __future__ import annotations

from contextlib import nullcontext
import time
from typing import Callable, Iterator, Optional

from pygame.math import Vector3

from game.config import (
    FOV,
    HEIGHT,
    PERFORMANCE_SETUP_TIMING,
    STARTING_POS,
    WIDTH,
)
from engine.camera import Camera
from engine.core.scene import Scene
from engine.entity import Entity
from engine.rendering.decal import Decal
from engine.rendering.decal_batch import DecalBatch
from engine.rendering.sprite import WorldSprite
from game.world import world_builder, world_runtime, world_setup
from game.world.combat import BattleController
from game.world.collision_index import SceneCollisionIndex
from game.world.entity_registry import SceneEntityRegistry
from game.world.lighting_controller import StaticLightingController
from game.world.objects import Chest, WallTile
from game.world.objects.polygon import Polygon
from game.world.scene_resources import SceneResourceDisposer
from game.world.ui.interactions import WorldUIInteractions
from game.world.world_content import WorldContent
from game.world.world_renderer import WorldRenderer
from game.world.world_road_planner import create_building_access_roads


class WorldScene(Scene):
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
        world_content: WorldContent | None = None,
        defer_setup: bool = False,
    ) -> None:
        super().__init__()
        self.sprite_items: list[WorldSprite] = []
        self.entities: list[Entity] = []
        self.immediate_entities: list[Entity] = []
        self.decals: list[Decal] = []
        self.decal_batches: list[DecalBatch] = []
        self.wall_tiles: list[object] = []
        self.wall_tile_batches: list[object] = []
        self.road_batches: list[object] = []
        self.door_batches: list[object] = []
        self.window_batches: list[object] = []
        self.polygons: list[Polygon] = []
        self.polygon_batches: list[object] = []
        self.chests: list[Chest] = []
        self.showcase_chests: list[Chest] = []
        self.inventory_items: list[object] = []
        self.others: list[object] = []
        self._texture_lighting_sync_key = None
        self._texture_lighting_sync_result = False
        self._collision_spatial_index = None
        self.collision_index = SceneCollisionIndex(self)
        self.entity_registry = SceneEntityRegistry(self)
        self.combat = BattleController(self)
        self.ui_interactions = WorldUIInteractions(self)
        self.lighting_controller = StaticLightingController(self)
        self.resource_disposer = SceneResourceDisposer(self)

        self.renderer = WorldRenderer(self)
        self._initialized = False
        self._grid_count = grid_count
        self._grid_tile_size = grid_tile_size
        self._grid_gap = grid_gap
        self._tree_count = tree_count
        self._grass_count = grass_count
        self._rock_count = rock_count
        self.building_count = building_count
        self.world_content = world_content

        spacing = grid_tile_size + grid_gap
        half = grid_tile_size / 2.0
        self._grid_spacing = spacing
        self._grid_half = half
        self.world_center = Vector3(
            (grid_count * spacing) / 2, 0, (grid_count * spacing) / 2
        )

        print("World Scene Initialized")

        self.camera = camera or Camera(
            position=Vector3(STARTING_POS),
            width=WIDTH,
            height=HEIGHT,
            fov=FOV,
            default_brightness=0.8,
        )

        self.ground_bounds = (
            0 + half,
            grid_count * spacing - half,
            0 + half,
            grid_count * spacing - half,
        )

        if not defer_setup:
            for _label, _progress in self.initialize_steps():
                pass

    def initialize_steps(self) -> Iterator[tuple[str, float]]:
        """Initialize the scene incrementally for loading-screen rendering."""
        if self._initialized:
            yield ("Ready", 1.0)
            return

        setup_steps: list[tuple[str, Callable[[], None]]] = [
            (
                "Setting up brightness areas",
                lambda: self._setup_brightness_areas(
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
        for label, finished in object_steps:
            if finished:
                completed_steps += 1
            yield (label, completed_steps / total_steps)

        self.log_timing("Creating world objects", start_time, time.perf_counter())
        self._initialized = True
        world_runtime.initialize_player_spawn_height(self)
        self._last_static_lighting_brightness = float(
            getattr(self.camera, "brightness_default", 1.0)
        )
        self._sync_lighting_uniforms(compile_shader=False)
        print("World scene initialization complete.")
        yield ("Ready", 1.0)

    def _setup_brightness_areas(
        self, grid_count: int, spacing: float, half: float
    ) -> None:
        return world_setup.setup_brightness_areas(self, grid_count, spacing, half)

    def _setup_controllers(self) -> None:
        return world_setup.setup_controllers(self)

    def _setup_graphics(self) -> None:
        return world_setup.setup_graphics(self)

    def _load_assets(self) -> None:
        return world_setup.load_assets(self)

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

    def log_timing(
        self, message: str, start_time: float, end_time: float, log: bool | None = None
    ):
        """Logs timing information for WorldScene setup phases."""
        if log is None:
            log = bool(PERFORMANCE_SETUP_TIMING)
        if log:
            print(f"[setup] {message} took {end_time - start_time:.6f} seconds")

    def _profile(self, name: str):
        profiler = getattr(self, "profiler", None)
        if profiler is None or not getattr(profiler, "enabled", False):
            return nullcontext()
        return profiler.section(name)

    def add_entity(self, entity: Entity) -> Entity:
        """Register a runtime entity and its scene-facing resources."""
        return self.entity_registry.add(entity)

    def remove_entity(self, entity: Entity) -> None:
        """Unregister a runtime entity and its scene-facing resources."""
        return self.entity_registry.remove(entity)

    def start_battle(self, goblin: Entity) -> bool:
        return self.combat.start(goblin)

    def player_attack_damage_preview(self) -> int:
        return self.combat.player_attack_damage_preview()

    def roll_player_attack_damage(self) -> tuple[int, bool]:
        return self.combat.roll_player_attack_damage()

    def damage_battle_goblin(self, amount: int | None = None) -> int:
        return self.combat.damage_active_goblin(amount)

    def remove_battle_goblin(self) -> bool:
        return self.combat.remove_active_goblin()

    def end_battle(self) -> None:
        return self.combat.end()

    def refresh_immediate_entities(self) -> None:
        return self.entity_registry.refresh_immediate()

    def _sync_lighting_aliases(self):
        return self.lighting_controller.sync_aliases()

    def invalidate_texture_lighting_cache(self) -> None:
        return self.lighting_controller.invalidate_texture_lighting_cache()

    def draw_sky(self) -> None:  # pragma: no cover - visual
        return self.renderer.draw_sky()

    def draw(self, enable_timing: bool = False):  # pragma: no cover - visual
        return self.renderer.draw(enable_timing=enable_timing)

    def draw_world_objects(self, enable_timing: bool = False):  # pragma: no cover - visual
        return self.renderer.draw_world_objects(enable_timing=enable_timing)

    def contains_horizontal(self, pos: Vector3) -> bool:
        return world_runtime.contains_horizontal(self, pos)

    def is_on_road(self, x: float, z: float, *, margin: float = 0.0) -> bool:
        return world_runtime.is_on_road(self, x, z, margin=margin)

    def invalidate_collision_index(self) -> None:
        return self.collision_index.invalidate()

    def rebuild_collision_index(self) -> dict:
        return self.collision_index.rebuild()

    def collision_meshes_for_bounds(
        self,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        *,
        include_polygons: bool = True,
    ) -> list:
        return self.collision_index.meshes_for_bounds(
            min_x,
            max_x,
            min_z,
            max_z,
            include_polygons=include_polygons,
        )

    def collision_meshes_at(
        self,
        x: float,
        z: float,
        radius: float,
        *,
        include_polygons: bool = True,
    ) -> list:
        return self.collision_index.meshes_at(
            x,
            z,
            radius,
            include_polygons=include_polygons,
        )

    def ground_height_at(self, x: float, z: float) -> float:
        return world_runtime.ground_height_at(self, x, z)

    def view_space_position(
        self, *, dist: float, nx: float, ny: float, px: float = 0.0, py: float = 0.0
    ) -> Vector3:
        return world_runtime.view_space_position(
            self, dist=dist, nx=nx, ny=ny, px=px, py=py
        )

    def focused_interaction_prompt(self) -> str | None:
        return world_runtime.focused_interaction_prompt(self)

    def update(self, dt: float) -> None:
        return world_runtime.update(self, dt)

    def handle_event(self, event) -> None:
        return world_runtime.handle_event(self, event)

    def _compute_pause_buttons(self, width: int = WIDTH, height: int = HEIGHT):
        return self.ui_interactions.compute_pause_buttons(width=width, height=height)

    def _compute_battle_buttons(self, width: int = WIDTH, height: int = HEIGHT):
        return self.ui_interactions.compute_battle_buttons(width=width, height=height)

    def _handle_battle_click(self, pos):
        return self.ui_interactions.handle_battle_click(pos)

    def _handle_battle_motion(self, pos):
        return self.ui_interactions.handle_battle_motion(pos)

    def _handle_battle_release(self, pos):
        return self.ui_interactions.handle_battle_release(pos)

    def _handle_inventory_click(self, pos):
        return self.ui_interactions.handle_inventory_click(pos)

    def _handle_pause_click(self, pos):
        return self.ui_interactions.handle_pause_click(pos)

    def _handle_pause_motion(self, pos):
        return self.ui_interactions.handle_pause_motion(pos)

    def _handle_pause_release(self, pos):
        return self.ui_interactions.handle_pause_release(pos)

    def _draw_pause_menu(self, text):
        return self.renderer.draw_pause_menu(text)

    def render(
        self, *, show_hud: bool = True, text=None, fps: float | None = None
    ):  # pragma: no cover - visual
        return self.renderer.render(show_hud=show_hud, text=text, fps=fps)

    def apply_mouse_delta(self, dx: float, dy: float, dt: float | None = None) -> None:
        return world_runtime.apply_mouse_delta(self, dx, dy, dt)

    def set_brightness(self, value: float) -> float:
        return self.lighting_controller.set_brightness(value)

    def _sync_brightness_modifiers_from_camera(self) -> None:
        return self.lighting_controller.sync_brightness_modifiers_from_camera()

    @staticmethod
    def _dispose_renderable(obj) -> None:
        return SceneResourceDisposer.dispose_renderable(obj)

    def refresh_static_lighting(self) -> None:
        return self.lighting_controller.refresh_static()

    def apply_static_exposure(self, brightness: float) -> None:
        return self.lighting_controller.apply_static_exposure(brightness)

    def _sync_lighting_uniforms(
        self,
        *,
        base_brightness: float | None = None,
        compile_shader: bool = True,
    ) -> bool:
        return self.lighting_controller.sync_uniforms(
            base_brightness=base_brightness,
            compile_shader=compile_shader,
        )

    def _texture_lighting_fast_key(
        self,
        *,
        brightness: float,
        lighting,
        sun_direction,
        brightness_areas,
        covered_regions,
        compile_shader: bool,
    ):
        return self.lighting_controller.texture_lighting_fast_key(
            brightness=brightness,
            lighting=lighting,
            sun_direction=sun_direction,
            brightness_areas=brightness_areas,
            covered_regions=covered_regions,
            compile_shader=compile_shader,
        )

    @staticmethod
    def _collection_identity_key(values):
        return StaticLightingController.collection_identity_key(values)

    @classmethod
    def _texture_lighting_key(
        cls,
        *,
        brightness: float,
        lighting,
        sun_direction,
        brightness_areas,
        covered_regions,
        compile_shader: bool,
    ):
        return StaticLightingController.texture_lighting_key(
            brightness=brightness,
            lighting=lighting,
            sun_direction=sun_direction,
            brightness_areas=brightness_areas,
            covered_regions=covered_regions,
            compile_shader=compile_shader,
        )

    @staticmethod
    def _rounded(value, digits: int = 5):
        return StaticLightingController.rounded(value, digits=digits)

    @classmethod
    def _vector_key(cls, value):
        return StaticLightingController.vector_key(value)

    @classmethod
    def _brightness_areas_key(cls, areas):
        return StaticLightingController.brightness_areas_key(areas)

    @classmethod
    def _covered_regions_key(cls, regions):
        return StaticLightingController.covered_regions_key(regions)

    @classmethod
    def _opening_key(cls, opening):
        return StaticLightingController.opening_key(opening)

    @classmethod
    def _bounds_key(cls, bounds):
        return StaticLightingController.bounds_key(bounds)

    @staticmethod
    def _uses_texture_shader(obj) -> bool:
        return StaticLightingController.uses_texture_shader(obj)

    @staticmethod
    def _set_exposure_cpu(obj, exposure: float) -> None:
        return StaticLightingController.set_exposure_cpu(obj, exposure)

    def _apply_untextured_static_exposure_cpu(self, exposure: float) -> None:
        return self.lighting_controller.apply_untextured_static_exposure_cpu(exposure)

    def _apply_static_exposure_cpu(self, exposure: float) -> None:
        return self.lighting_controller.apply_static_exposure_cpu(exposure)

    def _road_lighting_candidates(self):
        return self.lighting_controller.road_lighting_candidates()

    @staticmethod
    def _dispose_renderable_batches(values) -> None:
        return SceneResourceDisposer.dispose_renderable_batches(values)

    def dispose(self) -> None:
        return self.resource_disposer.dispose()
