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
from game.world import world_builder, world_runtime, world_setup
from game.world.combat import BattleController
from game.world.collision_index import SceneCollisionIndex
from game.world.entity_registry import SceneEntityRegistry
from game.world.lighting_controller import StaticLightingController
from game.world.scene_resources import SceneResourceDisposer
from game.world.ui.interactions import WorldUIInteractions
from game.world.world_content import WorldContent
from game.world.world_renderer import WorldRenderer
from game.world.world_road_planner import create_building_access_roads
from game.world.world_state import (
    WorldBuildState,
    WorldRenderResources,
    WorldUIState,
    state_alias,
)


class WorldScene(Scene):
    # Temporary source-compatible aliases. New subsystem code should use the
    # explicit owner (`build_state`, `render_resources`, or `ui_state`).
    sprite_items = state_alias("render_resources", "sprite_items")
    entities = state_alias("render_resources", "entities")
    immediate_entities = state_alias("render_resources", "immediate_entities")
    decals = state_alias("render_resources", "decals")
    decal_batches = state_alias("render_resources", "decal_batches")
    wall_tiles = state_alias("render_resources", "wall_tiles")
    wall_tile_batches = state_alias("render_resources", "wall_tile_batches")
    road_batches = state_alias("render_resources", "road_batches")
    door_batches = state_alias("render_resources", "door_batches")
    window_batches = state_alias("render_resources", "window_batches")
    polygons = state_alias("render_resources", "polygons")
    polygon_batches = state_alias("render_resources", "polygon_batches")
    others = state_alias("render_resources", "others")
    fence_meshes = state_alias("render_resources", "fence_meshes")
    ground_mesh = state_alias("render_resources", "ground_mesh")
    sky = state_alias("render_resources", "sky")
    road = state_alias("render_resources", "road")
    decal_batch = state_alias("render_resources", "decal_batch")
    _ground_height_sampler = state_alias("render_resources", "ground_height_sampler")
    _collision_spatial_index = state_alias(
        "render_resources", "collision_spatial_index"
    )
    _sprite_update_cache = state_alias("render_resources", "sprite_update_cache")
    ground_tex = state_alias("render_resources", "ground_tex")
    road_tex = state_alias("render_resources", "road_tex")
    tree_textures = state_alias("render_resources", "tree_textures")
    grasses_textures = state_alias("render_resources", "grasses_textures")
    rock_textures = state_alias("render_resources", "rock_textures")
    fence_textures = state_alias("render_resources", "fence_textures")
    equipment_slot_textures = state_alias(
        "render_resources", "equipment_slot_textures"
    )
    wall_tex = state_alias("render_resources", "wall_tex")
    torch_tex = state_alias("render_resources", "torch_tex")
    door_tex = state_alias("render_resources", "door_tex")
    window_tex = state_alias("render_resources", "window_tex")
    goblin_tex = state_alias("render_resources", "goblin_tex")

    building_specs = state_alias("build_state", "building_specs")
    environment_volumes = state_alias("build_state", "environment_volumes")
    buildings = state_alias("build_state", "buildings")
    roads = state_alias("build_state", "roads")
    building_roads = state_alias("build_state", "building_roads")
    doors = state_alias("build_state", "doors")
    windows = state_alias("build_state", "windows")
    walls = state_alias("build_state", "walls")
    torches = state_alias("build_state", "torches")
    goblins = state_alias("build_state", "goblins")
    chests = state_alias("build_state", "chests")
    showcase_chests = state_alias("build_state", "showcase_chests")
    showcase_polygons = state_alias("build_state", "showcase_polygons")
    inventory_items = state_alias("build_state", "inventory_items")
    building_road_routes = state_alias("build_state", "building_road_routes")
    building_road_segments = state_alias("build_state", "building_road_segments")
    builder = state_alias("build_state", "builder")

    _hud = state_alias("ui_state", "hud")
    battle_cards = state_alias("ui_state", "battle_cards")
    battle_overlay = state_alias("ui_state", "battle_overlay")
    battle_menu = state_alias("ui_state", "battle_menu")
    pause_menu = state_alias("ui_state", "pause_menu")
    setting_menu = state_alias("ui_state", "setting_menu")
    paused = state_alias("ui_state", "paused")
    inventory_open = state_alias("ui_state", "inventory_open")
    showing_settings_menu = state_alias("ui_state", "showing_settings_menu")
    battle_mode = state_alias("ui_state", "battle_mode")
    active_battle_goblin = state_alias("ui_state", "active_battle_goblin")
    inventory_selected_slot = state_alias("ui_state", "inventory_selected_slot")
    inventory_drag_source = state_alias("ui_state", "inventory_drag_source")
    inventory_notice_text = state_alias("ui_state", "inventory_notice_text")
    inventory_notice_expires_at = state_alias("ui_state", "inventory_notice_expires_at")
    hud_visible = state_alias("ui_state", "hud_visible")
    compass_visible = state_alias("ui_state", "compass_visible")
    minimap_visible = state_alias("ui_state", "minimap_visible")
    held_item_visible = state_alias("ui_state", "held_item_visible")
    test_light_visible = state_alias("ui_state", "test_light_visible")
    controls_text_visible = state_alias("ui_state", "controls_text_visible")
    debug_text_visible = state_alias("ui_state", "debug_text_visible")
    _last_mouse_pos = state_alias("ui_state", "last_mouse_pos")
    fov = state_alias("ui_state", "fov")
    fog_enabled = state_alias("ui_state", "fog_enabled")
    fog_density = state_alias("ui_state", "fog_density")
    clouds_enabled = state_alias("ui_state", "clouds_enabled")
    cloud_density = state_alias("ui_state", "cloud_density")
    cloud_speed = state_alias("ui_state", "cloud_speed")
    cloud_opacity = state_alias("ui_state", "cloud_opacity")
    vibrance = state_alias("ui_state", "vibrance")
    mouse_sensitivity = state_alias("ui_state", "mouse_sensitivity")
    walk_speed = state_alias("ui_state", "walk_speed")
    sprint_speed = state_alias("ui_state", "sprint_speed")
    road_speed_multiplier = state_alias("ui_state", "road_speed_multiplier")
    jump_speed = state_alias("ui_state", "jump_speed")
    gravity = state_alias("ui_state", "gravity")
    camera_follow_smooth_hz = state_alias("ui_state", "camera_follow_smooth_hz")

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
        world_random_seed: int | None = None,
        defer_setup: bool = False,
        lighting_backend: str = "packet",
    ) -> None:
        super().__init__()
        self.build_state = WorldBuildState()
        self.render_resources = WorldRenderResources()
        self.ui_state = WorldUIState()
        self._texture_lighting_sync_key = None
        self._texture_lighting_sync_result = False
        backend = str(lighting_backend).strip().lower()
        if backend not in ("legacy", "packet"):
            raise ValueError(
                "lighting_backend must be 'legacy' or 'packet'"
            )
        self.lighting_backend = backend
        self.collision_index = SceneCollisionIndex(self.render_resources)
        self.entity_registry = SceneEntityRegistry(
            self.render_resources,
            self.build_state,
            invalidate_collision_index=self.collision_index.invalidate,
        )
        self.combat = BattleController(self)
        self.ui_interactions = WorldUIInteractions(self)
        self.lighting_controller = StaticLightingController(
            self,
            resources=self.render_resources,
            build_state=self.build_state,
        )
        self.resource_disposer = SceneResourceDisposer(
            self.render_resources,
            self.ui_state,
            self.build_state,
        )

        self.renderer = WorldRenderer(
            self,
            resources=self.render_resources,
            ui_state=self.ui_state,
            lighting_controller=self.lighting_controller,
        )
        self._initialized = False
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

    def set_lighting_backend(self, backend: str) -> str:
        """Transition lighting backends through the compatibility boundary."""
        return self.lighting_controller.activate_backend(backend)

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

    def end_player_turn(self) -> bool:
        return self.combat.end_player_turn()

    def remove_battle_goblin(self) -> bool:
        return self.combat.remove_active_goblin()

    def end_battle(self) -> None:
        return self.combat.end()

    def refresh_immediate_entities(self) -> None:
        return self.entity_registry.refresh_immediate()

    def invalidate_texture_lighting_cache(self) -> None:
        return self.lighting_controller.invalidate_texture_lighting_cache()

    def draw_sky(self) -> None:  # pragma: no cover - visual
        return self.renderer.draw_sky()

    def draw(self, enable_timing: bool = False):  # pragma: no cover - visual
        return self.renderer.draw(enable_timing=enable_timing)

    def draw_world_objects(
        self, enable_timing: bool = False
    ):  # pragma: no cover - visual
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

    def _handle_inventory_release(self, pos):
        return self.ui_interactions.handle_inventory_release(pos)

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

    def refresh_static_lighting(self) -> None:
        return self.lighting_controller.refresh_static()

    def apply_static_exposure(self, brightness: float) -> None:
        return self.lighting_controller.apply_static_exposure(brightness)

    def dispose(self) -> None:
        try:
            self.renderer.dispose()
        finally:
            self.resource_disposer.dispose()
