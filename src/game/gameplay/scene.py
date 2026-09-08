"""Shared gameplay scene lifecycle.

Levels provide layout and construction. This base supplies shared controls,
combat, inventory, rendering, collision and resource ownership.
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
    SKELETON_COUNT,
    STARTING_POS,
    WIDTH,
)
from engine.camera import Camera
from engine.core.scene import Scene
from engine.core.entity import Entity
from game.gameplay import runtime as world_runtime, setup as world_setup
from game.combat.controller import BattleController
from engine.physics.spatial import SceneCollisionIndex
from game.gameplay.entities import GameplayEntityRegistry
from game.gameplay.rendering.lighting import StaticLightingController
from game.gameplay.resources import GameplayResourceDisposer
from game.ui.interactions import WorldUIInteractions
from game.gameplay.rendering.renderer import GameplayRenderer
from game.gameplay.state import (
    GameplayState,
    GameplayResources,
    GameplayUIState,
    state_alias,
)


class PlayableScene(Scene):
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
    tree_meshes = state_alias("render_resources", "tree_meshes")
    grass_meshes = state_alias("render_resources", "grass_meshes")
    rock_meshes = state_alias("render_resources", "rock_meshes")
    goblin_batch = state_alias("render_resources", "goblin_batch")
    skeleton_batch = state_alias("render_resources", "skeleton_batch")
    ground_mesh = state_alias("render_resources", "ground_mesh")
    island_boundary = state_alias("render_resources", "island_boundary")
    sky = state_alias("render_resources", "sky")
    road = state_alias("render_resources", "road")
    decal_batch = state_alias("render_resources", "decal_batch")
    _ground_height_sampler = state_alias("render_resources", "ground_height_sampler")
    _sprite_update_cache = state_alias("render_resources", "sprite_update_cache")
    ground_tex = state_alias("render_resources", "ground_tex")
    road_tex = state_alias("render_resources", "road_tex")
    fence_textures = state_alias("render_resources", "fence_textures")
    item_textures = state_alias("render_resources", "item_textures")
    equipment_slot_textures = state_alias(
        "render_resources", "equipment_slot_textures"
    )
    wall_tex = state_alias("render_resources", "wall_tex")
    roof_tex = state_alias("render_resources", "roof_tex")
    building_material_tex = state_alias("render_resources", "building_material_tex")
    chest_tex = state_alias("render_resources", "chest_tex")
    torch_tex = state_alias("render_resources", "torch_tex")
    door_tex = state_alias("render_resources", "door_tex")
    window_tex = state_alias("render_resources", "window_tex")
    goblin_material_tex = state_alias("render_resources", "goblin_material_tex")

    building_specs = state_alias("build_state", "building_specs")
    environment_volumes = state_alias("build_state", "environment_volumes")
    buildings = state_alias("build_state", "buildings")
    trees = state_alias("build_state", "trees")
    grasses = state_alias("build_state", "grasses")
    rocks = state_alias("build_state", "rocks")
    roads = state_alias("build_state", "roads")
    building_roads = state_alias("build_state", "building_roads")
    doors = state_alias("build_state", "doors")
    windows = state_alias("build_state", "windows")
    walls = state_alias("build_state", "walls")
    building_details = state_alias("build_state", "building_details")
    torches = state_alias("build_state", "torches")
    creatures = state_alias("build_state", "creatures")
    goblins = state_alias("build_state", "goblins")
    skeletons = state_alias("build_state", "skeletons")
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
    active_battle_creature = state_alias("ui_state", "active_battle_creature")
    # Compatibility alias for integrations written before combat was generalized.
    active_battle_goblin = state_alias("ui_state", "active_battle_creature")
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

    def __init__(self, camera: Optional[Camera] = None, *, lighting_backend="packet", world_random_seed=None):
        super().__init__()
        self.build_state = GameplayState()
        self.render_resources = GameplayResources()
        self.ui_state = GameplayUIState()
        self._texture_lighting_sync_key = None
        self._texture_lighting_sync_result = False
        backend = str(lighting_backend).strip().lower()
        if backend not in ("legacy", "packet"):
            raise ValueError(
                "lighting_backend must be 'legacy' or 'packet'"
            )
        self.lighting_backend = backend
        self.collision_index = SceneCollisionIndex(
            lambda: self.render_resources.wall_tiles,
            lambda: self.render_resources.polygons,
            cell_size=self.render_resources.collision_cell_size,
            is_dynamic=lambda mesh: (
                getattr(mesh,"door_render_batched",False) or hasattr(mesh,"open_amount")
            ),
        )
        self.entity_registry = GameplayEntityRegistry(
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
        self.resource_disposer = GameplayResourceDisposer(
            self.render_resources,
            self.ui_state,
            self.build_state,
        )

        self.renderer = GameplayRenderer(
            self,
            resources=self.render_resources,
            ui_state=self.ui_state,
            lighting_controller=self.lighting_controller,
        )
        self._initialized = False
        self.camera = camera or Camera(
            position=Vector3(STARTING_POS),
            width=WIDTH,
            height=HEIGHT,
            fov=FOV,
            default_brightness=0.8,
        )
        set_projection = getattr(self.camera, "set_projection", None)
        if callable(set_projection):
            if camera is not None:
                self.fov = float(getattr(self.camera, "fov", self.fov))
            set_projection(fov=self.fov, width=WIDTH, height=HEIGHT)

        self.world_random_seed = world_random_seed

    def set_lighting_backend(self, backend: str) -> str:
        """Transition lighting backends through the compatibility boundary."""
        return self.lighting_controller.activate_backend(backend)


    def _setup_controllers(self) -> None:
        return world_setup.setup_controllers(self)


    def _setup_graphics(self) -> None:
        return world_setup.setup_graphics(self)


    def _load_assets(self) -> None:
        return world_setup.load_assets(self)


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


    def start_battle(self, creature: Entity) -> bool:
        return self.combat.start(creature)


    def player_attack_damage_preview(self) -> int:
        return self.combat.player_attack_damage_preview()


    def roll_player_attack_damage(self) -> tuple[int, bool]:
        return self.combat.roll_player_attack_damage()


    def damage_battle_creature(self, amount: int | None = None) -> int:
        return self.combat.damage_active_creature(amount)


    def damage_battle_goblin(self, amount: int | None = None) -> int:
        """Compatibility wrapper; use damage_battle_creature()."""
        return self.damage_battle_creature(amount)


    def end_player_turn(self) -> bool:
        return self.combat.end_player_turn()


    def remove_battle_creature(self) -> bool:
        return self.combat.remove_active_creature()


    def remove_battle_goblin(self) -> bool:
        """Compatibility wrapper; use remove_battle_creature()."""
        return self.remove_battle_creature()


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


    def _handle_battle_escape(self):
        return self.ui_interactions.handle_battle_escape()


    def _handle_inventory_click(self, pos):
        return self.ui_interactions.handle_inventory_click(pos)


    def _handle_inventory_release(self, pos):
        return self.ui_interactions.handle_inventory_release(pos)


    def _handle_inventory_escape(self):
        return self.ui_interactions.handle_inventory_escape()


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
            try:
                self.resource_disposer.dispose()
            finally:
                self.collision_index.invalidate()


    def rebuild_legacy_road_batches(self):
        """Level hook for rebuilding baked road geometry after a lighting edit."""

    def rebuild_legacy_fences(self):
        """Level hook for rebuilding baked fence geometry after a lighting edit."""
