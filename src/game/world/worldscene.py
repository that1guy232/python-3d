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

from game.config import *
from engine.camera import Camera
from engine.core.compat_shader import set_texture_lighting_state
from engine.core.mesh import BatchedMesh
from engine.core.scene import Scene
from engine.entity import Entity
from engine.rendering.decal import Decal
from engine.rendering.decal_batch import DecalBatch
from engine.rendering.sprite import WorldSprite, draw_sprites_batched
from game.world import world_builder, world_runtime, world_setup
from game.world.objects import Chest, WallTile
from game.world.objects.goblin import draw_goblin_shadows_batched
from game.world.objects.wall_tile import build_wall_tile_batches
from game.world.objects.polygon import Polygon
from game.world.player_stats import PlayerStats
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
        self.battle_mode = False
        self.active_battle_goblin = None
        self.battle_cards = None
        self.player_stats = PlayerStats()
        self.last_player_attack: dict[str, int | bool] | None = None

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
        total_steps = len(setup_steps) + world_builder.CREATE_WORLD_OBJECT_STEPS
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
            log = bool(globals().get("PERFORMANCE_SETUP_TIMING", False))
        if log:
            print(f"[setup] {message} took {end_time - start_time:.6f} seconds")

    def _profile(self, name: str):
        profiler = getattr(self, "profiler", None)
        if profiler is None or not getattr(profiler, "enabled", False):
            return nullcontext()
        return profiler.section(name)

    def add_entity(self, entity: Entity) -> Entity:
        """Register a runtime entity and its scene-facing resources."""
        if entity not in self.entities:
            self.entities.append(entity)
        if (
            entity not in self.immediate_entities
            and not getattr(entity, "door_render_batched", False)
            and not getattr(entity, "render_batched", False)
            and not getattr(entity, "shadow_render_batched", False)
        ):
            self.immediate_entities.append(entity)

        get_sprites = getattr(entity, "get_sprites", None)
        if callable(get_sprites):
            for sprite in get_sprites() or ():
                if sprite not in self.sprite_items:
                    self.sprite_items.append(sprite)

        get_collision_meshes = getattr(entity, "get_collision_meshes", None)
        if callable(get_collision_meshes):
            for mesh in get_collision_meshes() or ():
                if mesh not in self.wall_tiles:
                    self.wall_tiles.append(mesh)

        return entity

    def remove_entity(self, entity: Entity) -> None:
        """Unregister a runtime entity and its scene-facing resources."""
        if entity is None:
            return

        kill = getattr(entity, "kill", None)
        if callable(kill):
            kill()
        else:
            setattr(entity, "enabled", False)
            setattr(entity, "visible", False)

        def without_item(values):
            return [value for value in (values or []) if value is not entity]

        for attr_name in (
            "entities",
            "immediate_entities",
            "goblins",
            "chests",
            "showcase_chests",
        ):
            if hasattr(self, attr_name):
                setattr(self, attr_name, without_item(getattr(self, attr_name)))

        get_sprites = getattr(entity, "get_sprites", None)
        if callable(get_sprites):
            sprites = tuple(get_sprites() or ())
            if sprites:
                sprite_ids = {id(sprite) for sprite in sprites}
                self.sprite_items = [
                    sprite
                    for sprite in self.sprite_items
                    if id(sprite) not in sprite_ids
                ]

        get_collision_meshes = getattr(entity, "get_collision_meshes", None)
        if callable(get_collision_meshes):
            meshes = tuple(get_collision_meshes() or ())
            if meshes:
                mesh_ids = {id(mesh) for mesh in meshes}
                self.wall_tiles = [
                    mesh for mesh in self.wall_tiles if id(mesh) not in mesh_ids
                ]
                self.invalidate_collision_index()

        self._sprite_update_cache = None

    def start_battle(self, goblin: Entity) -> bool:
        if goblin is None or not getattr(goblin, "enabled", True):
            return False

        max_hp = max(1, int(getattr(goblin, "max_hp", 5)))
        setattr(goblin, "max_hp", max_hp)
        if not hasattr(goblin, "hp"):
            setattr(goblin, "hp", max_hp)

        self.battle_mode = True
        self.active_battle_goblin = goblin
        self.paused = False
        self.inventory_open = False
        self.showing_settings_menu = False

        controller = getattr(self, "_camera_controller", None)
        look_at = getattr(controller, "look_at", None)
        if callable(look_at):
            position = getattr(goblin, "position", None)
            if position is not None:
                look_at(position)

        import pygame

        self.mouse_visible = True
        self.mouse_grabbed = False
        try:
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
        except pygame.error:
            pass
        return True

    def player_attack_damage_preview(self) -> int:
        stats = getattr(self, "player_stats", None)
        if stats is None:
            return 1
        preview = getattr(stats, "base_attack_damage", None)
        if callable(preview):
            return max(0, int(preview()))
        strength = max(0, int(getattr(stats, "strength", 1)))
        elemental = max(0, int(getattr(stats, "elemental_damage", 0)))
        return strength + elemental

    def roll_player_attack_damage(self) -> tuple[int, bool]:
        stats = getattr(self, "player_stats", None)
        if stats is None:
            self.last_player_attack = {"damage": 1, "critical": False}
            return 1, False

        roll_damage = getattr(stats, "roll_attack_damage", None)
        if callable(roll_damage):
            rng = getattr(self, "rng", None)
            try:
                damage, critical = roll_damage(rng)
            except TypeError:
                damage, critical = roll_damage()
        else:
            damage = self.player_attack_damage_preview()
            critical = False

        damage = max(0, int(damage))
        critical = bool(critical)
        self.last_player_attack = {"damage": damage, "critical": critical}
        return damage, critical

    def damage_battle_goblin(self, amount: int | None = None) -> int:
        goblin = getattr(self, "active_battle_goblin", None)
        if goblin is None or not getattr(goblin, "enabled", True):
            return 0

        if amount is None:
            amount, _critical = self.roll_player_attack_damage()

        take_damage = getattr(goblin, "take_damage", None)
        if callable(take_damage):
            hp = int(take_damage(amount))
        else:
            max_hp = max(1, int(getattr(goblin, "max_hp", 5)))
            hp = max(0, int(getattr(goblin, "hp", max_hp)) - max(0, int(amount)))
            setattr(goblin, "max_hp", max_hp)
            setattr(goblin, "hp", hp)

        if hp <= 0:
            self.remove_battle_goblin()
        return hp

    def remove_battle_goblin(self) -> bool:
        goblin = getattr(self, "active_battle_goblin", None)
        if goblin is None:
            self.end_battle()
            return False

        max_hp = max(1, int(getattr(goblin, "max_hp", 5)))
        hp = int(getattr(goblin, "hp", max_hp))
        if hp > 0:
            return False

        self.remove_entity(goblin)
        self.end_battle()
        return True

    def end_battle(self) -> None:
        self.active_battle_goblin = None
        self.battle_mode = False

        controller = getattr(self, "_camera_controller", None)
        sync_target = getattr(controller, "sync_rotation_target_to_camera", None)
        if callable(sync_target):
            sync_target()

        import pygame

        self.mouse_visible = False
        self.mouse_grabbed = True
        try:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        except pygame.error:
            pass

    def refresh_immediate_entities(self) -> None:
        self.immediate_entities = [
            entity
            for entity in self.entities
            if not getattr(entity, "door_render_batched", False)
            and not getattr(entity, "render_batched", False)
            and not getattr(entity, "shadow_render_batched", False)
        ]

    def _sync_lighting_aliases(self):
        """Keep older scene attributes pointing at the shared lighting model."""
        lighting = getattr(self, "lighting", None)
        if lighting is None:
            return None
        self.sun_pos = lighting.sun_position
        self.sun_direction = lighting.sun_direction
        self.brightness_modifiers = lighting.brightness_modifiers
        self.covered_regions = lighting.covered_regions
        return lighting

    def invalidate_texture_lighting_cache(self) -> None:
        self._texture_lighting_sync_key = None

    def draw_sky(self) -> None:  # pragma: no cover - visual
        return self.renderer.draw_sky()

    def draw(self, enable_timing: bool = False):  # pragma: no cover - visual
        return self.renderer.draw(enable_timing=enable_timing)

    def draw_world_objects(self, enable_timing: bool = False):  # pragma: no cover - visual
        def _approx_pos(obj):
            position = getattr(obj, "position", None)
            if position is not None:
                try:
                    return (
                        float(position.x),
                        float(getattr(position, "y", 0.0)),
                        float(position.z),
                    )
                except Exception:
                    pass

            center = getattr(obj, "center", None)
            if center is not None:
                try:
                    return (
                        float(center.x),
                        float(getattr(center, "y", 0.0)),
                        float(center.z),
                    )
                except Exception:
                    pass

            if hasattr(obj, "start") and hasattr(obj, "end"):
                try:
                    start = obj.start
                    end = obj.end
                    cx = (float(start.x) + float(end.x)) * 0.5
                    cz = (float(start.z) + float(end.z)) * 0.5
                    cy = float(
                        getattr(
                            obj,
                            "ground_y",
                            getattr(self.camera.position, "y", 0.0)
                            if self.camera
                            else 0.0,
                        )
                    )
                    return (cx, cy, cz)
                except Exception:
                    pass

            return None

        def _sphere_for_vertices(vertices):
            if not vertices:
                return None
            min_x = min(float(v.x) for v in vertices)
            max_x = max(float(v.x) for v in vertices)
            min_y = min(float(v.y) for v in vertices)
            max_y = max(float(v.y) for v in vertices)
            min_z = min(float(v.z) for v in vertices)
            max_z = max(float(v.z) for v in vertices)
            center = (
                (min_x + max_x) * 0.5,
                (min_y + max_y) * 0.5,
                (min_z + max_z) * 0.5,
            )
            radius = (
                ((max_x - min_x) * 0.5) ** 2
                + ((max_y - min_y) * 0.5) ** 2
                + ((max_z - min_z) * 0.5) ** 2
            ) ** 0.5
            return center, radius

        def _object_render_sphere(obj):
            center = getattr(obj, "bounds_center", None)
            if center is not None:
                return center, float(getattr(obj, "bounds_radius", 0.0))

            for method_name in ("get_render_bounding_sphere", "get_bounding_sphere"):
                method = getattr(obj, method_name, None)
                if callable(method):
                    try:
                        sphere = method()
                    except Exception:
                        sphere = None
                    if sphere:
                        return sphere

            visual_vertices = getattr(obj, "_visual_vertices", None)
            if callable(visual_vertices):
                try:
                    sphere = _sphere_for_vertices(visual_vertices())
                except Exception:
                    sphere = None
                if sphere:
                    return sphere

            get_vertices = getattr(obj, "get_world_vertices", None)
            if callable(get_vertices):
                try:
                    sphere = _sphere_for_vertices(get_vertices())
                except Exception:
                    sphere = None
                if sphere:
                    return sphere

            get_bounds = getattr(obj, "get_bounding_box", None)
            if callable(get_bounds):
                try:
                    bbox = get_bounds()
                    if bbox:
                        min_x, max_x, min_z, max_z = (float(v) for v in bbox)
                        pos = _approx_pos(obj)
                        cy = pos[1] if pos is not None else (
                            float(getattr(self.camera.position, "y", 0.0))
                            if self.camera
                            else 0.0
                        )
                        center = (
                            (min_x + max_x) * 0.5,
                            cy,
                            (min_z + max_z) * 0.5,
                        )
                        radius = (
                            ((max_x - min_x) * 0.5) ** 2
                            + ((max_z - min_z) * 0.5) ** 2
                        ) ** 0.5
                        return center, radius
                except Exception:
                    pass

            pos = _approx_pos(obj)
            if pos is None:
                return None
            radius = max(
                0.0,
                float(
                    getattr(
                        obj,
                        "render_radius",
                        getattr(obj, "collision_radius", 0.0),
                    )
                    or 0.0
                ),
            )
            return pos, radius

        def _object_visible(obj) -> bool:
            camera = self.camera
            if camera is None:
                return True
            tester = getattr(camera, "sphere_in_frustum", None)
            if not callable(tester):
                return True
            sphere = _object_render_sphere(obj)
            if sphere is None:
                return True
            center, radius = sphere
            return bool(tester(center, radius, far_distance=VIEWDISTANCE))

        start_draw_decal_batches_time = time.perf_counter()
        with self._profile("objects.decal_batches"):
            profiler = getattr(self, "profiler", None)
            for batch in self.decal_batches:
                batch.draw(camera=self.camera, profiler=profiler)
        end_draw_decal_batches_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing decal batches took "
                f"{end_draw_decal_batches_time - start_draw_decal_batches_time:.6f} seconds"
            )

        start_draw_wall_tiles_time = time.perf_counter()
        with self._profile("objects.wall_tiles"):
            if self.wall_tile_batches:
                BatchedMesh.draw_many(
                    self.wall_tile_batches,
                    camera=self.camera,
                    view_distance=VIEWDISTANCE,
                )
            else:
                entity_ids = {id(entity) for entity in self.entities}
                for wall in self.wall_tiles:
                    if id(wall) in entity_ids:
                        continue
                    if not _object_visible(wall):
                        continue
                    wall.draw(camera=self.camera)
        end_draw_wall_tiles_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing wall tiles took "
                f"{end_draw_wall_tiles_time - start_draw_wall_tiles_time:.6f} seconds"
            )

        start_draw_polygons_time = time.perf_counter()
        with self._profile("objects.polygon_batches"):
            for batch in getattr(self, "polygon_batches", ()) or ():
                batch.draw(camera=self.camera, view_distance=VIEWDISTANCE)
        with self._profile("objects.polygons"):
            for polygon in self.polygons:
                if getattr(polygon, "render_batched", False):
                    continue
                if not _object_visible(polygon):
                    continue
                polygon.draw(camera=self.camera)
        end_draw_polygons_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing polygons took "
                f"{end_draw_polygons_time - start_draw_polygons_time:.6f} seconds"
            )

        starting_draw_other_time = time.perf_counter()
        with self._profile("objects.road_batches"):
            for batch in getattr(self, "road_batches", ()) or ():
                batch.draw(camera=self.camera, view_distance=VIEWDISTANCE)

        with self._profile("objects.others"):
            for obj in self.others:
                if getattr(obj, "render_batched", False):
                    continue
                if not _object_visible(obj):
                    continue

                obj.draw()
        end_draw_other_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing other objects took "
                f"{end_draw_other_time - starting_draw_other_time:.6f} seconds"
            )

        start_draw_entities_time = time.perf_counter()

        with self._profile("objects.door_batches"):
            for batch in getattr(self, "door_batches", ()) or ():
                batch.draw(camera=self.camera, view_distance=VIEWDISTANCE)

        with self._profile("objects.window_batches"):
            for batch in getattr(self, "window_batches", ()) or ():
                batch.draw(camera=self.camera, view_distance=VIEWDISTANCE)

        with self._profile("objects.goblin_shadows"):
            goblins = getattr(self, "goblins", None) or self.entities
            draw_goblin_shadows_batched(
                goblins,
                camera=self.camera,
                view_distance=VIEWDISTANCE,
            )

        with self._profile("objects.entities"):
            for entity in getattr(self, "immediate_entities", ()) or ():
                if not getattr(entity, "enabled", True) or not getattr(
                    entity,
                    "visible",
                    True,
                ):
                    continue

                if not _object_visible(entity):
                    continue

                draw_entity = getattr(entity, "draw", None)
                if callable(draw_entity):
                    with self._profile(f"entities.{type(entity).__name__}"):
                        try:
                            draw_entity(camera=self.camera)
                        except TypeError:
                            draw_entity()
        end_draw_entities_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing entities took "
                f"{end_draw_entities_time - start_draw_entities_time:.6f} seconds"
            )

        start_draw_sprites_time = time.perf_counter()
        with self._profile("objects.sprites"):
            if self.sprite_items and self.camera is not None:
                draw_sprites_batched(
                    self.sprite_items,
                    self.camera,
                    self.ground_height_at,
                    lighting=getattr(self, "lighting", None),
                    sun_direction=getattr(self, "sun_direction", None),
                    profiler=getattr(self, "profiler", None),
                    static_data=True,
                )

        end_draw_sprites_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing sprites took "
                f"{end_draw_sprites_time - start_draw_sprites_time:.6f} seconds"
            )

    def contains_horizontal(self, pos: Vector3) -> bool:
        return world_runtime.contains_horizontal(self, pos)

    def is_on_road(self, x: float, z: float, *, margin: float = 0.0) -> bool:
        return world_runtime.is_on_road(self, x, z, margin=margin)

    def invalidate_collision_index(self) -> None:
        return world_runtime.invalidate_collision_index(self)

    def rebuild_collision_index(self) -> dict:
        return world_runtime.rebuild_collision_index(self)

    def collision_meshes_for_bounds(
        self,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        *,
        include_polygons: bool = True,
    ) -> list:
        return world_runtime.collision_meshes_for_bounds(
            self,
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
        return world_runtime.collision_meshes_at(
            self,
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
        return self.renderer.compute_pause_buttons(width=width, height=height)

    def _compute_battle_buttons(self, width: int = WIDTH, height: int = HEIGHT):
        return self.renderer.compute_battle_buttons(width=width, height=height)

    def _handle_battle_click(self, pos):
        return self.renderer.handle_battle_click(pos)

    def _handle_battle_motion(self, pos):
        return self.renderer.handle_battle_motion(pos)

    def _handle_battle_release(self, pos):
        return self.renderer.handle_battle_release(pos)

    def _handle_inventory_click(self, pos):
        return self.renderer.handle_inventory_click(pos)

    def _handle_pause_click(self, pos):
        return self.renderer.handle_pause_click(pos)

    def _handle_pause_motion(self, pos):
        return self.renderer.handle_pause_motion(pos)

    def _handle_pause_release(self, pos):
        return self.renderer.handle_pause_release(pos)

    def _draw_pause_menu(self, text):
        return self.renderer.draw_pause_menu(text)

    def render(
        self, *, show_hud: bool = True, text=None, fps: float | None = None
    ):  # pragma: no cover - visual
        return self.renderer.render(show_hud=show_hud, text=text, fps=fps)

    def apply_mouse_delta(self, dx: float, dy: float, dt: float | None = None) -> None:
        return world_runtime.apply_mouse_delta(self, dx, dy, dt)

    def set_brightness(self, value: float) -> float:
        """Set global brightness and refresh all baked lighting consumers."""
        camera = self.camera
        brightness = float(value)
        setter = getattr(camera, "set_brightness_default", None)
        if callable(setter):
            brightness = float(setter(brightness))
        else:
            camera.brightness_default = brightness
            cache = getattr(camera, "_brightness_cache", None)
            if cache is not None:
                cache.clear()

        lighting = getattr(self, "lighting", None)
        if lighting is not None:
            lighting.set_base_brightness(brightness)

        if (
            getattr(self, "_initialized", False)
            and getattr(self, "_last_static_lighting_brightness", None) == brightness
        ):
            return brightness

        if getattr(self, "_initialized", False):
            self._sync_brightness_modifiers_from_camera()
            if self.brightness_modifiers and self._sync_lighting_uniforms():
                self._apply_untextured_static_exposure_cpu(brightness)
                self._last_static_lighting_brightness = brightness
            elif self.brightness_modifiers:
                self.refresh_static_lighting()
            else:
                self.apply_static_exposure(brightness)
                self._last_static_lighting_brightness = brightness
        return brightness

    def _sync_brightness_modifiers_from_camera(self) -> None:
        camera = getattr(self, "camera", None)
        areas = getattr(camera, "brightness_areas", None)
        if areas is None:
            return

        modifiers = []
        for area in areas:
            try:
                modifiers.append(
                    {
                        "center": area["center"],
                        "radius": float(area["radius"]),
                        "value": float(area["value"]),
                        "falloff": float(area.get("falloff", 1.0)),
                        "bounds": area.get("bounds"),
                        "indoor_only": bool(area.get("indoor_only", False)),
                        "floor_scale": float(area.get("floor_scale", 1.0)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        lighting = getattr(self, "lighting", None)
        if lighting is not None:
            lighting.set_brightness_modifiers(modifiers)
            self._sync_lighting_aliases()
        else:
            self.brightness_modifiers = modifiers
        self.invalidate_texture_lighting_cache()

    @staticmethod
    def _dispose_renderable(obj) -> None:
        dispose = getattr(obj, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass

    def refresh_static_lighting(self) -> None:
        """Rebuild static VBOs whose vertex colors contain brightness."""
        if not getattr(self, "_initialized", False):
            return

        self._sync_brightness_modifiers_from_camera()
        camera = self.camera
        brightness = float(getattr(camera, "brightness_default", 1.0))
        lighting = getattr(self, "lighting", None)
        if lighting is not None:
            lighting.set_base_brightness(brightness)
            lighting.set_covered_regions(getattr(self, "covered_regions", ()))
            self._sync_lighting_aliases()
        sun_direction = getattr(
            lighting,
            "sun_direction",
            getattr(self, "sun_direction", None),
        )
        self.sun_direction = sun_direction

        builder = getattr(self, "builder", None)
        if builder is not None:
            builder.brightness_modifiers = self.brightness_modifiers
            builder.default_brightness = brightness
            builder.lighting = lighting
            builder.sun_direction = sun_direction
            builder.covered_regions = getattr(self, "covered_regions", ())
            self._dispose_renderable(getattr(self, "ground_mesh", None))
            self.ground_mesh = builder.build()
            self._ground_height_sampler = getattr(self.ground_mesh, "height_sampler", None)

        height_sampler = getattr(self, "_ground_height_sampler", None)
        refreshed_roads: set[int] = set()
        for road in self._road_lighting_candidates():
            if road is None:
                continue
            if id(road) in refreshed_roads:
                continue
            refreshed_roads.add(id(road))
            refresh = getattr(road, "refresh_lighting", None)
            if callable(refresh):
                refresh(
                    brightness_modifiers=self.brightness_modifiers,
                    default_brightness=brightness,
                    lighting=lighting,
                    sun_direction=sun_direction,
                    height_sampler=height_sampler,
                )
        world_builder._build_road_batches(self)

        for wall in getattr(self, "walls", ()) or ():
            wall.sun_direction = sun_direction
            wall.lighting = lighting

        self._dispose_renderable_batches(getattr(self, "wall_tile_batches", ()))
        self.wall_tile_batches = build_wall_tile_batches(
            getattr(self, "walls", []),
            camera=camera,
            default_brightness=brightness,
            sun_direction=sun_direction,
            lighting=lighting,
        )

        if getattr(self, "ground_mesh", None) is not None:
            world_builder._build_fences(self)

        self._sync_lighting_uniforms(compile_shader=False)
        self._last_static_lighting_brightness = brightness

    def apply_static_exposure(self, brightness: float) -> None:
        """Apply global exposure without rebuilding static meshes."""
        exposure = float(brightness)
        if self._sync_lighting_uniforms(base_brightness=exposure):
            self._apply_untextured_static_exposure_cpu(exposure)
            return

        self._apply_static_exposure_cpu(exposure)

    def _sync_lighting_uniforms(
        self,
        *,
        base_brightness: float | None = None,
        compile_shader: bool = True,
    ) -> bool:
        camera = getattr(self, "camera", None)
        lighting = getattr(self, "lighting", None)
        brightness = (
            float(base_brightness)
            if base_brightness is not None
            else float(getattr(camera, "brightness_default", 1.0))
        )
        brightness_areas = getattr(
            lighting,
            "brightness_modifiers",
            getattr(camera, "brightness_areas", ()),
        )
        covered_regions = getattr(
            lighting,
            "covered_regions",
            getattr(self, "covered_regions", ()),
        )
        sun_direction = getattr(
            lighting,
            "sun_direction",
            getattr(self, "sun_direction", None),
        )
        sync_key = self._texture_lighting_fast_key(
            brightness=brightness,
            lighting=lighting,
            sun_direction=sun_direction,
            brightness_areas=brightness_areas,
            covered_regions=covered_regions,
            compile_shader=compile_shader,
        )
        if sync_key == self._texture_lighting_sync_key:
            return self._texture_lighting_sync_result

        if lighting is not None:
            lighting.set_base_brightness(brightness)
            self._sync_lighting_aliases()
        result = set_texture_lighting_state(
            base_brightness=brightness,
            lighting=lighting,
            sun_direction=sun_direction,
            brightness_areas=brightness_areas,
            covered_regions=covered_regions,
            exposure_scale=1.0,
            compile_shader=compile_shader,
        )
        self._texture_lighting_sync_key = sync_key
        self._texture_lighting_sync_result = result
        return result

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
        return (
            bool(compile_shader),
            self._rounded(brightness),
            self._vector_key(sun_direction),
            self._rounded(getattr(lighting, "ambient", 0.72)),
            self._rounded(getattr(lighting, "diffuse", 0.48)),
            self._rounded(getattr(lighting, "max_factor", 1.15)),
            self._collection_identity_key(brightness_areas),
            self._collection_identity_key(covered_regions),
            tuple(
                self._rounded(getattr(door, "open_amount", 0.0), digits=4)
                for door in getattr(self, "doors", ()) or ()
                if getattr(door, "_doorway_light_region", None) is not None
                or getattr(door, "_doorway_brightness_modifier", None) is not None
            ),
        )

    @staticmethod
    def _collection_identity_key(values):
        try:
            return (id(values), len(values))
        except Exception:
            return (id(values), None)

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
        return (
            bool(compile_shader),
            cls._rounded(brightness),
            cls._vector_key(sun_direction),
            cls._rounded(getattr(lighting, "ambient", 0.72)),
            cls._rounded(getattr(lighting, "diffuse", 0.48)),
            cls._rounded(getattr(lighting, "max_factor", 1.15)),
            cls._brightness_areas_key(brightness_areas),
            cls._covered_regions_key(covered_regions),
        )

    @staticmethod
    def _rounded(value, digits: int = 5):
        try:
            return round(float(value), digits)
        except Exception:
            return None

    @classmethod
    def _vector_key(cls, value):
        try:
            return (
                cls._rounded(value.x),
                cls._rounded(value.y),
                cls._rounded(value.z),
            )
        except Exception:
            try:
                return (
                    cls._rounded(value[0]),
                    cls._rounded(value[1]),
                    cls._rounded(value[2]),
                )
            except Exception:
                return None

    @classmethod
    def _brightness_areas_key(cls, areas):
        values = []
        for area in areas or ():
            try:
                if isinstance(area, dict):
                    center = area.get("center")
                    bounds = area.get("bounds")
                    values.append(
                        (
                            cls._vector_key(center),
                            cls._rounded(area.get("radius")),
                            cls._rounded(area.get("value")),
                            cls._rounded(area.get("falloff", 1.0)),
                            cls._bounds_key(bounds),
                            bool(area.get("indoor_only", False)),
                            cls._rounded(area.get("floor_scale", 1.0)),
                        )
                    )
                else:
                    center, radius, value, falloff = area[:4]
                    values.append(
                        (
                            cls._vector_key(center),
                            cls._rounded(radius),
                            cls._rounded(value),
                            cls._rounded(falloff),
                            cls._bounds_key(area[4] if len(area) > 4 else None),
                            False,
                            1.0,
                        )
                    )
            except Exception:
                continue
        return tuple(values)

    @classmethod
    def _covered_regions_key(cls, regions):
        values = []
        for region in regions or ():
            if not isinstance(region, dict):
                try:
                    values.append(tuple(cls._rounded(part) for part in region[:5]))
                except Exception:
                    continue
                continue

            openings = region.get("openings")
            if not isinstance(openings, (list, tuple)):
                openings = [
                    value
                    for value in (region.get("doorway"), *(region.get("windows") or ()))
                    if isinstance(value, dict)
                ]

            values.append(
                (
                    cls._rounded(region.get("min_x")),
                    cls._rounded(region.get("max_x")),
                    cls._rounded(region.get("min_z")),
                    cls._rounded(region.get("max_z")),
                    cls._rounded(region.get("factor", 1.0)),
                    tuple(cls._opening_key(opening) for opening in openings),
                )
            )
        return tuple(values)

    @classmethod
    def _opening_key(cls, opening):
        if not isinstance(opening, dict):
            return None
        return (
            str(opening.get("side", "")),
            cls._rounded(opening.get("center_x")),
            cls._rounded(opening.get("center_z")),
            cls._rounded(opening.get("width")),
            cls._rounded(opening.get("depth")),
            cls._rounded(opening.get("side_fade")),
            cls._rounded(opening.get("edge_factor", 1.0)),
        )

    @classmethod
    def _bounds_key(cls, bounds):
        if bounds is None:
            return None
        try:
            return tuple(cls._rounded(part) for part in bounds)
        except Exception:
            return None

    @staticmethod
    def _uses_texture_shader(obj) -> bool:
        return getattr(obj, "texture", None) is not None

    @staticmethod
    def _set_exposure_cpu(obj, exposure: float) -> None:
        setter = getattr(obj, "set_exposure", None)
        if callable(setter):
            setter(exposure)

    def _apply_untextured_static_exposure_cpu(self, exposure: float) -> None:
        mesh = getattr(self, "ground_mesh", None)
        if mesh is not None and not self._uses_texture_shader(mesh):
            self._set_exposure_cpu(mesh, exposure)

        for mesh in getattr(self, "fence_meshes", ()) or ():
            if not self._uses_texture_shader(mesh):
                self._set_exposure_cpu(mesh, exposure)

        for batch in getattr(self, "road_batches", ()) or ():
            if not self._uses_texture_shader(batch):
                self._set_exposure_cpu(batch, exposure)

        for mesh in getattr(self, "wall_tile_batches", ()) or ():
            if not self._uses_texture_shader(mesh):
                self._set_exposure_cpu(mesh, exposure)

        refreshed_roads: set[int] = set()
        for road in self._road_lighting_candidates():
            if road is None or id(road) in refreshed_roads:
                continue
            refreshed_roads.add(id(road))
            if not self._uses_texture_shader(road):
                self._set_exposure_cpu(road, exposure)

    def _apply_static_exposure_cpu(self, exposure: float) -> None:
        mesh = getattr(self, "ground_mesh", None)
        if mesh is not None:
            self._set_exposure_cpu(mesh, exposure)

        for mesh in getattr(self, "fence_meshes", ()) or ():
            self._set_exposure_cpu(mesh, exposure)

        for batch in getattr(self, "road_batches", ()) or ():
            self._set_exposure_cpu(batch, exposure)

        for mesh in getattr(self, "wall_tile_batches", ()) or ():
            self._set_exposure_cpu(mesh, exposure)

        refreshed_roads: set[int] = set()
        for road in self._road_lighting_candidates():
            if road is None or id(road) in refreshed_roads:
                continue
            refreshed_roads.add(id(road))
            self._set_exposure_cpu(road, exposure)

    def _road_lighting_candidates(self):
        return [
            getattr(self, "road", None),
            *(getattr(self, "roads", ()) or ()),
            *(
                obj
                for obj in (getattr(self, "others", ()) or ())
                if hasattr(obj, "refresh_lighting") or hasattr(obj, "set_exposure")
            ),
        ]

    @staticmethod
    def _dispose_renderable_batches(values) -> None:
        for value in values or ():
            WorldScene._dispose_renderable(value)

    def dispose(self) -> None:
        """Release scene-owned VBOs before the OpenGL context is destroyed."""
        world_runtime.stop_ambient_birds()
        disposed: set[int] = set()

        def dispose_once(obj) -> None:
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in disposed:
                return
            disposed.add(obj_id)

            dispose = getattr(obj, "dispose", None)
            if callable(dispose):
                try:
                    dispose()
                except Exception:
                    pass

        for attr_name in (
            "ground_mesh",
            "road",
            "decal_batch",
            "_hud",
        ):
            dispose_once(getattr(self, attr_name, None))

        for attr_name in (
            "fence_meshes",
            "wall_tile_batches",
            "road_batches",
            "decal_batches",
            "decals",
            "roads",
            "building_roads",
            "door_batches",
            "window_batches",
            "polygon_batches",
            "showcase_chests",
            "chests",
            "others",
            "entities",
        ):
            for obj in getattr(self, attr_name, ()) or ():
                dispose_once(obj)

        self.ground_mesh = None
        self._ground_height_sampler = None
        self.road = None
        self.decal_batch = None
        self.fence_meshes = []
        self.wall_tile_batches = []
        self.road_batches = []
        self.decal_batches = []
        self.decals = []
        self.roads = []
        self.building_roads = []
        self.door_batches = []
        self.window_batches = []
        self.polygon_batches = []
        self.showcase_chests = []
        self.chests = []
        self._collision_spatial_index = None
        self.others = []
        self.entities = []
        self.immediate_entities = []
