"""World scene orchestration.

The heavy lifting is split into focused modules:
- world_setup.py: rendering/input bootstrap and asset loading
- world_builder.py: ground, buildings, roads, sprites, fences, and decals
- world_runtime.py: update/input/height-query helpers
- world_road_planner.py: building driveway route planning
"""

from __future__ import annotations

import time
from typing import Callable, Iterator, Optional

from pygame.math import Vector3

from config import *
from camera import Camera
from core.compat_shader import set_texture_lighting_state
from core.scene import Scene
from engine.rendering.decal import Decal
from engine.rendering.decal_batch import DecalBatch
from engine.rendering.sprite import WorldSprite, draw_sprites_batched
from world import world_builder, world_runtime, world_setup
from world.objects import WallTile
from world.objects.wall_tile import build_wall_tile_batches
from world.objects.polygon import Polygon
from world.world_renderer import WorldRenderer
from world.world_road_planner import create_building_access_roads


class WorldScene(Scene):
    def __init__(
        self,
        camera: Optional[Camera] = None,
        *,
        grid_count: int = 200,
        grid_tile_size: int = 25,
        grid_gap: int = 0,
        tree_count: int = 1000,
        grass_count: int = 750,
        rock_count: int = 750,
        defer_setup: bool = False,
    ) -> None:
        super().__init__()
        self.sprite_items: list[WorldSprite] = []
        self.decals: list[Decal] = []
        self.decal_batches: list[DecalBatch] = []
        self.wall_tiles: list[WallTile] = []
        self.wall_tile_batches: list[object] = []
        self.polygons: list[Polygon] = []
        self.others: list[object] = []

        self.renderer = WorldRenderer(self)
        self._initialized = False
        self._grid_count = grid_count
        self._grid_tile_size = grid_tile_size
        self._grid_gap = grid_gap
        self._tree_count = tree_count
        self._grass_count = grass_count
        self._rock_count = rock_count

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
        self, message: str, start_time: float, end_time: float, log: bool = False
    ):
        """Logs timing information for WorldScene setup phases."""
        if log:
            print(f"{message} took {end_time - start_time:.6f} seconds")

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

    def draw_sky(self) -> None:  # pragma: no cover - visual
        return self.renderer.draw_sky()

    def draw(self, enable_timing: bool = False):  # pragma: no cover - visual
        return self.renderer.draw(enable_timing=enable_timing)

    def draw_world_objects(self, enable_timing: bool = False):  # pragma: no cover - visual
        start_draw_decal_batches_time = time.perf_counter()
        for batch in self.decal_batches:
            batch.draw(camera=self.camera)
        end_draw_decal_batches_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing decal batches took "
                f"{end_draw_decal_batches_time - start_draw_decal_batches_time:.6f} seconds"
            )

        start_draw_wall_tiles_time = time.perf_counter()
        if self.wall_tile_batches:
            for batch in self.wall_tile_batches:
                batch.draw()
        else:
            for wall in self.wall_tiles:
                wall.draw(camera=self.camera)
        end_draw_wall_tiles_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing wall tiles took "
                f"{end_draw_wall_tiles_time - start_draw_wall_tiles_time:.6f} seconds"
            )

        start_draw_polygons_time = time.perf_counter()
        for polygon in self.polygons:
            polygon.draw(camera=self.camera)
        end_draw_polygons_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing polygons took "
                f"{end_draw_polygons_time - start_draw_polygons_time:.6f} seconds"
            )

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

            if hasattr(obj, "get_bounding_box"):
                try:
                    bbox = obj.get_bounding_box()
                    if bbox:
                        min_x, max_x, min_z, max_z = bbox
                        cam_x = (
                            float(getattr(self.camera.position, "x", 0.0))
                            if self.camera
                            else 0.0
                        )
                        cam_z = (
                            float(getattr(self.camera.position, "z", 0.0))
                            if self.camera
                            else 0.0
                        )
                        px = max(min_x, min(max_x, cam_x))
                        pz = max(min_z, min(max_z, cam_z))
                        cy = (
                            float(getattr(self.camera.position, "y", 0.0))
                            if self.camera
                            else 0.0
                        )
                        return (px, cy, pz)
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

        starting_draw_other_time = time.perf_counter()
        cam_pos = self.camera.position if self.camera is not None else None
        view_distance_sq = VIEWDISTANCE * VIEWDISTANCE
        for obj in self.others:
            if cam_pos is not None:
                pos = _approx_pos(obj)
                if pos is not None:
                    try:
                        dx = pos[0] - cam_pos.x
                        dy = pos[1] - cam_pos.y
                        dz = pos[2] - cam_pos.z
                        if (dx * dx + dy * dy + dz * dz) > view_distance_sq:
                            continue
                    except Exception:
                        pass

            obj.draw()
        end_draw_other_time = time.perf_counter()
        if enable_timing:
            print(
                "Drawing other objects took "
                f"{end_draw_other_time - starting_draw_other_time:.6f} seconds"
            )

        start_draw_sprites_time = time.perf_counter()
        if self.sprite_items and self.camera is not None:
            draw_sprites_batched(
                self.sprite_items,
                self.camera,
                self.ground_height_at,
                lighting=getattr(self, "lighting", None),
                sun_direction=getattr(self, "sun_direction", None),
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

    def ground_height_at(self, x: float, z: float) -> float:
        return world_runtime.ground_height_at(self, x, z)

    def view_space_position(
        self, *, dist: float, nx: float, ny: float, px: float = 0.0, py: float = 0.0
    ) -> Vector3:
        return world_runtime.view_space_position(
            self, dist=dist, nx=nx, ny=ny, px=px, py=py
        )

    def update(self, dt: float) -> None:
        return world_runtime.update(self, dt)

    def handle_event(self, event) -> None:
        return world_runtime.handle_event(self, event)

    def _compute_pause_buttons(self, width: int = WIDTH, height: int = HEIGHT):
        return self.renderer.compute_pause_buttons(width=width, height=height)

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
        if lighting is not None:
            lighting.set_base_brightness(brightness)
            self._sync_lighting_aliases()
        return set_texture_lighting_state(
            base_brightness=brightness,
            lighting=lighting,
            sun_direction=getattr(
                lighting,
                "sun_direction",
                getattr(self, "sun_direction", None),
            ),
            brightness_areas=getattr(
                lighting,
                "brightness_modifiers",
                getattr(camera, "brightness_areas", ()),
            ),
            exposure_scale=1.0,
            compile_shader=compile_shader,
        )

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
        ):
            dispose_once(getattr(self, attr_name, None))

        for attr_name in (
            "fence_meshes",
            "wall_tile_batches",
            "decal_batches",
            "decals",
            "roads",
            "building_roads",
            "others",
        ):
            for obj in getattr(self, attr_name, ()) or ():
                dispose_once(obj)

        self.ground_mesh = None
        self._ground_height_sampler = None
        self.road = None
        self.decal_batch = None
        self.fence_meshes = []
        self.wall_tile_batches = []
        self.decal_batches = []
        self.decals = []
        self.roads = []
        self.building_roads = []
        self.others = []
