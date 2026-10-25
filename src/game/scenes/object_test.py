"""Standalone walkable scene for inspecting the game's world objects."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pygame
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FOG,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_TEXTURE_2D,
    glBindTexture,
    glClear,
    glClearColor,
    glDisable,
    glEnable,
    glLoadIdentity,
    glMatrixMode,
    glRotatef,
    glTranslatef,
)
from OpenGL.GLU import gluPerspective
from pygame.math import Vector3

from engine.camera import Camera
from engine.physics.collision import movement_blocked_by_wall
from engine.config import HEIGHT, WIDTH
from engine.rendering.state import use_fixed_pipeline
from engine.rendering.mesh import BatchedMesh
from engine.core.scene import Scene
from engine.core.resources import ResourceScope
from engine.rendering.style import (
    get_render_fog_state,
    get_render_shine_state,
    get_render_vibrance_state,
    update_render_fog_state,
    update_render_shine_state,
    update_render_vibrance_state,
)
from engine.rendering.lighting.model import SceneLighting
from engine.rendering.lighting.renderer import SceneLightingRenderer
from engine.rendering.sprite import WorldSprite, draw_sprites_batched
from engine.textures.texture_utils import get_texture_size, load_texture
from game.resources.paths import GRASS_TEXTURE_PATH
from game.actors.goblin import Goblin, GoblinRenderBatch
from game.actors.skeleton import Skeleton, SkeletonRenderBatch
from game.resources.texture_manager import load_world_textures
from game.gameplay.rendering.receivers import (
    GROUND_LIGHTING_RECEIVER,
    PACKET_RUNTIME_LIGHTING_RECEIVERS,
    ROAD_LIGHTING_RECEIVER,
    SPRITE_LIGHTING_RECEIVER,
)
from game.objects.chest import Chest
from game.objects.architecture.door import Door
from engine.geometry.plane import GroundTile
from game.objects.road import Road
from game.objects.architecture.torch import Torch
from game.objects.architecture.window import Window
from game.objects.architecture.building import Building
from game.objects.architecture.door import build_door_render_batch
from game.objects.fence import build_textured_fence_ring
from engine.geometry.polygon import Polygon, build_polygon_render_batch
from game.objects.road import ROAD_SURFACE_CLEARANCE, road_crown_elevation
from game.objects.terrain.grass import GRASS_VARIANT_COUNT, Grass, build_grass_meshes
from game.objects.terrain.rock import ROCK_VARIANT_COUNT, Rock, build_rock_meshes
from game.objects.terrain.tree import TREE_VARIANT_COUNT, Tree, build_tree_meshes
from engine.geometry.wall import build_wall_tile_batches
from game.objects.architecture.window import build_window_render_batch


class ObjectTestScene(Scene):
    """A small first-person showroom that is independent of ``WorldScene``."""

    CLEAR_COLOR = (0.30, 0.40, 0.50, 1.0)
    FLOOR_CENTER_Z = -500.0
    FLOOR_HALF_WIDTH = 850.0
    FLOOR_HALF_DEPTH = 950.0
    MIN_X = -820.0
    MAX_X = 820.0
    MIN_Z = -1420.0
    MAX_Z = 420.0
    EYE_HEIGHT = 25.0
    PLAYER_RADIUS = 14.0
    WALK_SPEED = 180.0
    SPRINT_SPEED = 310.0
    MOUSE_SENSITIVITY = 0.0015
    VIEW_DISTANCE = 4_000.0

    def __init__(self) -> None:
        # Construction is deliberately CPU-only. LoadingScene advances the
        # GL-dependent initialize_steps() generator after a context exists.
        super().__init__()
        self.camera = Camera(
            position=Vector3(0.0, self.EYE_HEIGHT, 350.0),
            rotation=Vector3(0.0, 0.0, 0.0),
            width=WIDTH,
            height=HEIGHT,
            fov=75.0,
            default_brightness=1.0,
        )
        self.camera.update_rotation(0.0)
        self.next_scene = None
        self._initialized = False
        self._disposed = False
        self._resources = ResourceScope()
        self._previous_render_style = None
        self._placed_variant_counts = {
            "trees": 0,
            "grasses": 0,
            "rocks": 0,
            "fences": 0,
        }

        self.textures: dict[str, Any] = {}
        self.ground_tile: GroundTile | None = None
        self.floor_mesh: BatchedMesh | None = None
        self.roads: list[Road] = []
        self.buildings: list[Building] = []
        self.wall_tiles: list[Any] = []
        self.building_details: list[Any] = []
        self.wall_batches: list[BatchedMesh] = []
        self.fence_meshes: list[BatchedMesh] = []
        self.tree_meshes: list[BatchedMesh] = []
        self.grass_meshes: list[BatchedMesh] = []
        self.rock_meshes: list[BatchedMesh] = []
        self.doors: list[Door] = []
        self.windows: list[Window] = []
        self.torches: list[Torch] = []
        self.chests: list[Chest] = []
        self.polygons: list[Polygon] = []
        self.sprites: list[WorldSprite] = []
        self.goblin: Goblin | None = None
        self.goblin_batch: GoblinRenderBatch | None = None
        self.skeleton: Skeleton | None = None
        self.skeleton_batch: SkeletonRenderBatch | None = None
        self.door_batch = None
        self.window_batch = None
        self.polygon_batch = None
        self.object_manifest: dict[str, int] = {}
        self.sun_direction = Vector3(-0.35, -1.0, 0.20).normalize()
        center = Vector3(0.0, 0.0, self.FLOOR_CENTER_Z)
        self.lighting = SceneLighting(
            sun_position=center - self.sun_direction * 2200.0,
            sun_target=center,
            sky_color=self.CLEAR_COLOR,
        )
        self.lighting_renderer = self._resources.own(SceneLightingRenderer(
            receivers=PACKET_RUNTIME_LIGHTING_RECEIVERS,
            shadow_sources=self._shadow_sources,
            view_distance=self.VIEW_DISTANCE,
        ))
        self.lighting_renderer.shadows.sun_shadow_extent = 1400.0

    def initialize_steps(self) -> Iterator[tuple[str, float]]:
        """Create the scene incrementally for ``LoadingScene``."""

        if self._disposed:
            raise RuntimeError("Cannot initialize a disposed ObjectTestScene")
        if self._initialized:
            yield ("Object test scene ready", 1.0)
            return

        self._load_textures()
        self._configure_render_style()
        yield ("Loading object textures", 0.16)

        self._build_floor_and_road()
        yield ("Building the test floor", 0.32)

        self._place_nature_variants()
        yield ("Generating tree models and placing nature variants", 0.48)

        self._build_fence_displays()
        yield ("Building fence displays", 0.62)

        self._place_props_and_shapes()
        yield ("Placing interactive objects and shapes", 0.76)

        self._build_showcase_buildings()
        yield ("Constructing showcase buildings", 0.94)

        self._refresh_manifest()
        self._initialized = True
        yield ("Object test scene ready", 1.0)

    def _load_textures(self) -> None:
        textures = load_world_textures()
        required = (
            "ground_tex",
            "road_tex",
            "wall_tex",
            "roof_tex",
            "building_material_tex",
            "goblin_material_tex",
            "chest_tex",
            "door_tex",
            "window_tex",
            "torch_tex",
        )
        missing = [name for name in required if not textures.get(name)]

        expected_variants = {
            "fence_textures": 6,
        }
        for name, expected in expected_variants.items():
            values = tuple(value for value in textures.get(name, ()) if value)
            if len(values) < expected:
                missing.append(f"{name} ({len(values)}/{expected})")

        torch_frames = Torch.animation_frames(textures.get("torch_tex"))
        if len(torch_frames) < 5:
            missing.append(f"torch_tex ({len(torch_frames)}/5 frames)")

        if missing:
            raise RuntimeError(
                "Object test scene is missing required textures: "
                + ", ".join(missing)
            )
        self.textures = textures

    def _configure_render_style(self) -> None:
        if self._previous_render_style is None:
            self._previous_render_style = (
                get_render_fog_state(),
                get_render_vibrance_state(),
                get_render_shine_state(),
            )
        update_render_fog_state(
            enabled=False,
            density=0.0,
            color=self.CLEAR_COLOR,
        )
        update_render_vibrance_state(1.0)
        update_render_shine_state(enabled=True)

    def _own(self, value: Any) -> Any:
        if value is not None and callable(getattr(value, "dispose", None)):
            self._resources.own(value)
        return value

    @staticmethod
    def ground_height_at(x: float, z: float) -> float:
        return 0.0

    def _build_floor_and_road(self) -> None:
        self.ground_tile = GroundTile(
            position=Vector3(0.0, -5.0, self.FLOOR_CENTER_Z),
            width=self.FLOOR_HALF_WIDTH,
            height=5.0,
            depth=self.FLOOR_HALF_DEPTH,
        )
        vertices = self.ground_tile.get_world_vertices()
        u_repeat = 24.0
        v_repeat = 28.0
        uvs = {
            0: (0.0, 0.0),
            1: (u_repeat, 0.0),
            2: (u_repeat, v_repeat),
            3: (0.0, v_repeat),
        }
        rows = []
        for index in (0, 2, 1, 0, 3, 2):
            vertex = vertices[index]
            u, v = uvs[index]
            rows.append(
                (vertex.x, vertex.y, vertex.z, 0.82, 0.86, 0.78,
                 0.0, 1.0, 0.0, u, v)
            )

        ground_texture = load_texture(GRASS_TEXTURE_PATH, repeat=True)
        self.floor_mesh = self._own(
            BatchedMesh.from_vertex_data(
                np.asarray(rows, dtype=np.float32),
                texture=ground_texture,
                casts_shadows=False,
                casts_sun_shadows=False,
                casts_point_shadows=False,
                exposure_baseline=1.0,
                environment_lighting=False,
                shine_enabled=False,
                lighting_receiver=GROUND_LIGHTING_RECEIVER,
            )
        )

        road_width = 88.0
        road = Road(
            points=[(0.0, self.MAX_Z), (0.0, self.MIN_Z)],
            ground_y=0.0,
            width=road_width,
            texture=int(self.textures["road_tex"]),
            v_tiles=32.0,
            height_fn=self.ground_height_at,
            surface_offset=ROAD_SURFACE_CLEARANCE,
            elevation=road_crown_elevation(road_width),
            segment_length=20.0,
            brightness_modifiers=(),
            default_brightness=1.0,
            lighting=None,
            sun_direction=self.sun_direction,
            dynamic_lighting=True,
        )
        self.roads.append(self._own(road))

    def _place_nature_variants(self) -> None:
        tree_positions = (
            (-650.0, 150.0),
            (-440.0, 150.0),
            (-180.0, 150.0),
            (180.0, 150.0),
            (440.0, 150.0),
            (650.0, 150.0),
            (-650.0, -25.0),
        )
        tree_heights = (104.0, 112.0, 96.0, 108.0, 116.0, 100.0, 110.0)
        trees = [
            Tree.from_seed(
                Vector3(x, self.ground_height_at(x, z), z),
                seed=10_000 + variant,
                variant=variant,
                height=height,
            )
            for variant, ((x, z), height) in enumerate(
                zip(tree_positions, tree_heights)
            )
        ]
        if len(trees) != TREE_VARIANT_COUNT:
            raise RuntimeError(
                f"Expected {TREE_VARIANT_COUNT} procedural tree variants"
            )
        for mesh in build_tree_meshes(trees, chunk_size=180.0):
            self.tree_meshes.append(self._own(mesh))
        for _tree in trees:
            self._placed_variant_counts["trees"] += 1

        grass_positions = (
            (-510.0, 25.0),
            (-390.0, 25.0),
            (-270.0, 25.0),
            (-150.0, 25.0),
        )
        grasses = [
            Grass.from_seed(
                Vector3(x, self.ground_height_at(x, z), z),
                seed=20_000 + variant,
                variant=variant,
                height=22.0 + variant * 2.0,
            )
            for variant, (x, z) in enumerate(grass_positions)
        ]
        if len(grasses) != GRASS_VARIANT_COUNT:
            raise RuntimeError(
                f"Expected {GRASS_VARIANT_COUNT} procedural grass variants"
            )
        for mesh in build_grass_meshes(grasses, chunk_size=100.0):
            self.grass_meshes.append(self._own(mesh))
        self._placed_variant_counts["grasses"] += len(grasses)

        rock_positions = (
            (150.0, 25.0),
            (255.0, 25.0),
            (360.0, 25.0),
            (465.0, 25.0),
            (570.0, 25.0),
            (675.0, 25.0),
        )
        rocks = [
            Rock.from_seed(
                Vector3(x, self.ground_height_at(x, z), z),
                seed=30_000 + variant,
                variant=variant,
                height=26.0 + (variant % 3) * 5.0,
            )
            for variant, (x, z) in enumerate(rock_positions)
        ]
        if len(rocks) != ROCK_VARIANT_COUNT:
            raise RuntimeError(
                f"Expected {ROCK_VARIANT_COUNT} procedural rock variants"
            )
        for mesh in build_rock_meshes(rocks, chunk_size=100.0):
            self.rock_meshes.append(self._own(mesh))
        self._placed_variant_counts["rocks"] += len(rocks)

    def _build_fence_displays(self) -> None:
        centers = (-625.0, -375.0, -125.0, 125.0, 375.0, 625.0)
        for center_x, texture in zip(centers, self.textures["fence_textures"]):
            size = get_texture_size(texture) or (64, 64)
            px_to_world = 44.0 / max(1.0, float(size[1]))
            meshes = build_textured_fence_ring(
                center_x - 55.0,
                center_x + 55.0,
                -320.0,
                -225.0,
                ground_y=0.0,
                height_sampler=None,
                textures=[texture],
                px_to_world=px_to_world,
                color=(1.0, 1.0, 1.0),
                wave_amp=0.0,
                brightness_modifiers=(),
                default_brightness=1.0,
                lighting=None,
                sun_direction=self.sun_direction,
                dynamic_lighting=True,
            )
            for mesh in meshes:
                self.fence_meshes.append(self._own(mesh))
            if meshes:
                self._placed_variant_counts["fences"] += 1

    def _place_props_and_shapes(self) -> None:
        chest = Chest(
            Vector3(-280.0, 0.0, -505.0),
            texture=self.textures["chest_tex"],
            lighting=None,
            sun_direction=self.sun_direction,
            side="north",
        )
        self.chests.append(self._own(chest))

        self.goblin = Goblin(
            Vector3(280.0, 0.0, -505.0),
            material_texture=self.textures["goblin_material_tex"],
            camera=self.camera,
            ground_height_at=self.ground_height_at,
            position_blocked=None,
            player_in_building=lambda: False,
            move_speed=0.0,
            roam_radius=1.0,
            chase_radius=0.0,
            chase_give_up_radius=0.0,
            shadow_texture=None,
            sound_keys=(),
        )
        self.goblin.combat_enabled = False
        self.goblin_batch = self._own(
            GoblinRenderBatch(
                (self.goblin,),
                material_texture=self.textures["goblin_material_tex"],
            )
        )

        self.skeleton = self._own(Skeleton(
            Vector3(130.0, 0.0, -505.0),
            material_texture=self.textures["building_material_tex"],
            camera=self.camera,
            ground_height_at=self.ground_height_at,
            move_speed=0.0,
            roam_radius=1.0,
            chase_radius=0.0,
            chase_give_up_radius=0.0,
        ))
        self.skeleton.combat_enabled = False
        self.skeleton_batch = self._own(SkeletonRenderBatch((self.skeleton,)))

        polygon_specs = (
            (-640.0, [(-34.0, -28.0), (34.0, -28.0), (0.0, 34.0)]),
            (-430.0, [(-31.0, -31.0), (31.0, -31.0), (31.0, 31.0), (-31.0, 31.0)]),
            (-210.0, [(-35.0, -12.0), (0.0, -34.0), (35.0, -12.0), (22.0, 32.0), (-22.0, 32.0)]),
            (
                240.0,
                [
                    (-36.0, -26.0),
                    (4.0, -26.0),
                    (4.0, -42.0),
                    (38.0, 0.0),
                    (4.0, 42.0),
                    (4.0, 26.0),
                    (-36.0, 26.0),
                ],
            ),
            (
                500.0,
                [
                    (-36.0, -34.0),
                    (2.0, -34.0),
                    (2.0, 2.0),
                    (36.0, 2.0),
                    (36.0, 34.0),
                    (-36.0, 34.0),
                ],
            ),
        )
        polygon_textures = (
            self.textures["wall_tex"],
            self.textures["roof_tex"],
        )
        for index, (x, points) in enumerate(polygon_specs):
            self.polygons.append(
                Polygon(
                    position=Vector3(x, 55.0, -700.0),
                    points_2d=points,
                    thickness=6.0,
                    texture=polygon_textures[index % len(polygon_textures)],
                )
            )
        self.polygon_batch = self._own(build_polygon_render_batch(self.polygons))

    def _building_specs(self) -> tuple[dict[str, Any], ...]:
        return (
            {
                "position": Vector3(-320.0, 0.0, -1080.0),
                "width": 300.0,
                "depth": 250.0,
                "height": 84.0,
                "base_y": 0.0,
                "wall_thickness": 3.5,
                "doorway_side": "north",
                "doorway_width": Door.DEFAULT_WIDTH,
                "doorway_height": Door.DEFAULT_HEIGHT,
                "windows": [
                    {
                        "side": "north",
                        "offset": -88.0,
                        "width": Window.DEFAULT_WIDTH,
                        "height": Window.DEFAULT_HEIGHT,
                        "sill_height": Window.DEFAULT_SILL_HEIGHT,
                    },
                    {
                        "side": "north",
                        "offset": 88.0,
                        "width": Window.DEFAULT_WIDTH,
                        "height": Window.DEFAULT_HEIGHT,
                        "sill_height": Window.DEFAULT_SILL_HEIGHT,
                    },
                    {
                        "side": "east",
                        "offset": 0.0,
                        "width": Window.DEFAULT_WIDTH,
                        "height": Window.DEFAULT_HEIGHT,
                        "sill_height": Window.DEFAULT_SILL_HEIGHT,
                    },
                    {
                        "side": "west",
                        "offset": 0.0,
                        "width": Window.DEFAULT_WIDTH,
                        "height": Window.DEFAULT_HEIGHT,
                        "sill_height": Window.DEFAULT_SILL_HEIGHT,
                    },
                ],
                "torches": [
                    {"side": "north", "offset": -43.0},
                    {"side": "north", "offset": 43.0},
                ],
                "detail_variant": 0,
                "chimney": True,
            },
            {
                "position": Vector3(340.0, 0.0, -1100.0),
                "width": 250.0,
                "depth": 320.0,
                "height": 92.0,
                "base_y": 0.0,
                "wall_thickness": 3.5,
                "doorway_side": "north",
                "doorway_width": Door.DEFAULT_WIDTH,
                "doorway_height": Door.DEFAULT_HEIGHT,
                "windows": [
                    {
                        "side": "north",
                        "offset": -68.0,
                        "width": Window.DEFAULT_WIDTH,
                        "height": Window.DEFAULT_HEIGHT,
                        "sill_height": Window.DEFAULT_SILL_HEIGHT,
                    },
                    {
                        "side": "north",
                        "offset": 68.0,
                        "width": Window.DEFAULT_WIDTH,
                        "height": Window.DEFAULT_HEIGHT,
                        "sill_height": Window.DEFAULT_SILL_HEIGHT,
                    },
                    {
                        "side": "east",
                        "offset": -72.0,
                        "width": Window.DEFAULT_WIDTH,
                        "height": Window.DEFAULT_HEIGHT,
                        "sill_height": Window.DEFAULT_SILL_HEIGHT,
                    },
                    {
                        "side": "east",
                        "offset": 72.0,
                        "width": Window.DEFAULT_WIDTH,
                        "height": Window.DEFAULT_HEIGHT,
                        "sill_height": Window.DEFAULT_SILL_HEIGHT,
                    },
                    {
                        "side": "west",
                        "offset": 0.0,
                        "width": Window.DEFAULT_WIDTH,
                        "height": Window.DEFAULT_HEIGHT,
                        "sill_height": Window.DEFAULT_SILL_HEIGHT,
                    },
                ],
                "torches": [
                    {"side": "north", "offset": -36.0},
                    {"side": "north", "offset": 36.0},
                ],
                "detail_variant": 1,
                "chimney": False,
            },
        )

    def _build_showcase_buildings(self) -> None:
        for building_index, spec in enumerate(self._building_specs()):
            building = self._own(Building(position=spec["position"]))
            self.buildings.append(building)
            palette = building.style_palette(spec["detail_variant"])
            material_tex = self.textures["building_material_tex"]
            shell = building.create_perimeter_walls(
                wall_height=spec["height"],
                wall_thickness=spec["wall_thickness"],
                texture=material_tex,
                roof_texture=material_tex,
                uv_repeat=(2.0, 1.0),
                base_y=spec["base_y"],
                width=spec["width"],
                depth=spec["depth"],
                max_tile_width=95.0,
                doorway=True,
                doorway_side=spec["doorway_side"],
                doorway_width=spec["doorway_width"],
                doorway_height=spec["doorway_height"],
                windows=spec["windows"],
                roof=True,
                roof_thickness=4.0,
                roof_overhang=0.0,
                terrain_height_at=self.ground_height_at,
                terrain_embed_depth=0.0,
                wall_color=palette.wall,
                roof_color=palette.roof,
            )
            details = building.create_architectural_details(
                wall_height=spec["height"],
                wall_thickness=spec["wall_thickness"],
                width=spec["width"],
                depth=spec["depth"],
                wall_texture=material_tex,
                roof_texture=material_tex,
                doorway_side=spec["doorway_side"],
                doorway_width=spec["doorway_width"],
                doorway_height=spec["doorway_height"],
                windows=spec["windows"],
                base_y=spec["base_y"],
                roof_thickness=4.0,
                roof_overhang=7.0,
                detail_variant=spec["detail_variant"],
                chimney=spec["chimney"],
            )
            self.wall_tiles.extend(shell)
            self.building_details.extend(details)

            door = Door.from_building_spec(
                spec,
                texture=self.textures["door_tex"],
                camera=self.camera,
                ground_height_at=self.ground_height_at,
                lighting=None,
                sun_direction=self.sun_direction,
            )
            self.doors.append(self._own(door))

            for window_spec in spec["windows"]:
                window = Window.from_building_spec(
                    spec,
                    window_spec=window_spec,
                    texture=self.textures["window_tex"],
                    backing_texture=self.textures["wall_tex"],
                    camera=self.camera,
                    ground_height_at=self.ground_height_at,
                    lighting=None,
                    sun_direction=self.sun_direction,
                )
                self.windows.append(self._own(window))

            point_lights = Torch.point_lights_for_building_spec(
                spec,
                building_index=building_index,
            )
            self.lighting.extend_point_lights(point_lights)
            building_torches = Torch.build_for_point_lights(
                point_lights,
                texture=self.textures["torch_tex"],
                camera=self.camera,
            )
            self.torches.extend(building_torches)
            self.sprites.extend(building_torches)

        self.wall_batches = build_wall_tile_batches(
            [*self.wall_tiles, *self.building_details],
            camera=self.camera,
            default_brightness=1.0,
            sun_direction=self.sun_direction,
            lighting=None,
            dynamic_lighting=True,
        )
        for batch in self.wall_batches:
            self._own(batch)
        self.door_batch = self._own(build_door_render_batch(self.doors))
        self.window_batch = self._own(build_window_render_batch(self.windows))

    def _refresh_manifest(self) -> None:
        self.object_manifest = {
            "ground_tiles": 1 if self.ground_tile is not None else 0,
            "roads": len(self.roads),
            "tree_variants": self._placed_variant_counts["trees"],
            "grass_variants": self._placed_variant_counts["grasses"],
            "rock_variants": self._placed_variant_counts["rocks"],
            "fence_variants": self._placed_variant_counts["fences"],
            "torch_frames": len(Torch.animation_frames(self.textures["torch_tex"])),
            "goblin_model_vertices": (
                self.goblin.model_vertex_count if self.goblin is not None else 0
            ),
            "buildings": len(self.buildings),
            "wall_tiles": len(self.wall_tiles),
            "building_details": len(self.building_details),
            "doors": len(self.doors),
            "windows": len(self.windows),
            "torches": len(self.torches),
            "chests": len(self.chests),
            "polygons": len(self.polygons),
            "goblins": 1 if self.goblin is not None else 0,
            "skeletons": 1 if self.skeleton is not None else 0,
            "skeleton_model_vertices": (
                self.skeleton.model_vertex_count if self.skeleton is not None else 0
            ),
        }

    def apply_mouse_delta(
        self,
        dx: float,
        dy: float,
        dt: float | None = None,
    ) -> None:
        if not self._initialized:
            return
        self.camera.rotation.y -= float(dx) * self.MOUSE_SENSITIVITY
        pitch = self.camera.rotation.x - float(dy) * self.MOUSE_SENSITIVITY
        pitch_limit = math.pi * 0.5 - 0.001
        self.camera.rotation.x = max(-pitch_limit, min(pitch_limit, pitch))
        self.camera.update_rotation(float(dt or 0.0))

    def update(self, dt: float) -> None:
        if not self._initialized or self._disposed:
            return
        frame_dt = max(0.0, min(0.1, float(dt)))
        self._update_walking(frame_dt)

        for door in self.doors:
            door.update(frame_dt)
        for chest in self.chests:
            chest.update(frame_dt)
        for torch in self.torches:
            torch.update(frame_dt)
        if self.goblin is not None:
            self.goblin.update(frame_dt)
        if self.skeleton is not None:
            self.skeleton.update(frame_dt)

    def _update_walking(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        sprinting = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        speed = self.SPRINT_SPEED if sprinting else self.WALK_SPEED
        old_position = self.camera.position.copy()
        self.camera.move_camera(keys, speed, dt)
        candidate = self.camera.position.copy()
        candidate.x = max(self.MIN_X, min(self.MAX_X, candidate.x))
        candidate.z = max(self.MIN_Z, min(self.MAX_Z, candidate.z))
        candidate.y = self.EYE_HEIGHT
        self.camera.position = old_position

        if not self._movement_blocked(old_position, candidate):
            self.camera.position = candidate
            return

        # Preserve wall sliding by testing each horizontal axis separately.
        current = old_position.copy()
        x_candidate = Vector3(candidate.x, self.EYE_HEIGHT, current.z)
        if not self._movement_blocked(current, x_candidate):
            current = x_candidate
        z_candidate = Vector3(current.x, self.EYE_HEIGHT, candidate.z)
        if not self._movement_blocked(current, z_candidate):
            current = z_candidate
        self.camera.position = current

    def _movement_blocked(self, old_position: Vector3, new_position: Vector3) -> bool:
        return (
            movement_blocked_by_wall(
                self._active_collision_meshes(),
                old_position,
                new_position,
                player_radius=self.PLAYER_RADIUS,
                player_bottom_y=0.5,
                player_top_y=44.0,
            )
            is not None
        )

    def _active_collision_meshes(self) -> list[Any]:
        meshes = [*self.wall_tiles, *self.polygons]
        for entity in (*self.doors, *self.windows, *self.chests):
            if not getattr(entity, "collision_enabled", True):
                continue
            get_meshes = getattr(entity, "get_collision_meshes", None)
            if callable(get_meshes):
                meshes.extend(get_meshes() or ())
        return meshes

    def handle_event(self, event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_l:
            self.lighting_renderer.toggle()
            return
        if event.key == pygame.K_ESCAPE:
            if self.next_scene is not None:
                return
            from game.scenes.main_menu import MainMenuScene

            self.next_scene = MainMenuScene()
        elif event.key == pygame.K_e:
            target = self._focused_interactable()
            if target is not None:
                target.interact(actor=None, scene=self)

    def _focused_interactable(self) -> Any | None:
        camera_forward = Vector3(
            self.camera._forward.x,
            0.0,
            self.camera._forward.z,
        )
        if camera_forward.length_squared() <= 1e-8:
            return None
        camera_forward.normalize_ip()

        best_target = None
        best_score = float("inf")
        for target in (*self.doors, *self.chests):
            get_position = getattr(target, "get_interaction_position", None)
            position = get_position() if callable(get_position) else target.position
            delta = Vector3(
                position.x - self.camera.position.x,
                0.0,
                position.z - self.camera.position.z,
            )
            distance = delta.length()
            max_distance = float(getattr(target, "interaction_distance", 80.0))
            if distance <= 1e-6 or distance > max_distance:
                continue
            facing = camera_forward.dot(delta / distance)
            if facing < 0.30:
                continue
            score = distance + (1.0 - facing) * 24.0
            if score < best_score:
                best_score = score
                best_target = target
        return best_target

    def _shadow_sources(self):
        """Supply the same authored objects to the shared shadow renderer."""
        yield from self.tree_meshes
        yield from self.grass_meshes
        yield from self.rock_meshes
        yield from self.fence_meshes
        yield from self.wall_batches
        yield from self.roads
        yield from self.chests
        yield self.door_batch
        yield self.window_batch
        yield self.polygon_batch

    def render(
        self,
        *,
        show_hud: bool = True,
        text=None,
        fps: float | None = None,
    ) -> None:  # pragma: no cover - visual
        self.lighting_renderer.begin_frame(
            camera=self.camera,
            lighting=self.lighting,
            profiler=getattr(self, "profiler", None),
        )
        batch_lighting = self.lighting_renderer.batch_draw_args()
        use_fixed_pipeline()
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glDisable(GL_FOG)
        update_render_fog_state(
            enabled=False,
            density=0.0,
            color=self.CLEAR_COLOR,
        )
        glClearColor(*self.CLEAR_COLOR)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.camera.fov,
            self.camera.viewport_width / max(1.0, self.camera.viewport_height),
            1.0,
            self.VIEW_DISTANCE,
        )
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(math.degrees(-self.camera.rotation.x), 1.0, 0.0, 0.0)
        glRotatef(math.degrees(-self.camera.rotation.y), 0.0, 1.0, 0.0)
        glTranslatef(
            -self.camera.position.x,
            -self.camera.position.y,
            -self.camera.position.z,
        )

        if self.floor_mesh is not None:
            self.floor_mesh.draw(
                camera=self.camera,
                view_distance=self.VIEW_DISTANCE,
                **self.lighting_renderer.draw_args(GROUND_LIGHTING_RECEIVER),
            )
        for road in self.roads:
            road.draw(
                camera=self.camera,
                view_distance=self.VIEW_DISTANCE,
                **self.lighting_renderer.draw_args(ROAD_LIGHTING_RECEIVER),
            )
        BatchedMesh.draw_many(
            self.wall_batches,
            camera=self.camera,
            view_distance=self.VIEW_DISTANCE,
            **batch_lighting,
        )
        if self.polygon_batch is not None:
            self.polygon_batch.draw(
                camera=self.camera,
                view_distance=self.VIEW_DISTANCE,
                **batch_lighting,
            )
        BatchedMesh.draw_many(
            self.fence_meshes,
            camera=self.camera,
            view_distance=self.VIEW_DISTANCE,
            **batch_lighting,
        )
        BatchedMesh.draw_many(
            (*self.tree_meshes, *self.grass_meshes, *self.rock_meshes),
            camera=self.camera,
            view_distance=self.VIEW_DISTANCE,
            **batch_lighting,
        )
        if self.door_batch is not None:
            self.door_batch.draw(
                camera=self.camera,
                view_distance=self.VIEW_DISTANCE,
                **batch_lighting,
            )
        if self.window_batch is not None:
            self.window_batch.draw(
                camera=self.camera,
                view_distance=self.VIEW_DISTANCE,
                **batch_lighting,
            )
        for chest in self.chests:
            chest.draw(
                camera=self.camera,
                **self.lighting_renderer.draw_args(chest.packet_lighting_receiver),
            )
        for actor_batch in (self.goblin_batch, self.skeleton_batch):
            if actor_batch is None:
                continue
            actor_batch.draw(
                camera=self.camera,
                view_distance=self.VIEW_DISTANCE,
                **self.lighting_renderer.draw_args(
                    actor_batch.packet_lighting_receiver
                ),
            )
        draw_sprites_batched(
            self.sprites,
            self.camera,
            self.ground_height_at,
            lighting=None,
            sun_direction=self.sun_direction,
            profiler=getattr(self, "profiler", None),
            static_data=False,
            **self.lighting_renderer.draw_args(SPRITE_LIGHTING_RECEIVER),
        )

        if show_hud and text is not None:
            self._draw_hud(text, fps)

    def _draw_hud(self, text, fps: float | None) -> None:
        section = self._current_section()
        target = self._focused_interactable()
        text.begin()
        try:
            text.draw_text(
                "OBJECT TEST SCENE",
                24,
                24,
                color=(245, 248, 250, 255),
                key="object_test_title",
            )
            text.draw_text(
                "WASD move  |  Mouse look  |  Shift sprint  |  E interact  |  L lighting  |  Esc menu",
                24,
                52,
                color=(218, 229, 235, 255),
                key="object_test_controls",
            )
            text.draw_text(
                f"Exhibit: {section}  |  Lighting: {'ON' if self.lighting_renderer.enabled else 'OFF (unlit)'}",
                24,
                80,
                color=(255, 222, 154, 255),
                key="object_test_section",
            )
            if fps is not None:
                text.draw_text(
                    f"{fps:.0f} FPS",
                    WIDTH - 24,
                    24,
                    color=(210, 222, 228, 255),
                    key="object_test_fps",
                    align="right",
                )
            if target is not None:
                action = "Close" if getattr(target, "target_open", False) else "Open"
                label = type(target).__name__
                text.draw_text(
                    f"E  {action} {label}",
                    WIDTH * 0.5,
                    HEIGHT * 0.70,
                    color=(255, 239, 178, 255),
                    key="object_test_interaction",
                    align="center",
                )
            text.draw_text(
                "+",
                WIDTH * 0.5,
                HEIGHT * 0.5,
                color=(255, 255, 255, 220),
                key="object_test_crosshair",
                align="center",
            )
        finally:
            text.end()

    def _current_section(self) -> str:
        z = float(self.camera.position.z)
        if z > 90.0:
            return "Nature variants"
        if z > -390.0:
            return "Fence variants"
        if z > -600.0:
            return "Chest, goblin, and skeleton"
        if z > -830.0:
            return "Extruded polygons"
        return "Buildings, doors, windows, and torches"

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        try:
            try:
                if self.goblin is not None:
                    self.goblin.kill()
            finally:
                self._resources.dispose()
        finally:
            self._clear_references()

    def _clear_references(self) -> None:
        self.roads.clear()
        self.buildings.clear()
        self.wall_tiles.clear()
        self.building_details.clear()
        self.wall_batches.clear()
        self.fence_meshes.clear()
        self.tree_meshes.clear()
        self.grass_meshes.clear()
        self.rock_meshes.clear()
        self.doors.clear()
        self.windows.clear()
        self.torches.clear()
        self.chests.clear()
        self.polygons.clear()
        self.sprites.clear()
        self.textures.clear()
        self.ground_tile = None
        self.floor_mesh = None
        self.goblin = None
        self.goblin_batch = None
        self.skeleton = None
        self.skeleton_batch = None
        self.door_batch = None
        self.window_batch = None
        self.polygon_batch = None
        self._restore_render_style()

    def _restore_render_style(self) -> None:
        previous = self._previous_render_style
        self._previous_render_style = None
        if previous is None:
            return
        fog, vibrance, shine = previous
        update_render_fog_state(
            enabled=fog.enabled, density=fog.density, color=fog.color
        )
        update_render_vibrance_state(vibrance.vibrance)
        update_render_shine_state(
            enabled=shine.enabled,
            strength=shine.strength,
            power=shine.power,
            fresnel=shine.fresnel,
            tint=shine.tint,
        )


def create_object_test_scene() -> ObjectTestScene:
    """Return a CPU-only scene ready to be advanced by ``LoadingScene``."""

    return ObjectTestScene()


__all__ = ["ObjectTestScene", "create_object_test_scene"]
