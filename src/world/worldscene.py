"""World scene that owns the camera, input, movement, and full rendering.

This moves camera and head-bob logic from the engine into the scene so the
engine can host arbitrary scenes (e.g., a main menu) without carrying 3D
controls. The scene exposes its own render() and handle_event() hooks.
"""

from __future__ import annotations

from config import *

import math
import random
from typing import Tuple, Optional

from pygame.math import Vector3

from core.scene import Scene

import time

from world.objects.polygon import Polygon
from world.sprite import WorldSprite

from world.objects.fence import build_textured_fence_ring
from world.world_spawner import spawn_world_sprites

from world.objects.ground import TexturedGroundGridBuilder
from world.objects import Road
from world.objects.building import Building
from world.world_hud import WorldHUD

from world.decal import Decal
from world.decal_batch import DecalBatch

from textures.texture_utils import (
    load_texture,
    create_shadow_texture,
    create_polygon_shadow_texture,
    create_test_texture,
    get_texture_size,
)
from textures.texture_manager import load_world_textures
from textures.resoucepath import *

from sound.sound_utils import Sounds
from render.sky_renderer import SkyRenderer

from camera import Camera
from camera.headbob import HeadBob
from camera.sway_controller import SwayController
from camera.cameracontroller import CameraController

from OpenGL.GL import (
    glEnable,
    glFogf,
    glFogi,
    glHint,
    glClear,
    glMatrixMode,
    glLoadIdentity,
    glRotatef,
    glTranslatef,
    GL_FOG,
    GL_FOG_MODE,
    GL_FOG_COLOR,
    GL_FOG_DENSITY,
    GL_FOG_HINT,
    GL_EXP2,
    GL_FASTEST,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_PROJECTION,
    GL_MODELVIEW,
    glFogfv,
    glClearColor,
    
)
from OpenGL.GLU import gluPerspective

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
    ) -> None:
                #init super
        super().__init__()
        spacing = grid_tile_size + grid_gap
        half = grid_tile_size / 2.0
        self.world_center = Vector3(
            (grid_count * spacing) / 2, 0, (grid_count * spacing) / 2
        )

        print("World Scene Initialized")

        # If no camera provided (default), create one suitable for this scene
        cam = camera or Camera(
            position=Vector3(STARTING_POS), width=WIDTH, height=HEIGHT, fov=FOV, default_brightness=0.8
        )

        self.camera = cam
        start_time = time.perf_counter()
        self._setup_brightness_areas(grid_count, spacing, half)
        self.log_timing("Setting up brightness areas", start_time, time.perf_counter())

        start_time = time.perf_counter()
        self._setup_controllers()
        self.log_timing("Setting up controllers", start_time, time.perf_counter())
        self.ground_bounds = (
            0 + half,
            grid_count * spacing - half,
            0 + half,
            grid_count * spacing - half,
        )
        start_time = time.perf_counter()
        self._setup_graphics()
        self.log_timing("Setting up graphics", start_time, time.perf_counter())
        start_time = time.perf_counter()
        self._load_assets()
        self.log_timing("Loading assets", start_time, time.perf_counter())
        start_time = time.perf_counter()
        self._create_world_objects(grid_count, spacing, half, grid_tile_size, grid_gap, tree_count, grass_count, rock_count)
        self.log_timing("Creating world objects", start_time, time.perf_counter())


        print("World scene initialization complete.")



    def _setup_brightness_areas(self, grid_count: int, spacing: float, half: float) -> None:
        brightness_modifiers = []
        min_x = 0 + half
        max_x = grid_count * spacing - half
        min_z = 0 + half
        max_z = grid_count * spacing - half
        brightness_modifiers.append((
            Vector3(self.world_center.x, 0, self.world_center.z),
            1000,
            1,
            4,
        ))
        for i in range(0):
            cx = random.triangular(min_x, max_x, (min_x + max_x) * 0.5) + 1e-6
            cz = random.triangular(min_z, max_z, (min_z + max_z) * 0.5) + 1e-6
            radius = random.uniform(150.0, 250.0)
            brightness_modifiers.append((
                Vector3(cx, 0, cz),
                radius,
                random.uniform(0.5, 0.8),
                4,
            ))

        for bm in brightness_modifiers:
            try:
                self.camera.add_brightness_area(*bm)
            except Exception:
                pass
        self.brightness_modifiers = brightness_modifiers

    def _setup_controllers(self) -> None:
        def _headbob_on_footstep(intensity, sprinting, phase, foot):
            try:
                if hasattr(self, "on_footstep"):
                    try:
                        self.on_footstep(
                            intensity=intensity,
                            sprinting=sprinting,
                            phase=phase,
                            foot=foot,
                        )
                        return
                    except Exception:
                        pass

                base = 0.25 if sprinting else 0.18
                vol = max(0.05, min(1.0, base + 0.5 * intensity))

                try:
                    if getattr(self, "road", None) and self.road.contains_point(
                        self.camera.position.x, self.camera.position.z
                    ):
                        Sounds.play("step", volume=min(1.0, vol * 0.9))
                    else:
                        Sounds.play("footstep", volume=min(1.0, vol))
                except Exception:
                    pass
            except Exception:
                pass

        self._headbob = HeadBob(
            enabled=HEADBOB_ENABLED,
            frequency=HEADBOB_FREQUENCY,
            amplitude_y=HEADBOB_AMPLITUDE,
            amplitude_x=HEADBOB_AMPLITUDE_SIDE,
            sprint_mult=HEADBOB_SPRINT_MULT,
            damping=HEADBOB_DAMPING,
            on_footstep=_headbob_on_footstep,
        )

        self._sway_controller = SwayController(
            max_x=1.25,
            max_y=0.75,
            mouse_scale=0.01,
            responsiveness=12.0,
            return_rate=8.0,
            right_mult=1.1,
            up_mult=1.1,
            forward_mult=0.05,
        )

        self._camera_controller = CameraController(self, self.camera, rot_smooth_hz=4)

    def _setup_graphics(self) -> None:
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogf(GL_FOG_DENSITY, FOGDENSITY)
        glFogfv(GL_FOG_COLOR, LIGHT_BLUE)
        glHint(GL_FOG_HINT, GL_FASTEST)



        #TODO:Total rewrite to work with #skyrenderer
        self.sun_pos = Vector3(self.world_center.x + 10000.0, 20000.0, self.world_center.z + 3000.0)
        _sd = Vector3(self.world_center.x, 0.0, self.world_center.z) - self.sun_pos
        sd_len = _sd.length()
        if sd_len != 0:
            _sd = _sd / sd_len
        self.sun_direction = _sd

    def _load_assets(self) -> None:
        print("Beginning asset loading...")
        tex = load_world_textures()
        self.ground_tex = tex.get("ground_tex")
        self.road_tex = tex.get("road_tex")
        self.tree_textures = tex.get("tree_textures", [])
        self.grasses_textures = tex.get("grasses_textures", [])
        self.rock_textures = tex.get("rock_textures", [])
        self.fence_textures = tex.get("fence_textures", [])

        Sounds.ensure_init()
        Sounds.load_optional("footstep", LEAVES02_SOUND_PATH)
        Sounds.load_optional("ambient_birds", BIRDS_SOUND_PATH)
        Sounds.load_optional("step", STEP1_SOUND_PATH)
        print("Asset loading complete.")

        # Initialize visual/UI components after assets are loaded
        start_time = time.perf_counter()
        self.sky = SkyRenderer()
        self._hud = WorldHUD(self)
        self.log_timing("Initializing sky and HUD", start_time, time.perf_counter())

    def _create_world_objects(self, grid_count: int, spacing: float, half: float, grid_tile_size: int, grid_gap: int, tree_count: int, grass_count: int, rock_count: int) -> None:
        start_time = time.perf_counter()
        print("Creating buildings...")
        self.buildings: list[Building] = []
        building_pos = self.world_center + Vector3(0, 0, 200)
        building = Building(position=building_pos)
        self.buildings.append(building)

        self.builder = TexturedGroundGridBuilder(
            count=grid_count,
            tile_size=grid_tile_size,
            gap=grid_gap,
            texture=self.ground_tex,
            brightness_modifiers=self.brightness_modifiers,
            default_brightness=self.camera.brightness_default
        )

        self.log_timing("Creating buildings", start_time, time.perf_counter())

        start_time = time.perf_counter()
        print("Generating ground mesh...")
        self.ground_mesh = self.builder.build()
        self._ground_height_sampler = getattr(self.ground_mesh, "height_sampler", None)
        self.log_timing("Generating ground mesh", start_time, time.perf_counter())

        start_time = time.perf_counter()
        wall_tex = load_texture(WALL1_TEXTURE_PATH)

        for b in self.buildings:
            default_wall_height = 50
            building_width = 500
            building_depth = 100
            if b.target_height is None:
                bx, bz = b.position.x, b.position.z
                sampled_y = self.ground_height_at(bx, bz)
                base_y = sampled_y
            else:
                base_y = None

            self.walls = b.create_perimeter_walls(
                wall_height=default_wall_height,
                wall_thickness=2.5,
                width=building_width,
                depth=building_depth,
                texture=wall_tex,
                uv_repeat=(1.0, 1.0),
                base_y=base_y,
            )

        print(f"Built {len(self.walls)} building walls.")
        self.log_timing("Building walls", start_time, time.perf_counter())
        self.wall_tiles.extend(self.walls)


        tri_thickness = 5
        self.showcase_polygons: list[Polygon] = []
        off_ground = 40

        def regular_polygon(cx: float, cy: float, radius: float, sides: int) -> list[tuple[float, float]]:
            pts: list[tuple[float, float]] = []
            for i in range(sides):
                ang = math.radians(90.0 + 360.0 * i / sides)
                x = cx + math.cos(ang) * radius
                y = cy + math.sin(ang) * radius
                pts.append((x, y))
            return pts

        # Triangle
        triangle_points = [(0, 0), (60, 0), (30, 50)]
        self.showcase_polygons.append(
            Polygon(
                position=Vector3(self.world_center.x, self.ground_height_at(self.world_center.x, self.world_center.z) + off_ground, self.world_center.z),
                points_2d=triangle_points,
                thickness=tri_thickness,
                texture=wall_tex,
            )
        )

        # Square
        square_points = [(0, 0), (40, 0), (40, 40), (0, 40)]
        self.showcase_polygons.append(
            Polygon(
                position=Vector3(self.world_center.x - 100, self.ground_height_at(self.world_center.x - 100, self.world_center.z) + off_ground, self.world_center.z),
                points_2d=square_points,
                thickness=tri_thickness,
                texture=wall_tex,
            )
        )

        # Pentagon
        pent_points = regular_polygon(0.0, 0.0, 30.0, 5)
        self.showcase_polygons.append(
            Polygon(
                position=Vector3(self.world_center.x + 100, self.ground_height_at(self.world_center.x + 100, self.world_center.z) + off_ground, self.world_center.z),
                points_2d=pent_points,
                thickness=tri_thickness,
                texture=wall_tex,
            )
        )

        # Arrow
        arrow_points = [(0, 10), (40, 10), (40, -10), (60, 20), (40, 50), (40, 30), (0, 30)]
        self.showcase_polygons.append(
            Polygon(
                position=Vector3(self.world_center.x - 200, self.ground_height_at(self.world_center.x - 200, self.world_center.z) + off_ground, self.world_center.z - 200),
                points_2d=arrow_points,
                thickness=tri_thickness,
                texture=wall_tex,
            )
        )

        # L-shape
        l_points = [(0, 0), (60, 0), (60, 20), (20, 20), (20, 80), (0, 80)]
        self.showcase_polygons.append(
            Polygon(
                position=Vector3(self.world_center.x + 230, self.ground_height_at(self.world_center.x + 230, self.world_center.z) + off_ground, self.world_center.z),
                points_2d=l_points,
                thickness=tri_thickness,
                texture=wall_tex,
            )
        )

        start_time = time.perf_counter()
        self.log_timing("Showcase polygons", start_time, time.perf_counter())
        self.polygons.extend(self.showcase_polygons)

        start_time = time.perf_counter()
        print("Spawning world objects...")
        center_x = (self.ground_bounds[0] + self.ground_bounds[1]) * 0.5
        center_z = (self.ground_bounds[2] + self.ground_bounds[3]) * 0.5
        road_y = self.ground_height_at(center_x, center_z) + 1
        road_points = [
            (self.ground_bounds[0], center_z),
            (self.ground_bounds[1], center_z),
        ]
        road_width = 60.0

        if len(road_points) >= 2:
            x0, z0 = road_points[0]
            x1, z1 = road_points[1]
            t = 0.15
            sx = x0 + (x1 - x0) * t
            sz = z0 + (z1 - z0) * t
        else:
            sx, sz = road_points[0]

        min_x, max_x, min_z, max_z = self.ground_bounds
        margin = 1.0
        sx = max(min_x + margin, min(max_x - margin, sx))
        sz = max(min_z + margin, min(max_z - margin, sz))

        self.camera.position = self.world_center

        self.road = Road(
            points=road_points,
            ground_y=road_y,
            width=road_width,
            texture=self.road_tex,
            px_to_world=1.0,
            v_tiles=1.0,
            height_sampler=self._ground_height_sampler,
            elevation=3.0,
            segment_length=8.0,
            brightness_modifiers=self.brightness_modifiers,
            default_brightness=self.camera.brightness_default,
        )

        self.log_timing("Create road", start_time, time.perf_counter())
        self.others.append(self.road)
        start_time = time.perf_counter()
        self.trees = spawn_world_sprites(
            self,
            count=tree_count,
            textures=self.tree_textures,
            px_to_world=1.2,
            camera=self.camera,
            x_off=(self.ground_bounds[0] + self.ground_bounds[1]) / 2 + 25,
            z_off=(self.ground_bounds[2] + self.ground_bounds[3]) / 2 + 25,
            max_spawn_x=(self.ground_bounds[1] - self.ground_bounds[0]) / 2,
            max_spawn_z=(self.ground_bounds[3] - self.ground_bounds[2]) / 2,
            avoid_roads=[self.road],
            avoid_areas=self.buildings,
        )
        print(f"Spawned {len(self.trees)} trees.")
        self.log_timing("Spawn trees", start_time, time.perf_counter())
        self.sprite_items.extend(self.trees)
        start_time = time.perf_counter()
        self.grasses = spawn_world_sprites(
            self,
            count=grass_count,
            textures=self.grasses_textures,
            px_to_world=1.5,
            camera=self.camera,
            x_off=(self.ground_bounds[0] + self.ground_bounds[1]) / 2,
            z_off=(self.ground_bounds[2] + self.ground_bounds[3]) / 2,
            max_spawn_x=(self.ground_bounds[1] - self.ground_bounds[0]) / 2,
            max_spawn_z=(self.ground_bounds[3] - self.ground_bounds[2]) / 2,
            avoid_roads=[self.road],
            avoid_areas=self.buildings,
        )
        print(f"Spawned {len(self.grasses)} grasses.")
        self.log_timing("Spawn grasses", start_time, time.perf_counter())
        self.sprite_items.extend(self.grasses)

        start_time = time.perf_counter()
        self.rocks = spawn_world_sprites(
            self,
            count=rock_count,
            textures=self.rock_textures,
            px_to_world=1.0,
            camera=self.camera,
            x_off=(self.ground_bounds[0] + self.ground_bounds[1]) / 2,
            z_off=(self.ground_bounds[2] + self.ground_bounds[3]) / 2,
            max_spawn_x=(self.ground_bounds[1] - self.ground_bounds[0]) / 2,
            max_spawn_z=(self.ground_bounds[3] - self.ground_bounds[2]) / 2,
            avoid_roads=[self.road],
            avoid_areas=self.buildings,
        )
        print(f"Spawned {len(self.rocks)} rocks.")
        self.log_timing("Spawn rocks", start_time, time.perf_counter())
        self.sprite_items.extend(self.rocks)

        start_time = time.perf_counter()
        self.fence_meshes = build_textured_fence_ring(
            min_x=self.ground_bounds[0],
            max_x=self.ground_bounds[1],
            min_z=self.ground_bounds[2],
            max_z=self.ground_bounds[3],
            ground_y=self.ground_height_at(0, 0),
            height_sampler=getattr(self.ground_mesh, "height_sampler", None),
            textures=[t for t in self.fence_textures if t is not None],
            px_to_world=1.0,
            wave_amp=0.5,
            wave_freq=0.02,
            wave_phase=0.3,
            brightness_modifiers=self.brightness_modifiers,
            default_brightness=self.camera.brightness_default,
        )
        print(f"Built {len(self.fence_meshes)} fence segments.")
        self.log_timing("Build fences", start_time, time.perf_counter())
        

        start_time = time.perf_counter()

        #self.static_meshes = meshes + trees + grasses + rocks + [self.road]
        self.log_timing("Assemble static meshes", start_time, time.perf_counter())

        start_time = time.perf_counter()
        shadow_texture = create_shadow_texture(
            width_px=256,
            height_px=256,
            max_alpha=0.8,
            inner_ratio=0.02,
            outer_ratio=1,
            falloff_exp=0.55,
            pixelated=True,
            pixel_scale=16
        )
        print("Created shadow texture.")
        self.log_timing("Create shadow texture", start_time, time.perf_counter())
        
        decals: list[Decal] = []
        rng = random.Random()

        def make_decal_for_sprite(s: WorldSprite) -> Decal:
            w, h = s.size
            size_w = w * rng.uniform(0.45, 0.75)
            size_h = h * rng.uniform(0.45, 0.75)

            sun = getattr(self, "sun_direction", None)
            final_w, final_h = size_w, size_h
            offset_x, offset_z = 0.0, 0.0
            base_y = self.ground_height_at(s.position.x, s.position.z)
            center_y = base_y

            if sun is not None:
                proj_x = float(sun.x)
                proj_z = float(sun.z)
                proj_len = math.hypot(proj_x, proj_z)

                if proj_len >= 1e-6:
                    vert = abs(float(sun.y))
                    elong = 1.0 / max(0.05, vert)
                    elong = max(1.0, min(elong, 12.0))

                    seed = max(size_w, size_h)
                    major = max(14.0, min(400.0, seed * (0.9 + elong * 0.6)))
                    minor = max(8.0, min(200.0, min(size_w, size_h) * 0.9))
                    final_w, final_h = major, minor

                    dir_x = -proj_x / proj_len
                    dir_z = -proj_z / proj_len

                    offset_x = (-dir_x * (major * 0.45))
                    offset_z = -dir_z * (major * 0.5)

                    angle_rad = math.atan2(-proj_x, -proj_z)
                    angle_deg = math.degrees(angle_rad)
                    rot = (angle_deg + 90.0) % 360.0

                base_y = self.ground_height_at(
                    s.position.x + offset_x,
                    s.position.z + offset_z
                )
                center_y = base_y

                return Decal(
                    center=Vector3(s.position.x + offset_x, center_y, s.position.z + offset_z),
                    size=(final_w, final_h),
                    texture=shadow_texture,
                    rotation_deg=rot,
                    subdiv_u=8,
                    subdiv_v=8,
                    height_fn=self.ground_height_at,
                    elevation=1,
                    uv_repeat=(1.0, 1.0),
                    color=(1.0, 1.0, 1.0),
                    build_vbo=True,
                )

        start_time = time.perf_counter()
        for s in self.trees:
            decals.append(make_decal_for_sprite(s))

        print(f"Created {len(decals)} shadow decals.")
        self.log_timing("Create decals", start_time, time.perf_counter())
        self.decal_batch = DecalBatch.build(decals)
        self.decal_batches.append(self.decal_batch)
        start_time = time.perf_counter()
        self.log_timing("Build decal batch", start_time, time.perf_counter())

        

    def log_timing(self, message: str, start_time: float, end_time: float, log: bool = False):
        """Logs timing information for WorldScene setup phases."""
        if log:
            print(f"{message} took {end_time - start_time:.6f} seconds")

    def draw_sky(self) -> None:  # pragma: no cover - visual
        """Draw sky elements (delegated from engine)."""
        self.sky.draw(self.camera, sun_direction=self.sun_direction)

    def draw(self, enable_timing: bool = False):  # pragma: no cover - visual
        self.ground_mesh.draw()
        for m in self.fence_meshes:
            m.draw()
        glEnable(GL_FOG)
        super().draw(enable_timing=enable_timing)
        self._hud.draw()

    def contains_horizontal(self, pos: Vector3) -> bool:
        min_x, max_x, min_z, max_z = self.ground_bounds
        extra = -15.0
        return (min_x - extra <= pos.x <= max_x + extra) and (
            min_z - extra <= pos.z <= max_z + extra
        )

    def is_on_road(self, x: float, z: float, *, margin: float = 0.0) -> bool:
        r = getattr(self, "road", None)
        return bool(r and r.contains_point(x, z, margin=margin))

    def ground_height_at(self, x: float, z: float) -> float:
        sampler = getattr(self, "_ground_height_sampler", None)
        if sampler is not None and hasattr(sampler, "height_at"):
            try:
                return float(sampler.height_at(x, z))
            except Exception:
                pass
        fn = getattr(self, "_height_fn", None)
        return float(fn(x, z)) if callable(fn) else 5.0

    def view_space_position(
        self, *, dist: float, nx: float, ny: float, px: float = 0.0, py: float = 0.0
    ) -> Vector3:
        aspect = WIDTH / HEIGHT
        half_h = dist * math.tan(math.radians(FOV * 0.5))
        half_w = half_h * aspect
        wu_per_px_x = (2.0 * half_w) / WIDTH
        wu_per_px_y = (2.0 * half_h) / HEIGHT

        right = self.camera._right
        forward = self.camera._forward
        up = getattr(self.camera, "_up", right.cross(forward))

        center = self.camera.position + forward * dist
        off_right = (nx * half_w) + (px * wu_per_px_x)
        off_up = (ny * half_h) - (py * wu_per_px_y)
        return center + (right * off_right) + (up * off_up)

    def update(self, dt: float) -> None:
        if not Sounds.is_playing("ambient_birds"):
            Sounds.play("ambient_birds", volume=0.05)

        moving, sprinting = self._camera_controller.update(dt)
        self._sway_controller.update(dt)

        ground_y_here = self.ground_height_at(
            self.camera.position.x, self.camera.position.z
        )

        manual_offset = getattr(self.camera, "manual_height_offset", 0.0)
        target_cam_y = ground_y_here + CAMERA_GROUND_OFFSET + float(manual_offset)
        if CAMERA_FOLLOW_SMOOTH_HZ <= 0 or dt <= 0:
            self.camera.position.y = target_cam_y
        else:
            a = 1.0 - math.exp(-CAMERA_FOLLOW_SMOOTH_HZ * dt)
            self.camera.position.y += (target_cam_y - self.camera.position.y) * a

        self._hud.update(dt)
        self._headbob.update(moving=moving, sprinting=sprinting, dt=dt)

    def handle_event(self, event) -> None:
        pass

    def render(
        self, *, show_hud: bool = True, text=None, fps: float | None = None
    ):  # pragma: no cover - visual
        brightness = self.camera.brightness_default
        rgba = LIGHT_BLUE
        rgba = [c * brightness for c in rgba]
        rgba[3] = 0

        glFogfv(GL_FOG_COLOR, rgba)
        glClearColor(*rgba)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV, WIDTH / HEIGHT, 1, 1_000_000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.draw_sky()

        glRotatef(math.degrees(-self.camera.rotation.x), 1, 0, 0)
        glRotatef(math.degrees(-self.camera.rotation.y), 0, 1, 0)
        off_x, off_y = self._headbob.offsets()
        if HEADBOB_ENABLED:
            glTranslatef(-off_x, -off_y, 0)
        else:
            if off_y != 0.0:
                glTranslatef(0, -off_y, 0)
        glTranslatef(
            -self.camera.position.x, -self.camera.position.y, -self.camera.position.z
        )

        self.draw()

        if show_hud and text is not None and fps is not None:
            text.begin()
            text.draw_text(f"FPS: {fps:5.1f}", 12, 10, key="fps", align="topleft", color=[255,0,0,0])
            lorem = (
                "Lore Epsum: Vivamus sed nibh.\n"
                "Curabitur at leo quis nunc posuere congue.\n"
                "Praesent tristique sem at augue pharetra."
            )
            text.draw_text_multiline(lorem, 12, HEIGHT - 12, align="bottomleft")
            text.end()

    def apply_mouse_delta(self, dx: float, dy: float, dt: float | None = None) -> None:
        try:
            try:
                self._camera_controller.on_mouse_delta(dx, dy, dt)
            except TypeError:
                self._camera_controller.on_mouse_delta(dx, dy)
        except Exception:
            pass