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
from camera import Camera
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
    create_test_texture,
)
from textures.texture_manager import load_world_textures
from textures.resoucepath import *

from sound.sound_utils import Sounds
from render.sky_renderer import SkyRenderer
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
)
from OpenGL.GLU import gluPerspective


# It will be a sprite on the screen floating acting as if in the player hands
# class WorldSprite:
#     position: Vector3
#     size: tuple[float, float]
#     texture: int
#     camera: any  # expects Camera with _right and _forward vectors updated via update_rotation()
#     color: tuple[float, float, float] = (1.0, 1.0, 1.0)
class test_sword(WorldSprite):
    # it needs it's supers size, pos, cam & texture
    def __init__(
        self, position: Vector3, size: Tuple[float, float], camera: Camera, texture: int
    ):
        self.position = position
        self.size = size
        self.camera = camera
        self.texture = texture
        super().__init__(position=position, size=size, camera=camera, texture=texture)


class WorldScene(Scene):

    def __init__(
        self,
        camera: Optional[Camera] = None,
        *,
        grid_count: int = 50,
        grid_tile_size: int = 100,
        grid_gap: int = 0,
        tree_count: int = 2000,
        grass_count: int = 1000,
        rock_count: int = 1000,
        area_offset: Optional[Tuple[float, float]] = None,
        spawn_limits: Optional[Tuple[float, float]] = None,
    ) -> None:

        print("World Scene Initialized")

        # If no camera provided (default), create one suitable for this scene
        cam = camera or Camera(
            position=Vector3(STARTING_POS), width=WIDTH, height=HEIGHT, fov=FOV
        )

        super().__init__(camera=cam)

        # Create head-bob instance directly and wire up footstep handler
        def _headbob_on_footstep(intensity, sprinting, phase, foot):
            try:
                # Let the scene override the behavior if it provides on_footstep
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
                        # fall through to default handling
                        pass

                # Default footstep behavior: scale volume slightly with intensity
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

        spacing = grid_tile_size + grid_gap
        half = grid_tile_size / 2.0
        # calculate using gap, tile_size and grid_count
        self.world_center = Vector3(
            (grid_count * spacing) / 2, 0, (grid_count * spacing) / 2
        )
        self.ground_bounds = (
            0 + half,
            grid_count * spacing - half,
            0 + half,
            grid_count * spacing - half,
        )

        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogf(GL_FOG_DENSITY, FOGDENSITY)
        glFogfv(GL_FOG_COLOR, LIGHT_BLUE)
        glHint(GL_FOG_HINT, GL_FASTEST)

        self.sky = SkyRenderer()
        self._hud = WorldHUD(self)

        # Fake sun direction (unit vector pointing from sun toward the world).
        # Change this to move the sun: X,Z controls azimuth, Y should be negative
        # for a sun above the scene. We'll normalize it below.
        _sd = Vector3(0.2, -1.0, 0.3)
        sd_len = _sd.length()
        if sd_len != 0:
            _sd = _sd / sd_len
        self.sun_direction = _sd

        print("Beginning asset loading...")
        tex = load_world_textures()
        ground_tex = tex.get("ground_tex")
        road_tex = tex.get("road_tex")
        tree_textures = tex.get("tree_textures", [])
        grasses_textures = tex.get("grasses_textures", [])
        rock_textures = tex.get("rock_textures", [])
        fence_textures = tex.get("fence_textures", [])

        # Prepare sounds (safe if file missing / mixer unavailable)
        Sounds.ensure_init()
        Sounds.load_optional("footstep", LEAVES02_SOUND_PATH)
        Sounds.load_optional("ambient_birds", BIRDS_SOUND_PATH)
        Sounds.load_optional("step", STEP1_SOUND_PATH)
        print("Asset loading complete.")

        print("Creating buildings...")
        self.buildings: list[Building] = []

        building_height = 0
        building = Building(
            position=self.world_center,
            target_height=building_height,
        )
        self.buildings.append(building)

        builder = TexturedGroundGridBuilder(
            count=grid_count,
            tile_size=grid_tile_size,
            gap=grid_gap,
            texture=ground_tex,
            height_modifiers=None,
        )

        print("Generating ground mesh...")
        self.ground_mesh = builder.build()

        # If ground mesh exposes a height sampler, use it for precise ground queries
        self._ground_height_sampler = getattr(self.ground_mesh, "height_sampler", None)

        print("Spawning world objects...")
        # Example road from west edge toward east edge across center
        center_x = (self.ground_bounds[0] + self.ground_bounds[1]) * 0.5
        center_z = (self.ground_bounds[2] + self.ground_bounds[3]) * 0.5
        # Sample ground height near center for road elevation baseline
        road_y = self.ground_height_at(center_x, center_z) + 1

        # Straight road across the center
        road_points = [
            (self.ground_bounds[0], center_z),
            (self.ground_bounds[1], center_z),
        ]

        road_width = 60.0

        # Place camera closer to the entry edge: interpolate a small step from p0 toward p1
        if len(road_points) >= 2:
            x0, z0 = road_points[0]
            x1, z1 = road_points[1]
            t = 0.15  # small step in from the edge (lower = closer to edge)
            sx = x0 + (x1 - x0) * t
            sz = z0 + (z1 - z0) * t
        else:
            sx, sz = road_points[0]

        # Clamp inside playable area with a smaller margin to avoid pushing too far in
        min_x, max_x, min_z, max_z = self.ground_bounds
        margin = 1.0
        sx = max(min_x + margin, min(max_x - margin, sx))
        sz = max(min_z + margin, min(max_z - margin, sz))

        self.camera.position = building.center

        self.road = Road(
            points=road_points,
            ground_y=road_y,
            width=road_width,
            texture=road_tex,
            px_to_world=1.0,
            v_tiles=1.0,
            height_sampler=self._ground_height_sampler,
            elevation=3.0,
            segment_length=8.0,
        )
        print("Road created.")
        lil_buffer = 15
        # Center spawns around the middle of the ground, not shifted to +X/+Z
        min_x, max_x, min_z, max_z = self.ground_bounds
        center_x = (min_x + max_x) * 0.5
        center_z = (min_z + max_z) * 0.5
        # Allow a small buffer so a few sprites can appear just beyond the fence symmetrically
        half_extent_x = ((max_x - min_x) * 0.5) + lil_buffer
        half_extent_z = ((max_z - min_z) * 0.5) + lil_buffer
        if area_offset is None:
            area_offset = (center_x, center_z)
        if spawn_limits is None:
            spawn_limits = (half_extent_x, half_extent_z)

        # Populate vegetation
        x_off, z_off = area_offset
        max_spawn_x, max_spawn_z = spawn_limits

        # Generate sprites via helper to reduce duplication
        trees = spawn_world_sprites(
            self,
            count=tree_count,
            textures=tree_textures,
            px_to_world=1.2,  # ~100px trees -> ~120 world units tall
            camera=self.camera,
            x_off=x_off,
            z_off=z_off,
            max_spawn_x=max_spawn_x,
            max_spawn_z=max_spawn_z,
            avoid_roads=[self.road],
            avoid_areas=self.buildings,
        )
        print(f"Spawned {len(trees)} trees.")
        grasses = spawn_world_sprites(
            self,
            count=grass_count,
            textures=grasses_textures,
            px_to_world=1.5,  # ~10px grass -> ~15 world units tall
            camera=self.camera,
            x_off=x_off,
            z_off=z_off,
            max_spawn_x=max_spawn_x,
            max_spawn_z=max_spawn_z,
            avoid_roads=[self.road],
            avoid_areas=self.buildings,
        )
        print(f"Spawned {len(grasses)} grasses.")
        rocks = spawn_world_sprites(
            self,
            count=rock_count,
            textures=rock_textures,
            px_to_world=1.0,  # ~10px rocks -> ~10 world units tall
            camera=self.camera,
            x_off=x_off,
            z_off=z_off,
            max_spawn_x=max_spawn_x,
            max_spawn_z=max_spawn_z,
            avoid_roads=[self.road],
            avoid_areas=self.buildings,
        )
        print(f"Spawned {len(rocks)} rocks.")



        # Build fence ring just outside ground bounds so it sits at the edge
        min_x, max_x, min_z, max_z = self.ground_bounds
        fence_inset = 0.5  # nudge slightly to avoid z-fighting with ground edges
        fence_min_x = min_x - fence_inset
        fence_max_x = max_x + fence_inset
        fence_min_z = min_z - fence_inset
        fence_max_z = max_z + fence_inset

        # If the ground mesh exposes a height_sampler, pass it so the fence follows terrain
        fence_meshes = build_textured_fence_ring(
            min_x=fence_min_x,
            max_x=fence_max_x,
            min_z=fence_min_z,
            max_z=fence_max_z,
            ground_y=self.ground_height_at(0, 0),
            height_sampler=getattr(self.ground_mesh, "height_sampler", None),
            textures=[t for t in fence_textures if t is not None],
            px_to_world=1.0,
            wave_amp=0.5,
            wave_freq=0.02,
            wave_phase=0.3,
        )
        print(f"Built {len(fence_meshes)} fence segments.")

        meshes = [self.road]
        meshes.extend(fence_meshes)

        # Create perimeter walls for any buildings and add them to the scene meshes
        # so they are rendered. We load the shared wall texture and pass it into
        # the building helper; WallTile will compute sensible UV repeats at draw
        # time when uv_repeat is left at the default (1.0,1.0).
        wall_tex = load_texture(WALL1_TEXTURE_PATH)

        for b in self.buildings:
            base_y = None
            default_wall_height = 50

            walls = b.create_perimeter_walls(
                wall_height=default_wall_height,
                wall_thickness=2.5,
                width=500,
                texture=wall_tex,
                uv_repeat=(1.0, 1.0),
                base_y=base_y,
            )

            meshes.extend(walls)

        print(f"Built {len(walls)} building walls.")

        self.static_meshes = meshes + trees + grasses + rocks

        # --- Spawn procedural shadow decals under vegetation -------------
        # Build a reusable high‑quality blob shadow texture (linear filtered)
        shadow_texture = create_shadow_texture(
            width_px=256,
            height_px=256,
            max_alpha=0.26,
            inner_ratio=0.22,
            outer_ratio=0.96,
            falloff_exp=2.2,
        )
        print("Created shadow texture.")

        decals: list[Decal] = []
        rng = random.Random()

        def make_decal_for_sprite(s: WorldSprite) -> Decal:
            # Base oval shaped shadow scaled to sprite width/height in world units
            w, h = s.size
            # Slightly smaller than footprint, with variability
            size_w = max(14.0, min(200.0, float(w) * rng.uniform(0.35, 0.55)))
            size_h = max(10.0, min(160.0, float(h) * rng.uniform(0.25, 0.45)))
            rot = rng.uniform(0.0, 360.0)
            return Decal(
                center=Vector3(s.position.x, 0.0, s.position.z),
                size=(size_w, size_h),
                texture=shadow_texture,
                rotation_deg=rot,
                subdiv_u=8,
                subdiv_v=8,
                height_fn=self.ground_height_at,
                elevation=random.uniform(0.15, 0.35),
                uv_repeat=(1.0, 1.0),
                color=(1.0, 1.0, 1.0),
                build_vbo=True,  # we'll batch into one VBO per texture
            )

        for s in trees:
            decals.append(make_decal_for_sprite(s))
        for s in grasses:
            decals.append(make_decal_for_sprite(s))
        for s in rocks:
            decals.append(make_decal_for_sprite(s))

        print(f"Created {len(decals)} shadow decals.")

        # Build single-VBO-per-texture batches and store them as drawables
        decal_batch = DecalBatch.build(decals)

        # Attach a simple wrapper so Scene.draw can call draw() directly
        self.static_meshes.append(decal_batch)
        print("World scene initialization complete.")

    def draw_sky(self) -> None:  # pragma: no cover - visual
        """Draw sky elements (delegated from engine)."""
        self.sky.draw(self.camera)

    def draw(self):  # pragma: no cover - visual
        # Ensure fog is enabled for world rendering

        self.ground_mesh.draw()

        glEnable(GL_FOG)
        super().draw()
        # Draw HUD elements (held item + compass)
        try:
            self._hud.draw()
        except Exception:
            pass

    def draw_overlay(self) -> None:  # pragma: no cover - visual
        """Draw 2D overlay elements that belong to the world scene.

        Currently draws the world shade to slightly darken the world.
        """
        # Delegate overlay drawing to HUD
        try:
            self._hud.draw_overlay()
        except Exception:
            pass

    # Convenience for engine movement clamp
    def contains_horizontal(self, pos: Vector3) -> bool:
        min_x, max_x, min_z, max_z = self.ground_bounds
        extra = -15.0  # back up 15 units before reaching the ground bounds
        return (min_x - extra <= pos.x <= max_x + extra) and (
            min_z - extra <= pos.z <= max_z + extra
        )

    # --- Internals -------------------------------------------------------

    # Convenience road test for other systems (e.g., movement/speed)
    def is_on_road(self, x: float, z: float, *, margin: float = 0.0) -> bool:
        r = getattr(self, "road", None)
        return bool(r and r.contains_point(x, z, margin=margin))

    def ground_height_at(self, x: float, z: float) -> float:
        """Return top surface Y of the ground at (x,z).

        Prefer sampling from the generated ground mesh if available. Fallback to any
        configured function or the legacy flat height of 5.0.
        """
        sampler = getattr(self, "_ground_height_sampler", None)
        if sampler is not None and hasattr(sampler, "height_at"):
            try:
                return float(sampler.height_at(x, z))
            except Exception:
                pass
        fn = getattr(self, "_height_fn", None)
        return float(fn(x, z)) if callable(fn) else 5.0

    # Compute a world position in front of the camera using intuitive screen placement
    # nx, ny are normalized screen coords in [-1, 1] (x: -1 left .. +1 right, y: -1 bottom .. +1 top)
    # dist is forward distance in world units; optional px,py offsets are in screen pixels (+y is down)
    def view_space_position(
        self, *, dist: float, nx: float, ny: float, px: float = 0.0, py: float = 0.0
    ) -> Vector3:
        aspect = WIDTH / HEIGHT
        half_h = dist * math.tan(math.radians(FOV * 0.5))
        half_w = half_h * aspect
        # world units per pixel at this depth
        wu_per_px_x = (2.0 * half_w) / WIDTH
        wu_per_px_y = (2.0 * half_h) / HEIGHT

        right = self.camera._right
        forward = self.camera._forward
        up = getattr(self.camera, "_up", right.cross(forward))

        center = self.camera.position + forward * dist
        off_right = (nx * half_w) + (px * wu_per_px_x)
        off_up = (ny * half_h) - (py * wu_per_px_y)  # screen +y is down
        return center + (right * off_right) + (up * off_up)

    # update with super call
    def update(self, dt: float) -> None:
        """Update the world scene.

        This is called by the engine each frame.
        """
        # Start ambient loop if not playing
        if not Sounds.is_playing("ambient_birds"):
            Sounds.play("ambient_birds", volume=0.05)

        moving, sprinting = self._camera_controller.update(dt)

        self._sway_controller.update(dt)

        # Make camera follow terrain: set Y to ground height + offset (with smoothing)
        ground_y_here = self.ground_height_at(
            self.camera.position.x, self.camera.position.z
        )

        target_cam_y = ground_y_here + CAMERA_GROUND_OFFSET
        if CAMERA_FOLLOW_SMOOTH_HZ <= 0 or dt <= 0:
            self.camera.position.y = target_cam_y
        else:
            a = 1.0 - math.exp(-CAMERA_FOLLOW_SMOOTH_HZ * dt)
            self.camera.position.y += (target_cam_y - self.camera.position.y) * a

        self._hud.update(dt)
        self._headbob.update(moving=moving, sprinting=sprinting, dt=dt)

        super().update(dt)

    # Event handling moved into scene
    def handle_event(self, event) -> None:
        pass

    # Full render for 3D scene (projection, view, sky, world, overlay)
    def render(
        self, *, show_hud: bool = True, text=None, fps: float | None = None
    ):  # pragma: no cover - visual

        # Clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV, WIDTH / HEIGHT, 1, 1_000_000.0)

        # View
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Draw sky first
        self.draw_sky()

        # Camera orientation & head-bob local offset
        glRotatef(math.degrees(-self.camera.rotation.x), 1, 0, 0)
        glRotatef(math.degrees(-self.camera.rotation.y), 0, 1, 0)
        off_x, off_y = self._headbob.offsets()
        # If headbob is enabled, apply its full offsets (which include idle).
        # If headbob is disabled we still apply the idle vertical offset (off_y)
        # so the idle behavior remains active even when headbob is turned off.
        if HEADBOB_ENABLED:
            glTranslatef(-off_x, -off_y, 0)
        else:
            if off_y != 0.0:
                glTranslatef(0, -off_y, 0)
        glTranslatef(
            -self.camera.position.x, -self.camera.position.y, -self.camera.position.z
        )

        # World
        self.draw()

        # Overlay
        self.draw_overlay()

        # Optional HUD (delegated from Engine)
        if show_hud and text is not None and fps is not None:
            text.begin()
            text.draw_text(f"FPS: {fps:5.1f}", 12, 10, key="fps", align="topleft")
            lorem = (
                "Lore Epsum: Vivamus sed nibh.\n"
                "Curabitur at leo quis nunc posuere congue.\n"
                "Praesent tristique sem at augue pharetra."
            )
            text.draw_text_multiline(lorem, 12, HEIGHT - 12, align="bottomleft")
            text.end()

    # Mouse look targets updated from Engine each frame via mouse deltas
    def apply_mouse_delta(self, dx: float, dy: float) -> None:
        # Forward mouse delta to the camera controller (which also forwards sway)
        try:
            self._camera_controller.on_mouse_delta(dx, dy)
        except Exception:
            pass
