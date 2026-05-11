"""WorldScene setup helpers.

This module owns the bootstrap responsibilities that do not need to live in the
scene class body: lighting/fog setup, controller wiring, asset loading, and UI
component creation.
"""

from __future__ import annotations

import random
import time

from pygame.math import Vector3

from game.config import *
from engine.camera.headbob import HeadBob
from engine.camera.sway_controller import SwayController
from engine.rendering.lighting import SceneLighting
from engine.rendering.sky_renderer import SkyRenderer
from engine.sound.sound_utils import Sounds
from game.resources.paths import *
from game.resources.texture_manager import load_world_textures
from game.world.player_controller import PlayerCameraController
from game.world.ui.pause_menu import PauseMenu
from game.world.ui.setting_menu import SettingMenu
from game.world.ui.world_hud import WorldHUD

from OpenGL.GL import (
    glEnable,
    glFogf,
    glFogfv,
    glFogi,
    glHint,
    GL_EXP2,
    GL_FASTEST,
    GL_FOG,
    GL_FOG_COLOR,
    GL_FOG_DENSITY,
    GL_FOG_HINT,
    GL_FOG_MODE,
)


def setup_brightness_areas(scene, grid_count: int, spacing: float, half: float) -> None:
    brightness_modifiers = []
    min_x = 0 + half
    max_x = grid_count * spacing - half
    min_z = 0 + half
    max_z = grid_count * spacing - half
    for _ in range(0):
        cx = random.triangular(min_x, max_x, (min_x + max_x) * 0.5) + 1e-6
        cz = random.triangular(min_z, max_z, (min_z + max_z) * 0.5) + 1e-6
        radius = random.uniform(150.0, 250.0)
        brightness_modifiers.append(
            (
                Vector3(cx, 0, cz),
                radius,
                random.uniform(0.5, 0.8),
                4,
            )
        )

    for modifier in brightness_modifiers:
        try:
            scene.camera.add_brightness_area(*modifier)
        except Exception:
            pass
    scene.brightness_modifiers = brightness_modifiers
    lighting = getattr(scene, "lighting", None)
    if lighting is not None:
        lighting.set_brightness_modifiers(brightness_modifiers)


def setup_controllers(scene) -> None:
    def _headbob_on_footstep(intensity, sprinting, phase, foot):
        try:
            if hasattr(scene, "on_footstep"):
                try:
                    scene.on_footstep(
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
                if scene.is_on_road(scene.camera.position.x, scene.camera.position.z):
                    Sounds.play("step", volume=min(1.0, vol * 0.9))
                else:
                    Sounds.play("footstep", volume=min(1.0, vol))
            except Exception:
                pass
        except Exception:
            pass

    scene._headbob = HeadBob(
        enabled=HEADBOB_ENABLED,
        frequency=HEADBOB_FREQUENCY,
        amplitude_y=HEADBOB_AMPLITUDE,
        amplitude_x=HEADBOB_AMPLITUDE_SIDE,
        sprint_mult=HEADBOB_SPRINT_MULT,
        damping=HEADBOB_DAMPING,
        on_footstep=_headbob_on_footstep,
    )

    scene._sway_controller = SwayController(
        max_x=1.25,
        max_y=0.75,
        mouse_scale=0.01,
        responsiveness=12.0,
        return_rate=8.0,
        right_mult=1.1,
        up_mult=1.1,
        forward_mult=0.05,
    )

    scene._camera_controller = PlayerCameraController(
        scene,
        scene.camera,
        rot_smooth_hz=4,
    )


def setup_graphics(scene) -> None:
    glEnable(GL_FOG)
    glFogi(GL_FOG_MODE, GL_EXP2)
    glFogf(GL_FOG_DENSITY, FOGDENSITY)
    glFogfv(GL_FOG_COLOR, LIGHT_BLUE)
    glHint(GL_FOG_HINT, GL_FASTEST)

    scene.lighting = SceneLighting.from_world_center(
        scene.world_center,
        sky_color=LIGHT_BLUE,
        base_brightness=getattr(scene.camera, "brightness_default", 1.0),
    )
    scene.lighting.set_brightness_modifiers(
        getattr(scene, "brightness_modifiers", ()),
    )
    scene.lighting.set_covered_regions(getattr(scene, "covered_regions", ()))
    sync_aliases = getattr(scene, "_sync_lighting_aliases", None)
    if callable(sync_aliases):
        sync_aliases()
    else:
        scene.sun_pos = scene.lighting.sun_position
        scene.sun_direction = scene.lighting.sun_direction


def load_assets(scene) -> None:
    print("Beginning asset loading...")
    tex = load_world_textures()
    scene.ground_tex = tex.get("ground_tex")
    scene.road_tex = tex.get("road_tex")
    scene.tree_textures = tex.get("tree_textures", [])
    scene.grasses_textures = tex.get("grasses_textures", [])
    scene.rock_textures = tex.get("rock_textures", [])
    scene.fence_textures = tex.get("fence_textures", [])
    scene.wall_tex = tex.get("wall_tex")
    scene.torch_tex = tex.get("torch_tex")
    scene.door_tex = tex.get("door_tex")
    scene.window_tex = tex.get("window_tex")
    scene.goblin_tex = tex.get("goblin_tex", {})

    Sounds.ensure_init()
    Sounds.load_optional("footstep", LEAVES02_SOUND_PATH)
    Sounds.load_optional("ambient_birds", BIRDS_SOUND_PATH)
    Sounds.load_optional("step", STEP1_SOUND_PATH)
    Sounds.load_optional("door_open", DOOR_OPEN_SOUND_PATH)
    Sounds.load_optional("door_close", DOOR_CLOSE_SOUND_PATH)
    scene.goblin_sound_keys = []
    for index, sound_path in enumerate(GOBLIN_SOUND_PATHS, start=1):
        key = f"goblin_{index}"
        if Sounds.load_optional(key, sound_path) is not None:
            scene.goblin_sound_keys.append(key)
    print("Asset loading complete.")

    start_time = time.perf_counter()
    scene.sky = SkyRenderer(
        sun_texture_path=STAR_TEXTURE_PATH,
        moon_texture_path=MOON_TEXTURE_PATH,
    )
    scene._hud = WorldHUD(scene)
    scene.pause_menu = PauseMenu(scene)
    scene.setting_menu = SettingMenu(scene)
    scene.fov = FOV
    scene.fog_enabled = True
    scene.fog_density = FOGDENSITY
    scene.clouds_enabled = CLOUDS_ENABLED
    scene.cloud_density = CLOUD_DENSITY
    scene.cloud_speed = CLOUD_SPEED
    scene.cloud_opacity = CLOUD_OPACITY
    scene.vibrance = 1.15
    scene.hud_visible = True
    scene.compass_visible = True
    scene.held_item_visible = True
    scene.test_light_visible = True
    scene.debug_text_visible = True
    scene.mouse_sensitivity = MOUSE_SENSITIVITY
    scene.walk_speed = BASE_SPEED
    scene.sprint_speed = SPRINT_SPEED
    scene.road_speed_multiplier = 1.5
    scene.jump_speed = JUMP_SPEED
    scene.gravity = GRAVITY
    scene.camera_follow_smooth_hz = CAMERA_FOLLOW_SMOOTH_HZ
    scene.paused = False
    scene.inventory_open = False
    scene.showing_settings_menu = False
    scene._last_mouse_pos = (0, 0)
    scene.log_timing("Initializing sky and HUD", start_time, time.perf_counter())
