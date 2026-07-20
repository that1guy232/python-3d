"""WorldScene setup helpers.

This module owns the bootstrap responsibilities that do not need to live in the
scene class body: lighting/fog setup, controller wiring, asset loading, and UI
component creation.
"""

from __future__ import annotations

import random
import time

from pygame.math import Vector3

from game.config import (
    BASE_SPEED,
    CAMERA_FOLLOW_SMOOTH_HZ,
    CLOUDS_ENABLED,
    CLOUD_DENSITY,
    CLOUD_OPACITY,
    CLOUD_SPEED,
    FOGDENSITY,
    FOV,
    GOBLIN_BATTLE_LOOK_SMOOTH_HZ,
    GOBLIN_BATTLE_TRIGGER_DISTANCE,
    GRAVITY,
    HEADBOB_AMPLITUDE,
    HEADBOB_AMPLITUDE_SIDE,
    HEADBOB_DAMPING,
    HEADBOB_ENABLED,
    HEADBOB_FREQUENCY,
    HEADBOB_SPRINT_MULT,
    JUMP_SPEED,
    LIGHT_BLUE,
    MOUSE_SENSITIVITY,
    SPRINT_SPEED,
)
from engine.camera.headbob import HeadBob
from engine.camera.sway_controller import SwayController
from engine.rendering.lighting import SceneLighting
from engine.rendering.lighting_state import LocalBrightnessLight
from engine.rendering.sky_renderer import SkyRenderer
from engine.sound.sound_utils import Sounds
from game.resources.paths import (
    BIRDS_SOUND_PATH,
    DOOR_CLOSE_SOUND_PATH,
    DOOR_OPEN_SOUND_PATH,
    GOBLIN_SOUND_PATHS,
    LEAVES02_SOUND_PATH,
    MOON_TEXTURE_PATH,
    STAR_TEXTURE_PATH,
    STEP1_SOUND_PATH,
)
from game.resources.texture_manager import load_world_textures
from game.world.battle_cards import BattleCardLoadout
from game.world.player_controller import PlayerCameraController
from game.world.ui.battle_menu import BattleMenu
from game.world.ui.battle_overlay import BattleResourceOverlay
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


def setup_local_lights(scene, grid_count: int, spacing: float, half: float) -> None:
    """Author initial typed local lights before SceneLighting is constructed."""

    local_lights: list[LocalBrightnessLight] = []
    min_x = 0 + half
    max_x = grid_count * spacing - half
    min_z = 0 + half
    max_z = grid_count * spacing - half
    for _ in range(0):
        cx = random.triangular(min_x, max_x, (min_x + max_x) * 0.5) + 1e-6
        cz = random.triangular(min_z, max_z, (min_z + max_z) * 0.5) + 1e-6
        radius = random.uniform(150.0, 250.0)
        local_lights.append(
            LocalBrightnessLight(
                light_id=f"world:initial:{len(local_lights)}",
                center=(cx, 0.0, cz),
                radius=radius,
                value=random.uniform(0.5, 0.8),
                falloff=4.0,
            )
        )

    replace_query_lights = getattr(
        scene.camera,
        "replace_brightness_query_lights",
        None,
    )
    if callable(replace_query_lights):
        replace_query_lights(local_lights)
    build_state = getattr(scene, "build_state", None)
    if build_state is not None:
        build_state.initial_local_lights = list(local_lights)
    packet_backend = getattr(scene, "lighting_backend", "legacy") == "packet"
    if not packet_backend:
        scene.brightness_modifiers = [
            light.to_legacy_dict() for light in local_lights
        ]
    lighting = getattr(scene, "lighting", None)
    if lighting is not None:
        lighting.replace_local_lights(local_lights, camera=scene.camera)


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
    packet_backend = getattr(scene, "lighting_backend", "legacy") == "packet"
    initial_lights = getattr(
        getattr(scene, "build_state", None),
        "initial_local_lights",
        (),
    )
    scene.lighting.replace_local_lights(
        initial_lights,
        camera=scene.camera,
        project_to_camera=True,
    )
    lighting_controller = getattr(scene, "lighting_controller", None)
    if lighting_controller is not None and not packet_backend:
        lighting_controller.set_legacy_covered_regions(
            getattr(scene, "covered_regions", ())
        )
        lighting_controller.sync_aliases()
    elif not packet_backend:
        scene.sun_pos = scene.lighting.sun_position
        scene.sun_direction = scene.lighting.sun_direction


def load_assets(scene) -> None:
    print("Beginning asset loading...")
    resources = scene.render_resources
    ui_state = scene.ui_state
    tex = load_world_textures()
    resources.ground_tex = tex.get("ground_tex")
    resources.road_tex = tex.get("road_tex")
    resources.tree_textures = tex.get("tree_textures", [])
    resources.grasses_textures = tex.get("grasses_textures", [])
    resources.rock_textures = tex.get("rock_textures", [])
    resources.fence_textures = tex.get("fence_textures", [])
    resources.item_textures = tex.get("item_textures", {})
    resources.equipment_slot_textures = tex.get("equipment_slot_textures", {})
    resources.wall_tex = tex.get("wall_tex")
    resources.torch_tex = tex.get("torch_tex")
    resources.door_tex = tex.get("door_tex")
    resources.window_tex = tex.get("window_tex")
    resources.goblin_tex = tex.get("goblin_tex", {})

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
    resources.sky = SkyRenderer(
        sun_texture_path=STAR_TEXTURE_PATH,
        moon_texture_path=MOON_TEXTURE_PATH,
    )
    ui_state.hud = WorldHUD(scene)
    ui_state.battle_cards = BattleCardLoadout(scene)
    ui_state.battle_overlay = BattleResourceOverlay(scene)
    ui_state.battle_menu = BattleMenu(scene)
    ui_state.pause_menu = PauseMenu(scene)
    ui_state.fov = FOV
    ui_state.fog_enabled = True
    ui_state.fog_density = FOGDENSITY
    ui_state.clouds_enabled = CLOUDS_ENABLED
    ui_state.cloud_density = CLOUD_DENSITY
    ui_state.cloud_speed = CLOUD_SPEED
    ui_state.cloud_opacity = CLOUD_OPACITY
    ui_state.vibrance = 1.15
    ui_state.hud_visible = True
    ui_state.compass_visible = True
    ui_state.minimap_visible = True
    ui_state.held_item_visible = True
    ui_state.test_light_visible = True
    ui_state.controls_text_visible = True
    ui_state.debug_text_visible = ui_state.controls_text_visible
    ui_state.mouse_sensitivity = MOUSE_SENSITIVITY
    ui_state.walk_speed = BASE_SPEED
    ui_state.sprint_speed = SPRINT_SPEED
    ui_state.road_speed_multiplier = 1.5
    ui_state.jump_speed = JUMP_SPEED
    ui_state.gravity = GRAVITY
    ui_state.camera_follow_smooth_hz = CAMERA_FOLLOW_SMOOTH_HZ
    ui_state.setting_menu = SettingMenu(scene)
    ui_state.setting_menu.apply_saved_settings(scene)
    scene.goblin_battle_trigger_distance = GOBLIN_BATTLE_TRIGGER_DISTANCE
    scene.goblin_battle_look_smooth_hz = GOBLIN_BATTLE_LOOK_SMOOTH_HZ
    ui_state.battle_mode = False
    ui_state.active_battle_goblin = None
    ui_state.paused = False
    ui_state.inventory_open = False
    ui_state.inventory_selected_slot = None
    ui_state.inventory_drag_source = None
    ui_state.showing_settings_menu = False
    ui_state.last_mouse_pos = (0, 0)
    scene.log_timing("Initializing sky and HUD", start_time, time.perf_counter())
