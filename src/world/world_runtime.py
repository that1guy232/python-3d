"""Runtime helpers for WorldScene.

Camera movement, pause/inventory input, height lookup, and a few geometry
queries live here so the scene class can stay focused on ownership and wiring.
"""

from __future__ import annotations

import math
import pygame

from pygame.math import Vector3

from config import *
from sound.sound_utils import Sounds
from world.world_collision import (
    player_support_height_at,
    resolve_player_vertical_collision,
)


def contains_horizontal(scene, pos: Vector3) -> bool:
    min_x, max_x, min_z, max_z = scene.ground_bounds
    extra = -15.0
    return (min_x - extra <= pos.x <= max_x + extra) and (
        min_z - extra <= pos.z <= max_z + extra
    )


def is_on_road(scene, x: float, z: float, *, margin: float = 0.0) -> bool:
    roads = getattr(scene, "roads", None)
    if roads is None:
        road = getattr(scene, "road", None)
        roads = [road] if road is not None else []
    return any(r.contains_point(x, z, margin=margin) for r in roads if r is not None)


def ground_height_at(scene, x: float, z: float) -> float:
    sampler = getattr(scene, "_ground_height_sampler", None)
    if sampler is not None and hasattr(sampler, "height_at"):
        try:
            return float(sampler.height_at(x, z))
        except Exception:
            pass
    fn = getattr(scene, "_height_fn", None)
    return float(fn(x, z)) if callable(fn) else 5.0


def _player_radius(scene) -> float:
    return float(getattr(scene, "player_radius", PLAYER_RADIUS))


def _player_foot_offset(scene) -> float:
    manual_offset = getattr(scene.camera, "manual_height_offset", 0.0)
    return CAMERA_GROUND_OFFSET + float(manual_offset)


def _player_head_offset(scene) -> float:
    return float(getattr(scene, "player_head_clearance", PLAYER_HEAD_CLEARANCE))


def _wall_collision_meshes(scene):
    return getattr(scene, "wall_tiles", None) or []


def _support_height_at(scene, x: float, z: float, foot_y: float | None = None) -> float:
    ground_y = scene.ground_height_at(x, z)
    if foot_y is None:
        foot_y = scene.camera.position.y - _player_foot_offset(scene)
    wall_y = player_support_height_at(
        _wall_collision_meshes(scene),
        x,
        z,
        foot_y,
        _player_radius(scene),
    )
    if wall_y is None:
        return ground_y
    return max(float(ground_y), float(wall_y))


def view_space_position(
    scene, *, dist: float, nx: float, ny: float, px: float = 0.0, py: float = 0.0
) -> Vector3:
    aspect = WIDTH / HEIGHT
    fov = float(getattr(scene, "fov", FOV))
    half_h = dist * math.tan(math.radians(fov * 0.5))
    half_w = half_h * aspect
    wu_per_px_x = (2.0 * half_w) / WIDTH
    wu_per_px_y = (2.0 * half_h) / HEIGHT

    right = scene.camera._right
    forward = scene.camera._forward
    up = getattr(scene.camera, "_up", right.cross(forward))

    center = scene.camera.position + forward * dist
    off_right = (nx * half_w) + (px * wu_per_px_x)
    off_up = (ny * half_h) - (py * wu_per_px_y)
    return center + (right * off_right) + (up * off_up)


def update(scene, dt: float) -> None:
    if getattr(scene, "paused", False) or getattr(scene, "inventory_open", False):
        try:
            scene._hud.update(dt)
        except Exception:
            pass
        return

    if not Sounds.is_playing("ambient_birds"):
        Sounds.play("ambient_birds", volume=0.05)

    moving, sprinting = scene._camera_controller.update(dt)
    scene._sway_controller.update(dt)

    foot_offset = _player_foot_offset(scene)
    head_offset = _player_head_offset(scene)
    foot_y = scene.camera.position.y - foot_offset
    support_y_here = _support_height_at(
        scene,
        scene.camera.position.x,
        scene.camera.position.z,
        foot_y,
    )

    target_cam_y = support_y_here + foot_offset

    if scene.camera.is_jumping:
        old_vertical_position = scene.camera.position.copy()
        scene.camera.vertical_velocity -= float(getattr(scene, "gravity", GRAVITY)) * dt
        scene.camera.position.y += scene.camera.vertical_velocity * dt

        vertical_hit = resolve_player_vertical_collision(
            _wall_collision_meshes(scene),
            old_vertical_position,
            scene.camera.position,
            foot_offset=foot_offset,
            head_offset=head_offset,
            player_radius=_player_radius(scene),
        )
        if vertical_hit is not None:
            scene.camera.position.y = vertical_hit.camera_y
            if vertical_hit.kind == "floor":
                scene.camera.vertical_velocity = 0.0
                scene.camera.is_jumping = False
            elif scene.camera.vertical_velocity > 0.0:
                scene.camera.vertical_velocity = 0.0

        foot_y = scene.camera.position.y - foot_offset
        support_y_here = _support_height_at(
            scene,
            scene.camera.position.x,
            scene.camera.position.z,
            foot_y,
        )
        target_cam_y = support_y_here + foot_offset
        if scene.camera.position.y <= target_cam_y:
            scene.camera.position.y = target_cam_y
            scene.camera.vertical_velocity = 0.0
            scene.camera.is_jumping = False
    else:
        smooth_hz = float(
            getattr(scene, "camera_follow_smooth_hz", CAMERA_FOLLOW_SMOOTH_HZ)
        )
        if smooth_hz <= 0 or dt <= 0:
            scene.camera.position.y = target_cam_y
        else:
            a = 1.0 - math.exp(-smooth_hz * dt)
            scene.camera.position.y += (target_cam_y - scene.camera.position.y) * a

    scene._hud.update(dt)
    scene._headbob.update(moving=moving, sprinting=sprinting, dt=dt)


def handle_event(scene, event) -> None:

    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        if getattr(scene, "inventory_open", False):
            scene.inventory_open = False
            scene.paused = False
            scene.showing_settings_menu = False
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        else:
            scene.paused = not getattr(scene, "paused", False)
            if not scene.paused:
                scene.showing_settings_menu = False
            pygame.mouse.set_visible(scene.paused)
            pygame.event.set_grab(not scene.paused)
        return

    if event.type == pygame.KEYDOWN and event.key in (pygame.K_i, pygame.K_TAB):
        scene.inventory_open = not getattr(scene, "inventory_open", False)
        scene.paused = scene.inventory_open
        scene.showing_settings_menu = False
        pygame.mouse.set_visible(scene.paused)
        pygame.event.set_grab(not scene.paused)
        return

    if getattr(scene, "paused", False) and not getattr(scene, "inventory_open", False):
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) == 1:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            scene._handle_pause_click(pos)
            return
        if event.type == pygame.MOUSEBUTTONUP and getattr(event, "button", None) == 1:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            scene._handle_pause_release(pos)
            return
        if event.type == pygame.MOUSEMOTION:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            scene._last_mouse_pos = pos
            scene._handle_pause_motion(pos)
            return
        return

    if getattr(scene, "inventory_open", False):
        return

    try:
        if hasattr(scene, "_camera_controller") and hasattr(
            scene._camera_controller, "handle_event"
        ):
            scene._camera_controller.handle_event(event)
    except Exception:
        pass


def apply_mouse_delta(scene, dx: float, dy: float, dt: float | None = None) -> None:
    if getattr(scene, "paused", False) or getattr(scene, "inventory_open", False):
        return
    try:
        try:
            scene._camera_controller.on_mouse_delta(dx, dy, dt)
        except TypeError:
            scene._camera_controller.on_mouse_delta(dx, dy)
    except Exception:
        pass
