"""Runtime helpers for WorldScene.

Camera movement, pause/inventory input, height lookup, and a few geometry
queries live here so the scene class can stay focused on ownership and wiring.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
import pygame

from pygame.math import Vector3

from game.config import *
from engine.entity import Entity
from engine.collision import (
    player_support_height_at,
    resolve_player_vertical_collision,
)
from engine.rendering.lighting import INDOOR_LIGHT_FACTOR, covered_region_factor_at
from engine.sound.sound_utils import Sounds


_BASE_ENTITY_UPDATE = Entity.update
_COLLISION_CELL_SIZE = 128.0
_COLLISION_FALLBACK_CELL_LIMIT = 256
_AMBIENT_BIRDS_KEY = "ambient_birds"
_AMBIENT_BIRDS_OUTDOOR_VOLUME = 0.035
_AMBIENT_BIRDS_INDOOR_VOLUME = 0.004


def _profile(scene, name: str):
    profiler = getattr(scene, "profiler", None)
    if profiler is None or not getattr(profiler, "enabled", False):
        return nullcontext()
    return profiler.section(name)


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


def _ambient_birds_volume(scene) -> float:
    camera = getattr(scene, "camera", None)
    if camera is None:
        return _AMBIENT_BIRDS_OUTDOOR_VOLUME

    try:
        region_factor = covered_region_factor_at(
            camera.position.x,
            camera.position.z,
            covered_regions=getattr(scene, "covered_regions", ()),
        )
    except (TypeError, ValueError, AttributeError):
        region_factor = 1.0

    indoor_factor = max(0.0, min(0.999, float(INDOOR_LIGHT_FACTOR)))
    outside_mix = (max(0.0, min(1.0, region_factor)) - indoor_factor) / (
        1.0 - indoor_factor
    )
    outside_mix = max(0.0, min(1.0, outside_mix))
    return _AMBIENT_BIRDS_INDOOR_VOLUME + (
        _AMBIENT_BIRDS_OUTDOOR_VOLUME - _AMBIENT_BIRDS_INDOOR_VOLUME
    ) * outside_mix


def _update_ambient_birds(scene) -> None:
    volume = _ambient_birds_volume(scene)
    if Sounds.is_playing(_AMBIENT_BIRDS_KEY):
        Sounds.set_playing_volume(_AMBIENT_BIRDS_KEY, volume)
    else:
        Sounds.play(_AMBIENT_BIRDS_KEY, volume=volume, loops=-1, fade_ms=250)


def stop_ambient_birds() -> None:
    """Stop the world-owned ambient bird loop."""
    Sounds.stop(_AMBIENT_BIRDS_KEY)


def _iter_collision_sources(scene):
    seen: set[int] = set()
    for mesh in getattr(scene, "wall_tiles", ()) or ():
        if mesh is None:
            continue
        mesh_id = id(mesh)
        if mesh_id in seen:
            continue
        seen.add(mesh_id)
        yield mesh, False
    for mesh in getattr(scene, "polygons", ()) or ():
        if mesh is None:
            continue
        mesh_id = id(mesh)
        if mesh_id in seen:
            continue
        seen.add(mesh_id)
        yield mesh, True


def _collision_source_key(scene):
    wall_tiles = getattr(scene, "wall_tiles", ()) or ()
    polygons = getattr(scene, "polygons", ()) or ()

    def edge_ids(values):
        if not values:
            return (None, None)
        return (id(values[0]), id(values[-1]))

    return (
        id(wall_tiles),
        len(wall_tiles),
        *edge_ids(wall_tiles),
        id(polygons),
        len(polygons),
        *edge_ids(polygons),
    )


def _collision_mesh_dynamic(mesh) -> bool:
    return (
        getattr(mesh, "door_render_batched", False)
        or hasattr(mesh, "open_amount")
        or type(mesh).__name__ == "Door"
    )


def _mesh_bounds(mesh):
    get_bounds = getattr(mesh, "get_bounding_box", None)
    if not callable(get_bounds):
        return None
    try:
        bounds = get_bounds()
    except Exception:
        return None
    if not bounds:
        return None
    try:
        min_x, max_x, min_z, max_z = bounds
        return (float(min_x), float(max_x), float(min_z), float(max_z))
    except Exception:
        return None


def rebuild_collision_index(scene) -> dict:
    cell_size = float(getattr(scene, "collision_cell_size", _COLLISION_CELL_SIZE))
    if cell_size <= 1.0:
        cell_size = _COLLISION_CELL_SIZE

    cells: dict[tuple[int, int], list] = {}
    wall_cells: dict[tuple[int, int], list] = {}
    dynamic = []
    wall_dynamic = []
    fallback = []
    wall_fallback = []

    def add_to_grid(grid, mesh, bounds) -> bool:
        min_x, max_x, min_z, max_z = bounds
        min_cx = math.floor(min_x / cell_size)
        max_cx = math.floor(max_x / cell_size)
        min_cz = math.floor(min_z / cell_size)
        max_cz = math.floor(max_z / cell_size)
        cell_count = (max_cx - min_cx + 1) * (max_cz - min_cz + 1)
        if cell_count > _COLLISION_FALLBACK_CELL_LIMIT:
            return False
        for cx in range(min_cx, max_cx + 1):
            for cz in range(min_cz, max_cz + 1):
                grid.setdefault((cx, cz), []).append(mesh)
        return True

    for mesh, is_polygon in _iter_collision_sources(scene):
        is_wall = not is_polygon
        if _collision_mesh_dynamic(mesh):
            dynamic.append(mesh)
            if is_wall:
                wall_dynamic.append(mesh)
            continue

        bounds = _mesh_bounds(mesh)
        if bounds is None:
            fallback.append(mesh)
            if is_wall:
                wall_fallback.append(mesh)
            continue

        if not add_to_grid(cells, mesh, bounds):
            fallback.append(mesh)
        if is_wall and not add_to_grid(wall_cells, mesh, bounds):
            wall_fallback.append(mesh)

    index = {
        "key": _collision_source_key(scene),
        "cell_size": cell_size,
        "cells": cells,
        "wall_cells": wall_cells,
        "dynamic": tuple(dynamic),
        "wall_dynamic": tuple(wall_dynamic),
        "fallback": tuple(fallback),
        "wall_fallback": tuple(wall_fallback),
    }
    scene._collision_spatial_index = index
    return index


def invalidate_collision_index(scene) -> None:
    scene._collision_spatial_index = None


def _collision_index(scene) -> dict:
    index = getattr(scene, "_collision_spatial_index", None)
    key = _collision_source_key(scene)
    if not isinstance(index, dict) or index.get("key") != key:
        index = rebuild_collision_index(scene)
    return index


def collision_meshes_for_bounds(
    scene,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    *,
    include_polygons: bool = True,
) -> list:
    index = _collision_index(scene)
    cell_size = float(index.get("cell_size") or _COLLISION_CELL_SIZE)
    cells = index["cells"] if include_polygons else index["wall_cells"]
    dynamic = index["dynamic"] if include_polygons else index["wall_dynamic"]
    fallback = index["fallback"] if include_polygons else index["wall_fallback"]

    min_cx = math.floor(float(min_x) / cell_size)
    max_cx = math.floor(float(max_x) / cell_size)
    min_cz = math.floor(float(min_z) / cell_size)
    max_cz = math.floor(float(max_z) / cell_size)

    candidates = []
    seen: set[int] = set()

    def add(mesh) -> None:
        mesh_id = id(mesh)
        if mesh_id in seen:
            return
        seen.add(mesh_id)
        candidates.append(mesh)

    for cx in range(min_cx, max_cx + 1):
        for cz in range(min_cz, max_cz + 1):
            for mesh in cells.get((cx, cz), ()):
                add(mesh)
    for mesh in dynamic:
        add(mesh)
    for mesh in fallback:
        add(mesh)

    return candidates


def collision_meshes_at(
    scene,
    x: float,
    z: float,
    radius: float,
    *,
    include_polygons: bool = True,
) -> list:
    x = float(x)
    z = float(z)
    radius = max(0.0, float(radius))
    return collision_meshes_for_bounds(
        scene,
        x - radius,
        x + radius,
        z - radius,
        z + radius,
        include_polygons=include_polygons,
    )


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


def _update_entities(scene, dt: float) -> None:
    for entity in getattr(scene, "entities", ()) or ():
        if not getattr(entity, "enabled", True):
            continue
        if getattr(entity, "runtime_update_enabled", True) is False:
            continue
        if getattr(type(entity), "update", None) is _BASE_ENTITY_UPDATE:
            continue
        update_entity = getattr(entity, "update", None)
        if callable(update_entity):
            update_entity(dt)


def _update_sprites(scene, dt: float) -> None:
    sprites = getattr(scene, "sprite_items", ()) or ()
    if not sprites:
        return

    for update_sprite in _sprite_update_callables(scene, sprites):
        update_sprite(dt)


def _sprite_update_callables(scene, sprites) -> tuple:
    sprite_count = len(sprites)
    first = sprites[0] if sprite_count else None
    last = sprites[-1] if sprite_count else None
    cache = getattr(scene, "_sprite_update_cache", None)
    if (
        cache is not None
        and cache.get("sprites") is sprites
        and cache.get("count") == sprite_count
        and cache.get("first") is first
        and cache.get("last") is last
    ):
        return cache["updates"]

    updates = []
    for sprite in sprites:
        update_sprite = getattr(sprite, "update", None)
        if callable(update_sprite):
            updates.append(update_sprite)

    updates = tuple(updates)
    scene._sprite_update_cache = {
        "sprites": sprites,
        "count": sprite_count,
        "first": first,
        "last": last,
        "updates": updates,
    }
    return updates


def _goblin_battle_candidate(scene):
    camera = getattr(scene, "camera", None)
    player_position = getattr(camera, "position", None)
    if player_position is None:
        return None

    trigger_distance = max(
        0.0,
        float(
            getattr(
                scene,
                "goblin_battle_trigger_distance",
                GOBLIN_BATTLE_TRIGGER_DISTANCE,
            )
        ),
    )
    trigger_distance_sq = trigger_distance * trigger_distance

    best_goblin = None
    best_distance_sq = trigger_distance_sq
    for goblin in getattr(scene, "goblins", ()) or ():
        if not getattr(goblin, "enabled", True):
            continue
        position = getattr(goblin, "position", None)
        if position is None:
            continue

        dx = float(position.x) - float(player_position.x)
        dz = float(position.z) - float(player_position.z)
        distance_sq = dx * dx + dz * dz
        if distance_sq <= best_distance_sq:
            best_distance_sq = distance_sq
            best_goblin = goblin

    return best_goblin


def _start_battle_if_goblin_close(scene) -> bool:
    if getattr(scene, "battle_mode", False):
        return True
    goblin = _goblin_battle_candidate(scene)
    if goblin is None:
        return False

    start_battle = getattr(scene, "start_battle", None)
    if not callable(start_battle):
        return False
    return bool(start_battle(goblin))


def _update_battle_camera(scene, dt: float) -> None:
    controller = getattr(scene, "_camera_controller", None)
    update_rotation = getattr(controller, "update_rotation_only", None)
    if not callable(update_rotation):
        return

    smooth_hz = max(
        0.001,
        float(
            getattr(
                scene,
                "goblin_battle_look_smooth_hz",
                GOBLIN_BATTLE_LOOK_SMOOTH_HZ,
            )
        ),
    )
    try:
        update_rotation(dt, smooth_hz=smooth_hz)
    except TypeError:
        update_rotation(dt)


def _entity_interaction_position(entity):
    get_position = getattr(entity, "get_interaction_position", None)
    if callable(get_position):
        try:
            return get_position()
        except Exception:
            return None
    return getattr(entity, "position", None)


def _try_interact_with_focused_entity(scene) -> bool:
    camera = getattr(scene, "camera", None)
    if camera is None:
        return False

    forward = getattr(camera, "_forward", None)
    camera_position = getattr(camera, "position", None)
    if forward is None or camera_position is None:
        return False

    forward_xz = Vector3(float(forward.x), 0.0, float(forward.z))
    if forward_xz.length_squared() <= 1e-8:
        return False
    forward_xz = forward_xz.normalize()

    best_entity = None
    best_score = float("inf")
    for entity in getattr(scene, "entities", ()) or ():
        if not getattr(entity, "enabled", True):
            continue
        interact = getattr(entity, "interact", None)
        if not callable(interact):
            continue

        max_distance = float(getattr(entity, "interaction_distance", 0.0) or 0.0)
        if max_distance <= 0.0:
            continue

        position = _entity_interaction_position(entity)
        if position is None:
            continue

        delta = Vector3(
            float(position.x) - float(camera_position.x),
            0.0,
            float(position.z) - float(camera_position.z),
        )
        distance_sq = delta.length_squared()
        if distance_sq > max_distance * max_distance:
            continue

        distance = math.sqrt(distance_sq)
        facing = 1.0 if distance <= 1e-6 else forward_xz.dot(delta / distance)
        if facing < -0.15:
            continue

        score = distance - facing * 20.0
        if score < best_score:
            best_score = score
            best_entity = entity

    if best_entity is None:
        return False

    interact = getattr(best_entity, "interact")
    try:
        return bool(interact(actor=camera, scene=scene))
    except TypeError:
        try:
            return bool(interact(camera))
        except TypeError:
            return bool(interact())


def _support_height_at(scene, x: float, z: float, foot_y: float | None = None) -> float:
    ground_y = scene.ground_height_at(x, z)
    if foot_y is None:
        foot_y = scene.camera.position.y - _player_foot_offset(scene)
    player_radius = _player_radius(scene)
    candidates = collision_meshes_at(
        scene,
        x,
        z,
        player_radius,
        include_polygons=False,
    )
    wall_y = player_support_height_at(
        candidates,
        x,
        z,
        foot_y,
        player_radius,
    )
    if wall_y is None:
        return ground_y
    return max(float(ground_y), float(wall_y))


def initialize_player_spawn_height(scene) -> None:
    """Apply the configured starting Y once the terrain height is known."""
    if getattr(scene, "_player_spawn_height_initialized", False):
        return

    camera = getattr(scene, "camera", None)
    if camera is None or getattr(camera, "position", None) is None:
        scene._player_spawn_height_initialized = True
        return

    foot_offset = _player_foot_offset(scene)
    foot_y = camera.position.y - foot_offset
    support_y = _support_height_at(
        scene,
        camera.position.x,
        camera.position.z,
        foot_y,
    )
    target_cam_y = support_y + foot_offset

    if camera.position.y < target_cam_y:
        camera.position.y = target_cam_y
        camera.vertical_velocity = 0.0
        camera.is_jumping = False
    elif camera.position.y > target_cam_y + 0.1:
        camera.is_jumping = True
        if not hasattr(camera, "vertical_velocity"):
            camera.vertical_velocity = 0.0

    scene._player_spawn_height_initialized = True


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
    if getattr(scene, "battle_mode", False):
        with _profile(scene, "update.battle_camera"):
            _update_battle_camera(scene, dt)
        with _profile(scene, "update.battle_hud"):
            try:
                scene._hud.update(dt)
            except Exception:
                pass
        return

    if getattr(scene, "paused", False) or getattr(scene, "inventory_open", False):
        with _profile(scene, "update.paused_hud"):
            try:
                scene._hud.update(dt)
            except Exception:
                pass
        return

    with _profile(scene, "update.audio"):
        _update_ambient_birds(scene)

    with _profile(scene, "update.player_spawn_height"):
        initialize_player_spawn_height(scene)
    with _profile(scene, "update.camera_controller"):
        moving, sprinting = scene._camera_controller.update(dt)
    with _profile(scene, "update.sway"):
        scene._sway_controller.update(dt)
    with _profile(scene, "update.entities"):
        _update_entities(scene, dt)
        if _start_battle_if_goblin_close(scene):
            _update_battle_camera(scene, dt)
            try:
                scene._hud.update(dt)
            except Exception:
                pass
            return
    with _profile(scene, "update.sprites"):
        _update_sprites(scene, dt)

    with _profile(scene, "update.player_height"):
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
            player_radius = _player_radius(scene)
            vertical_candidates = collision_meshes_at(
                scene,
                scene.camera.position.x,
                scene.camera.position.z,
                player_radius,
                include_polygons=False,
            )

            vertical_hit = resolve_player_vertical_collision(
                vertical_candidates,
                old_vertical_position,
                scene.camera.position,
                foot_offset=foot_offset,
                head_offset=head_offset,
                player_radius=player_radius,
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

    with _profile(scene, "update.hud_headbob"):
        scene._hud.update(dt)
        scene._headbob.update(moving=moving, sprinting=sprinting, dt=dt)


def handle_event(scene, event) -> None:
    if getattr(scene, "battle_mode", False):
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) == 1:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            scene._handle_battle_click(pos)
            return
        if event.type == pygame.MOUSEMOTION:
            scene._last_mouse_pos = getattr(event, "pos", pygame.mouse.get_pos())
            return
        return

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

    if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
        scene.minimap_visible = not getattr(scene, "minimap_visible", True)
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

    if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
        if _try_interact_with_focused_entity(scene):
            return

    try:
        if hasattr(scene, "_camera_controller") and hasattr(
            scene._camera_controller, "handle_event"
        ):
            scene._camera_controller.handle_event(event)
    except Exception:
        pass


def apply_mouse_delta(scene, dx: float, dy: float, dt: float | None = None) -> None:
    if (
        getattr(scene, "battle_mode", False)
        or getattr(scene, "paused", False)
        or getattr(scene, "inventory_open", False)
    ):
        return
    try:
        try:
            scene._camera_controller.on_mouse_delta(dx, dy, dt)
        except TypeError:
            scene._camera_controller.on_mouse_delta(dx, dy)
    except Exception:
        pass
