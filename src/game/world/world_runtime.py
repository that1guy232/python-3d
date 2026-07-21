"""Runtime helpers for WorldScene.

Camera movement, pause/inventory input, height lookup, and a few geometry
queries live here so the scene class can stay focused on ownership and wiring.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
import pygame

from pygame.math import Vector3

from game.config import (
    BATTLE_LOOK_SMOOTH_HZ,
    BATTLE_TRIGGER_DISTANCE,
    CAMERA_FOLLOW_SMOOTH_HZ,
    CAMERA_GROUND_OFFSET,
    FOV,
    GRAVITY,
    HEIGHT,
    PLAYER_HEAD_CLEARANCE,
    PLAYER_RADIUS,
    WIDTH,
)
from engine.entity import Entity
from engine.collision import (
    player_support_height_at,
    resolve_player_vertical_collision,
)
from engine.rendering.lighting import INDOOR_LIGHT_FACTOR, covered_region_factor_at
from engine.sound.sound_utils import Sounds
from game.world.environment import environment_factor_at
from game.actors.creature import is_combat_creature

_BASE_ENTITY_UPDATE = Entity.update
_AMBIENT_BIRDS_KEY = "ambient_birds"
_AMBIENT_BIRDS_OUTDOOR_VOLUME = 0.035
_AMBIENT_BIRDS_INDOOR_VOLUME = 0.004
_DOOR_INTERACTION_FOCUS_PADDING = 4.0
_DOOR_INTERACTION_DEPTH_PADDING = 8.0
_MAX_PLAYER_VERTICAL_STEP = 1.0 / 60.0


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
        environment_volumes = getattr(scene, "environment_volumes", ()) or ()
        if environment_volumes:
            region_factor = environment_factor_at(
                camera.position.x,
                camera.position.z,
                volumes=environment_volumes,
            )
        else:
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
    return (
        _AMBIENT_BIRDS_INDOOR_VOLUME
        + (_AMBIENT_BIRDS_OUTDOOR_VOLUME - _AMBIENT_BIRDS_INDOOR_VOLUME) * outside_mix
    )


def _update_ambient_birds(scene) -> None:
    volume = _ambient_birds_volume(scene)
    if Sounds.is_playing(_AMBIENT_BIRDS_KEY):
        Sounds.set_playing_volume(_AMBIENT_BIRDS_KEY, volume)
    else:
        Sounds.play(_AMBIENT_BIRDS_KEY, volume=volume, loops=-1, fade_ms=250)


def stop_ambient_birds() -> None:
    """Stop the world-owned ambient bird loop."""
    Sounds.stop(_AMBIENT_BIRDS_KEY)


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


def _integrate_vertical_motion(
    position_y: float,
    velocity_y: float,
    gravity: float,
    dt: float,
) -> tuple[float, float]:
    """Integrate constant downward acceleration without frame-rate drift."""
    step = max(0.0, float(dt))
    gravity = float(gravity)
    velocity_y = float(velocity_y)
    position_y = float(position_y)
    return (
        position_y + velocity_y * step - 0.5 * gravity * step * step,
        velocity_y - gravity * step,
    )


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
    resources = getattr(scene, "render_resources", scene)
    sprites = getattr(resources, "sprite_items", ()) or ()
    if not sprites:
        return

    for update_sprite in _sprite_update_callables(scene, sprites):
        update_sprite(dt)


def _sprite_update_callables(scene, sprites) -> tuple:
    resources = getattr(scene, "render_resources", scene)
    sprite_count = len(sprites)
    first = sprites[0] if sprite_count else None
    last = sprites[-1] if sprite_count else None
    cache = getattr(
        resources,
        "sprite_update_cache",
        getattr(resources, "_sprite_update_cache", None),
    )
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
    cache_value = {
        "sprites": sprites,
        "count": sprite_count,
        "first": first,
        "last": last,
        "updates": updates,
    }
    if hasattr(resources, "sprite_update_cache"):
        resources.sprite_update_cache = cache_value
    else:
        resources._sprite_update_cache = cache_value
    return updates


def _combat_creatures(scene) -> tuple:
    creatures = getattr(scene, "creatures", None)
    if creatures is None:
        creatures = tuple(
            entity
            for entity in (getattr(scene, "entities", ()) or ())
            if is_combat_creature(entity)
        )
    return tuple(creatures or ())


def _battle_candidate(scene):
    camera = getattr(scene, "camera", None)
    player_position = getattr(camera, "position", None)
    if player_position is None:
        return None

    default_trigger_distance = max(
        0.0,
        float(
            getattr(
                scene,
                "battle_trigger_distance",
                getattr(
                    scene,
                    "goblin_battle_trigger_distance",
                    BATTLE_TRIGGER_DISTANCE,
                ),
            )
        ),
    )

    best_creature = None
    best_distance_sq = math.inf
    for creature in _combat_creatures(scene):
        if not is_combat_creature(creature) or not getattr(creature, "enabled", True):
            continue
        position = getattr(creature, "position", None)
        if position is None:
            continue

        creature_trigger = getattr(creature, "battle_trigger_distance", None)
        trigger_distance = (
            default_trigger_distance
            if creature_trigger is None
            else max(0.0, float(creature_trigger))
        )
        trigger_distance_sq = trigger_distance * trigger_distance

        dx = float(position.x) - float(player_position.x)
        dz = float(position.z) - float(player_position.z)
        distance_sq = dx * dx + dz * dz
        if distance_sq <= trigger_distance_sq and distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_creature = creature

    return best_creature


def _start_battle_if_creature_close(scene) -> bool:
    if getattr(scene, "battle_mode", False):
        return True
    creature = _battle_candidate(scene)
    if creature is None:
        return False

    start_battle = getattr(scene, "start_battle", None)
    if not callable(start_battle):
        return False
    return bool(start_battle(creature))


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
                "battle_look_smooth_hz",
                getattr(
                    scene,
                    "goblin_battle_look_smooth_hz",
                    BATTLE_LOOK_SMOOTH_HZ,
                ),
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


def _normalized_vector(value) -> Vector3 | None:
    try:
        vector = Vector3(float(value.x), float(value.y), float(value.z))
    except Exception:
        return None
    if vector.length_squared() <= 1e-8:
        return None
    return vector.normalize()


def _ray_oriented_box_hit_distance(
    origin: Vector3,
    direction: Vector3,
    center: Vector3,
    axes: tuple[Vector3, Vector3, Vector3],
    half_extents: tuple[float, float, float],
    max_distance: float,
) -> float | None:
    local_origin = origin - center
    t_min = 0.0
    t_max = float(max_distance)

    for axis, half_extent in zip(axes, half_extents):
        axis_n = _normalized_vector(axis)
        if axis_n is None:
            return None
        origin_component = local_origin.dot(axis_n)
        direction_component = direction.dot(axis_n)
        half_extent = max(0.0, float(half_extent))

        if abs(direction_component) <= 1e-8:
            if abs(origin_component) > half_extent:
                return None
            continue

        inv_direction = 1.0 / direction_component
        near_t = (-half_extent - origin_component) * inv_direction
        far_t = (half_extent - origin_component) * inv_direction
        if near_t > far_t:
            near_t, far_t = far_t, near_t

        t_min = max(t_min, near_t)
        t_max = min(t_max, far_t)
        if t_min > t_max:
            return None

    return t_min if t_max >= 0.0 else None


def _is_door_like(entity) -> bool:
    return (
        type(entity).__name__ == "Door"
        or getattr(entity, "door_render_batched", False)
        or hasattr(entity, "target_open")
        and hasattr(entity, "panel_axis")
        and hasattr(entity, "depth_axis")
    )


def _door_focus_hit_distance(entity, camera, max_distance: float) -> float | None:
    origin = getattr(camera, "position", None)
    direction = _normalized_vector(getattr(camera, "_forward", None))
    center = getattr(entity, "position", None)
    if origin is None or direction is None or center is None:
        return None

    width_axis = getattr(entity, "panel_axis", None)
    depth_axis = getattr(entity, "depth_axis", None)
    if width_axis is None or depth_axis is None:
        return None

    padding = _DOOR_INTERACTION_FOCUS_PADDING
    half_width = max(0.5, float(getattr(entity, "width", 1.0)) * 0.5 + padding)
    half_height = max(0.5, float(getattr(entity, "height", 1.0)) * 0.5 + padding)
    half_depth = max(
        0.5,
        float(getattr(entity, "thickness", 1.0)) * 0.5
        + _DOOR_INTERACTION_DEPTH_PADDING,
    )

    return _ray_oriented_box_hit_distance(
        origin,
        direction,
        center,
        (width_axis, Vector3(0.0, 1.0, 0.0), depth_axis),
        (half_width, half_height, half_depth),
        max_distance,
    )


def _focused_interactable_entity(scene):
    camera = getattr(scene, "camera", None)
    if camera is None:
        return None

    forward = getattr(camera, "_forward", None)
    camera_position = getattr(camera, "position", None)
    if forward is None or camera_position is None:
        return None

    forward_xz = Vector3(float(forward.x), 0.0, float(forward.z))
    if forward_xz.length_squared() <= 1e-8:
        return None
    forward_xz = forward_xz.normalize()

    best_focused_door = None
    best_focused_door_distance = float("inf")
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

        if _is_door_like(entity):
            hit_distance = _door_focus_hit_distance(entity, camera, max_distance)
            if hit_distance is None:
                continue
            if hit_distance < best_focused_door_distance:
                best_focused_door_distance = hit_distance
                best_focused_door = entity
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

    return best_focused_door if best_focused_door is not None else best_entity


def focused_interaction_prompt(scene) -> str | None:
    entity = _focused_interactable_entity(scene)
    if entity is None or not _is_door_like(entity):
        return None
    action = "close" if bool(getattr(entity, "target_open", False)) else "open"
    return f"E to {action} door"


def _try_interact_with_focused_entity(scene) -> bool:
    best_entity = _focused_interactable_entity(scene)
    if best_entity is None:
        return False

    camera = getattr(scene, "camera", None)
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
    candidates = scene.collision_meshes_at(
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
    ui = getattr(scene, "ui_state", scene)
    hud = getattr(ui, "hud", getattr(scene, "_hud", None))
    if getattr(ui, "battle_mode", False):
        with _profile(scene, "update.battle_camera"):
            _update_battle_camera(scene, dt)
        with _profile(scene, "update.battle_hud"):
            try:
                hud.update(dt)
            except Exception:
                pass
        return

    if getattr(ui, "paused", False) or getattr(ui, "inventory_open", False):
        with _profile(scene, "update.paused_hud"):
            try:
                hud.update(dt)
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
        if _start_battle_if_creature_close(scene):
            _update_battle_camera(scene, dt)
            try:
                hud.update(dt)
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
            gravity = float(getattr(scene, "gravity", GRAVITY))
            player_radius = _player_radius(scene)
            vertical_candidates = scene.collision_meshes_at(
                scene.camera.position.x,
                scene.camera.position.z,
                player_radius,
                include_polygons=False,
            )
            remaining = max(0.0, float(dt))
            while remaining > 1e-9 and scene.camera.is_jumping:
                step = min(remaining, _MAX_PLAYER_VERTICAL_STEP)
                old_vertical_position = scene.camera.position.copy()
                (
                    scene.camera.position.y,
                    scene.camera.vertical_velocity,
                ) = _integrate_vertical_motion(
                    scene.camera.position.y,
                    scene.camera.vertical_velocity,
                    gravity,
                    step,
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
                remaining -= step

            foot_y = scene.camera.position.y - foot_offset
            support_y_here = _support_height_at(
                scene,
                scene.camera.position.x,
                scene.camera.position.z,
                foot_y,
            )
            target_cam_y = support_y_here + foot_offset
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
        hud.update(dt)
        scene._headbob.update(moving=moving, sprinting=sprinting, dt=dt)


def handle_event(scene, event) -> None:
    ui = getattr(scene, "ui_state", scene)

    def remember_mouse(pos) -> None:
        if ui is scene:
            scene._last_mouse_pos = pos
        else:
            ui.last_mouse_pos = pos

    if getattr(ui, "battle_mode", False):
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) == 1:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            scene._handle_battle_click(pos)
            return
        if event.type == pygame.MOUSEBUTTONUP and getattr(event, "button", None) == 1:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            scene._handle_battle_release(pos)
            return
        if event.type == pygame.MOUSEMOTION:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            remember_mouse(pos)
            scene._handle_battle_motion(pos)
            return
        return

    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        if getattr(ui, "inventory_open", False):
            ui.inventory_selected_slot = None
            ui.inventory_drag_source = None
            ui.inventory_open = False
            ui.paused = False
            ui.showing_settings_menu = False
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        else:
            ui.paused = not getattr(ui, "paused", False)
            if not ui.paused:
                ui.showing_settings_menu = False
            pygame.mouse.set_visible(ui.paused)
            pygame.event.set_grab(not ui.paused)
        return

    if event.type == pygame.KEYDOWN and event.key in (pygame.K_i, pygame.K_TAB):
        ui.inventory_open = not getattr(ui, "inventory_open", False)
        if not ui.inventory_open:
            ui.inventory_selected_slot = None
            ui.inventory_drag_source = None
        ui.paused = ui.inventory_open
        ui.showing_settings_menu = False
        pygame.mouse.set_visible(ui.paused)
        pygame.event.set_grab(not ui.paused)
        return

    if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
        ui.minimap_visible = not getattr(ui, "minimap_visible", True)
        return

    if getattr(ui, "inventory_open", False):
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) == 1:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            scene._handle_inventory_click(pos)
            return
        if event.type == pygame.MOUSEBUTTONUP and getattr(event, "button", None) == 1:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            scene._handle_inventory_release(pos)
            return
        if event.type == pygame.MOUSEMOTION:
            remember_mouse(getattr(event, "pos", pygame.mouse.get_pos()))
            return
        return

    if getattr(ui, "paused", False) and not getattr(ui, "inventory_open", False):
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
            remember_mouse(pos)
            scene._handle_pause_motion(pos)
            return
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
    ui = getattr(scene, "ui_state", scene)
    if (
        getattr(ui, "battle_mode", False)
        or getattr(ui, "paused", False)
        or getattr(ui, "inventory_open", False)
    ):
        return
    try:
        try:
            scene._camera_controller.on_mouse_delta(dx, dy, dt)
        except TypeError:
            scene._camera_controller.on_mouse_delta(dx, dy)
    except Exception:
        pass
