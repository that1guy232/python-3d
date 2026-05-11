"""Roaming goblin entity backed by camera-aware directional billboards."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from pygame.math import Vector3
from OpenGL.GL import (
    GL_ALPHA_TEST,
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_FLOAT,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POLYGON_OFFSET_FILL,
    GL_QUADS,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_COORD_ARRAY,
    GL_TEXTURE_ENV,
    GL_TEXTURE_ENV_MODE,
    GL_MODULATE,
    GL_TRIANGLES,
    GL_VERTEX_ARRAY,
    glBegin,
    glBindBuffer,
    glBindTexture,
    glBlendFunc,
    glColor4f,
    glDepthMask,
    glDisable,
    glDisableClientState,
    glEnable,
    glEnableClientState,
    glEnd,
    glDrawArrays,
    glPolygonOffset,
    glTexCoord2f,
    glTexEnvi,
    glTexCoordPointer,
    glVertexPointer,
    glVertex3f,
)

from engine.entity import Entity
from engine.rendering.sprite import AnimatedWorldSprite
from game.resources.paths import (
    GOBLIN_BACK_TEXTURE_DIR_PATH,
    GOBLIN_FRONT_TEXTURE_DIR_PATH,
    GOBLIN_RIGHT_TEXTURE_DIR_PATH,
)
from engine.textures.texture_utils import get_texture_size, load_texture_atlas


GOBLIN_SPRITE_HEIGHT = 42.0
GOBLIN_ANIMATION_FPS = 8.0
GOBLIN_ANIMATION_FRAME_DURATION = 1.0 / GOBLIN_ANIMATION_FPS
GOBLIN_ROAM_RADIUS = 145.0
GOBLIN_MOVE_SPEED = 32.0
GOBLIN_COLLISION_RADIUS = 18.0
GOBLIN_CHASE_RADIUS = 260.0
GOBLIN_CHASE_GIVE_UP_RADIUS = 380.0
GOBLIN_CHASE_STOP_DISTANCE = 34.0
GOBLIN_TARGET_REACHED_DISTANCE = 7.0
GOBLIN_MIN_IDLE_SECONDS = 0.35
GOBLIN_MAX_IDLE_SECONDS = 1.45
GOBLIN_VIEW_FRONT_DOT = 0.55
GOBLIN_LOGIC_UPDATE_INTERVAL = 1.0 / 30.0
GOBLIN_DIRECTION_UPDATE_INTERVAL = 1.0 / 20.0
GOBLIN_SHADOW_SIZE = (30.0, 22.0)
GOBLIN_SHADOW_ELEVATION = 0.35
GOBLIN_PATH_SAMPLE_SPACING = 12.0
GOBLIN_NAV_MIN_STEP = 0.15
GOBLIN_NAV_STEP_SCALES = (1.0, 0.55, 0.25)
GOBLIN_NAV_STEER_DEGREES = (30.0, 60.0, 90.0, 135.0)

PositionBlockedFn = Callable[[float, float, float], bool]
PlayerInBuildingFn = Callable[[], bool]
AnimationFrames = tuple[Any, ...]
AnimationSets = dict[str, AnimationFrames]


def _shadow_batch_scratch(owner, shadow_count: int) -> dict:
    scratch = getattr(owner, "_goblin_shadow_batch_scratch", None)
    current_capacity = int(scratch.get("shadow_capacity", 0)) if scratch else 0
    if scratch is not None and current_capacity >= shadow_count:
        return scratch

    shadow_capacity = max(64, int(shadow_count), int(current_capacity * 1.5))
    vertex_capacity = shadow_capacity * 6
    scratch = {
        "shadow_capacity": shadow_capacity,
        "vertices": np.empty((vertex_capacity, 3), dtype=np.float32),
        "texcoords": np.empty((vertex_capacity, 2), dtype=np.float32),
    }
    setattr(owner, "_goblin_shadow_batch_scratch", scratch)
    return scratch


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.lower())


def _frame_paths(directory: str) -> list[str]:
    frame_dir = Path(directory)
    if not frame_dir.is_dir():
        return []
    return [str(path) for path in sorted(frame_dir.glob("*.png"), key=_frame_sort_key)]


def _load_frames(directory: str) -> AnimationFrames:
    paths = _frame_paths(directory)
    return tuple(load_texture_atlas(paths)) if paths else ()


class Goblin(Entity):
    """A simple wandering forest creature with camera-aware directional art."""

    DEFAULT_HEIGHT = GOBLIN_SPRITE_HEIGHT
    DEFAULT_ROAM_RADIUS = GOBLIN_ROAM_RADIUS
    DEFAULT_MOVE_SPEED = GOBLIN_MOVE_SPEED

    def __init__(
        self,
        position: Vector3,
        *,
        texture: Any,
        camera: object,
        ground_height_at: Callable[[float, float], float],
        position_blocked: PositionBlockedFn | None = None,
        player_in_building: PlayerInBuildingFn | None = None,
        sprite_height: float = GOBLIN_SPRITE_HEIGHT,
        roam_radius: float = GOBLIN_ROAM_RADIUS,
        move_speed: float = GOBLIN_MOVE_SPEED,
        collision_radius: float = GOBLIN_COLLISION_RADIUS,
        chase_radius: float = GOBLIN_CHASE_RADIUS,
        chase_give_up_radius: float = GOBLIN_CHASE_GIVE_UP_RADIUS,
        chase_stop_distance: float = GOBLIN_CHASE_STOP_DISTANCE,
        frame_duration: float = GOBLIN_ANIMATION_FRAME_DURATION,
        shadow_texture: int | None = None,
        shadow_size: tuple[float, float] = GOBLIN_SHADOW_SIZE,
        rng: random.Random | None = None,
    ) -> None:
        animations = self.animation_sets(texture)
        front_frames = animations.get("front", ())
        if not front_frames:
            raise ValueError("Goblin requires at least one texture frame")

        self.camera = camera
        self.ground_height_at = ground_height_at
        self.position_blocked = position_blocked
        self.player_in_building = player_in_building
        self.rng = rng or random.Random()
        self._animations: AnimationSets = {
            "front": front_frames,
            "right": animations.get("right", ()) or front_frames,
            "back": animations.get("back", ()) or front_frames,
        }
        self.spawn_position = Vector3(float(position.x), 0.0, float(position.z))
        self.roam_radius = max(1.0, float(roam_radius))
        self.move_speed = max(0.0, float(move_speed))
        self.collision_radius = max(0.0, float(collision_radius))
        self.chase_radius = max(0.0, float(chase_radius))
        self.chase_give_up_radius = max(
            self.chase_radius,
            float(chase_give_up_radius),
        )
        self.chase_stop_distance = max(0.0, float(chase_stop_distance))
        self._sprite_height = max(1.0, float(sprite_height))
        self._target: Vector3 | None = None
        self._chasing_player = False
        self._idle_timer = self.rng.uniform(0.0, GOBLIN_MAX_IDLE_SECONDS)
        self._facing_xz = Vector3(0.0, 0.0, 1.0)
        self._avoidance_turn = -1.0 if self.rng.random() < 0.5 else 1.0
        self._current_animation = ""
        self._current_flip_x = False
        self._logic_update_elapsed = self.rng.uniform(
            0.0,
            GOBLIN_LOGIC_UPDATE_INTERVAL,
        )
        self._direction_update_elapsed = self.rng.uniform(
            0.0,
            GOBLIN_DIRECTION_UPDATE_INTERVAL,
        )
        self.shadow_texture = int(shadow_texture or 0)
        self.shadow_size = (
            max(1.0, float(shadow_size[0])),
            max(1.0, float(shadow_size[1])),
        )
        self.shadow_render_batched = bool(self.shadow_texture)

        height = self._sprite_height
        y = float(ground_height_at(self.spawn_position.x, self.spawn_position.z))
        self._ground_y = y
        super().__init__(
            position=Vector3(
                self.spawn_position.x,
                y + height * 0.5,
                self.spawn_position.z,
            )
        )

        self.sprite = AnimatedWorldSprite(
            position=self.position,
            size=self.sprite_size_for_texture(front_frames[0], sprite_height=height),
            texture=front_frames[0],
            camera=camera,
            frames=front_frames,
            frame_duration=frame_duration,
            frame_index=self.rng.randrange(len(front_frames)),
        )
        self._set_directional_animation("front")

    @classmethod
    def animation_frames(cls, texture: Any | None = None) -> tuple[Any, ...]:
        if isinstance(texture, (list, tuple)):
            return tuple(frame for frame in texture if frame)
        return (texture,) if texture else ()

    @classmethod
    def animation_sets(cls, texture: Any | None = None) -> AnimationSets:
        animations: AnimationSets = {}
        if isinstance(texture, dict):
            animations = {
                "front": cls.animation_frames(texture.get("front")),
                "right": cls.animation_frames(texture.get("right")),
                "back": cls.animation_frames(texture.get("back")),
            }
        else:
            animations["front"] = cls.animation_frames(texture)

        if not animations.get("front"):
            animations["front"] = _load_frames(GOBLIN_FRONT_TEXTURE_DIR_PATH)
        if not animations.get("right"):
            animations["right"] = _load_frames(GOBLIN_RIGHT_TEXTURE_DIR_PATH)
        if not animations.get("back"):
            animations["back"] = _load_frames(GOBLIN_BACK_TEXTURE_DIR_PATH)
        return animations

    @classmethod
    def texture_or_load(cls, texture: Any | None = None) -> AnimationSets:
        return cls.animation_sets(texture)

    @staticmethod
    def sprite_size_for_texture(
        texture: Any,
        *,
        sprite_height: float = GOBLIN_SPRITE_HEIGHT,
    ) -> tuple[float, float]:
        tex_size = get_texture_size(texture)
        aspect = (tex_size[0] / tex_size[1]) if tex_size and tex_size[1] else 1.0
        height = max(1.0, float(sprite_height))
        return (height * aspect, height)

    def get_sprites(self) -> Iterable[object]:
        return (self.sprite,)

    def draw(self, camera=None) -> None:  # pragma: no cover - visual
        self._draw_shadow()

    def update(self, dt: float) -> None:
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return

        self._direction_update_elapsed += dt
        direction_due = self._direction_update_elapsed >= GOBLIN_DIRECTION_UPDATE_INTERVAL
        if direction_due:
            self._direction_update_elapsed %= GOBLIN_DIRECTION_UPDATE_INTERVAL

        self._logic_update_elapsed += dt
        if self._logic_update_elapsed < GOBLIN_LOGIC_UPDATE_INTERVAL:
            if direction_due:
                self._update_directional_animation()
            return

        logic_dt = min(self._logic_update_elapsed, GOBLIN_LOGIC_UPDATE_INTERVAL * 2.0)
        self._logic_update_elapsed %= GOBLIN_LOGIC_UPDATE_INTERVAL
        try:
            chase_target = self._current_chase_target()
            if chase_target is not None:
                self._move_towards(
                    chase_target,
                    logic_dt,
                    clamp_to_roam=False,
                    stop_distance=self.chase_stop_distance,
                )
                return

            returning_to_spawn = self._outside_roam_radius()
            if self._idle_timer > 0.0 and not returning_to_spawn:
                self._idle_timer = max(0.0, self._idle_timer - logic_dt)
                return

            if returning_to_spawn:
                self._target = Vector3(
                    self.spawn_position.x,
                    0.0,
                    self.spawn_position.z,
                )
            elif self._target is None:
                self._target = self._pick_roam_target()

            move_result = self._move_towards(
                self._target,
                logic_dt,
                clamp_to_roam=not returning_to_spawn,
                stop_distance=GOBLIN_TARGET_REACHED_DISTANCE,
            )
            if move_result in {"reached", "blocked"}:
                self._pause_before_next_roam()
        finally:
            if direction_due:
                self._update_directional_animation()

    def _current_chase_target(self) -> Vector3 | None:
        player_position = getattr(self.camera, "position", None)
        if player_position is None or self._player_is_in_building():
            self._stop_chasing()
            return None

        dx = float(player_position.x) - self.position.x
        dz = float(player_position.z) - self.position.z
        distance_sq = dx * dx + dz * dz
        active_radius = (
            self.chase_give_up_radius if self._chasing_player else self.chase_radius
        )
        if distance_sq > active_radius * active_radius:
            self._stop_chasing()
            return None

        if not self._chasing_player:
            self._target = None
        self._chasing_player = True
        self._idle_timer = 0.0
        return Vector3(float(player_position.x), 0.0, float(player_position.z))

    def _stop_chasing(self) -> None:
        if not self._chasing_player:
            return
        self._chasing_player = False
        self._target = None
        self._idle_timer = 0.0

    def _player_is_in_building(self) -> bool:
        if self.player_in_building is None:
            return False
        try:
            return bool(self.player_in_building())
        except Exception:
            return False

    def _outside_roam_radius(self) -> bool:
        dx = self.position.x - self.spawn_position.x
        dz = self.position.z - self.spawn_position.z
        return dx * dx + dz * dz > self.roam_radius * self.roam_radius

    def _move_towards(
        self,
        target: Vector3,
        logic_dt: float,
        *,
        clamp_to_roam: bool,
        stop_distance: float,
    ) -> str:
        dx = float(target.x) - self.position.x
        dz = float(target.z) - self.position.z
        distance_sq = dx * dx + dz * dz
        reached = max(0.0, float(stop_distance))
        if self._target_reached(target, distance_sq, reached):
            self._set_facing(dx, dz)
            return "reached"

        distance = math.sqrt(distance_sq)
        if distance <= 1e-6:
            return "reached"

        step = min(distance, self.move_speed * logic_dt)
        candidate = self._find_walk_step(
            dx / distance,
            dz / distance,
            step,
            clamp_to_roam=clamp_to_roam,
        )
        if candidate is None:
            self._set_facing(dx, dz)
            return "blocked"

        next_x, next_z = candidate
        self._set_facing(next_x - self.position.x, next_z - self.position.z)
        self._set_xz(next_x, next_z)
        return "moved"

    def _find_walk_step(
        self,
        dir_x: float,
        dir_z: float,
        step: float,
        *,
        clamp_to_roam: bool,
    ) -> tuple[float, float] | None:
        step = max(0.0, float(step))
        if step <= 1e-6:
            return None

        start_x = float(self.position.x)
        start_z = float(self.position.z)

        direct = self._walk_candidate(
            dir_x,
            dir_z,
            step,
            clamp_to_roam=clamp_to_roam,
        )
        if direct is not None:
            self._avoidance_turn = -1.0 if self.rng.random() < 0.5 else 1.0
            return direct

        for scale in GOBLIN_NAV_STEP_SCALES:
            stride = step * float(scale)
            if stride < GOBLIN_NAV_MIN_STEP:
                continue

            for turn in (self._avoidance_turn, -self._avoidance_turn):
                for degrees in GOBLIN_NAV_STEER_DEGREES:
                    angle = math.radians(float(degrees) * float(turn))
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    move_x = dir_x * cos_a - dir_z * sin_a
                    move_z = dir_x * sin_a + dir_z * cos_a
                    candidate = self._walk_candidate(
                        move_x,
                        move_z,
                        stride,
                        clamp_to_roam=clamp_to_roam,
                    )
                    if candidate is not None:
                        self._avoidance_turn = float(turn)
                        return candidate

        return None

    def _walk_candidate(
        self,
        move_x: float,
        move_z: float,
        stride: float,
        *,
        clamp_to_roam: bool,
    ) -> tuple[float, float] | None:
        start_x = float(self.position.x)
        start_z = float(self.position.z)
        next_x = start_x + float(move_x) * float(stride)
        next_z = start_z + float(move_z) * float(stride)
        if clamp_to_roam:
            next_x, next_z = self._clamp_to_roam_radius(next_x, next_z)

        actual_dx = next_x - start_x
        actual_dz = next_z - start_z
        if actual_dx * actual_dx + actual_dz * actual_dz < 1e-8:
            return None
        if not self._can_stand_at(next_x, next_z):
            return None
        return float(next_x), float(next_z)

    def _target_reached(
        self,
        target: Vector3,
        horizontal_distance_sq: float,
        stop_distance: float,
    ) -> bool:
        stop_distance = max(0.0, float(stop_distance))
        if horizontal_distance_sq > stop_distance * stop_distance:
            return False
        if stop_distance <= 0.0:
            return True

        try:
            target_ground_y = float(
                self.ground_height_at(float(target.x), float(target.z))
            )
        except Exception:
            return True

        height_delta = target_ground_y - float(
            getattr(self, "_ground_y", self.position.y)
        )
        surface_distance_sq = horizontal_distance_sq + height_delta * height_delta
        return surface_distance_sq <= stop_distance * stop_distance

    def _pause_before_next_roam(self) -> None:
        self._target = None
        self._idle_timer = self.rng.uniform(
            GOBLIN_MIN_IDLE_SECONDS,
            GOBLIN_MAX_IDLE_SECONDS,
        )

    def _pick_roam_target(self) -> Vector3:
        for _ in range(16):
            angle = self.rng.uniform(0.0, math.tau)
            radius = self.roam_radius * math.sqrt(self.rng.random())
            x = self.spawn_position.x + math.cos(angle) * radius
            z = self.spawn_position.z + math.sin(angle) * radius
            if self._can_stand_at(x, z) and self._path_clear_to(x, z):
                return Vector3(x, 0.0, z)

        return Vector3(self.spawn_position.x, 0.0, self.spawn_position.z)

    def _path_clear_to(self, x: float, z: float) -> bool:
        dx = float(x) - self.position.x
        dz = float(z) - self.position.z
        distance = math.hypot(dx, dz)
        if distance <= 1e-6:
            return True

        spacing = max(2.0, min(GOBLIN_PATH_SAMPLE_SPACING, self.collision_radius))
        samples = max(1, int(math.ceil(distance / spacing)))
        for index in range(1, samples + 1):
            t = index / samples
            if not self._can_stand_at(
                self.position.x + dx * t,
                self.position.z + dz * t,
            ):
                return False
        return True

    def _can_stand_at(self, x: float, z: float) -> bool:
        if self.position_blocked is None:
            return True
        return not self.position_blocked(float(x), float(z), self.collision_radius)

    def _clamp_to_roam_radius(self, x: float, z: float) -> tuple[float, float]:
        dx = float(x) - self.spawn_position.x
        dz = float(z) - self.spawn_position.z
        distance_sq = dx * dx + dz * dz
        radius_sq = self.roam_radius * self.roam_radius
        if distance_sq <= radius_sq or distance_sq <= 1e-6:
            return float(x), float(z)

        scale = self.roam_radius / math.sqrt(distance_sq)
        return (
            self.spawn_position.x + dx * scale,
            self.spawn_position.z + dz * scale,
        )

    def _set_xz(self, x: float, z: float) -> None:
        self.position.x = float(x)
        self.position.z = float(z)
        self._ground_y = float(self.ground_height_at(self.position.x, self.position.z))
        self.position.y = self._ground_y + self._sprite_height * 0.5
        self.sprite.position = self.position

    def _set_facing(self, dx: float, dz: float) -> None:
        length_sq = dx * dx + dz * dz
        if length_sq <= 1e-8:
            return
        length = math.sqrt(length_sq)
        self._facing_xz = Vector3(dx / length, 0.0, dz / length)

    def _direction_for_camera(self) -> tuple[str, bool]:
        camera_position = getattr(self.camera, "position", None)
        if camera_position is None:
            return "front", False

        view_x = float(camera_position.x) - self.position.x
        view_z = float(camera_position.z) - self.position.z
        view_len_sq = view_x * view_x + view_z * view_z
        if view_len_sq <= 1e-8:
            return "front", False

        facing = self._facing_xz
        front_dot = facing.x * view_x + facing.z * view_z
        front_threshold_sq = (
            GOBLIN_VIEW_FRONT_DOT
            * GOBLIN_VIEW_FRONT_DOT
            * view_len_sq
        )
        if front_dot >= 0.0 and front_dot * front_dot >= front_threshold_sq:
            return "front", False
        if front_dot <= 0.0 and front_dot * front_dot >= front_threshold_sq:
            return "back", False

        right_x = facing.z
        right_z = -facing.x
        right_dot = right_x * view_x + right_z * view_z
        return "right", right_dot > 0.0

    def _set_directional_animation(self, animation: str, *, flip_x: bool = False) -> None:
        frames = self._animations.get(animation, ())
        if not frames:
            frames = self._animations["front"]
            animation = "front"
            flip_x = False

        visual_name = "left" if animation == "right" and flip_x else animation
        if (
            self._current_animation == visual_name
            and self._current_flip_x == bool(flip_x)
        ):
            return

        self.sprite.frames = frames
        self.sprite.flip_x = bool(flip_x)
        self.sprite.frame_index %= len(frames)
        self.sprite._apply_frame(self.sprite.frame_index)
        self.sprite.size = self.sprite_size_for_texture(
            frames[0],
            sprite_height=self._sprite_height,
        )
        self._current_animation = visual_name
        self._current_flip_x = bool(flip_x)

    def _update_directional_animation(self) -> None:
        animation, flip_x = self._direction_for_camera()
        self._set_directional_animation(animation, flip_x=flip_x)

    def _draw_shadow(self) -> None:
        if not self.shadow_texture:
            return

        x = float(self.position.x)
        z = float(self.position.z)
        y = float(getattr(self, "_ground_y", self.position.y)) + GOBLIN_SHADOW_ELEVATION
        width, depth = self.shadow_size
        half_w = width * 0.5
        half_d = depth * 0.5

        glDepthMask(False)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_ALPHA_TEST)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glBindTexture(GL_TEXTURE_2D, self.shadow_texture)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        begun = False
        try:
            glBegin(GL_QUADS)
            begun = True
            glTexCoord2f(0.0, 0.0)
            glVertex3f(x - half_w, y, z - half_d)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(x + half_w, y, z - half_d)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(x + half_w, y, z + half_d)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(x - half_w, y, z + half_d)
        finally:
            if begun:
                glEnd()
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glDisable(GL_POLYGON_OFFSET_FILL)
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_BLEND)
            glDepthMask(True)


def draw_goblin_shadows_batched(
    goblins,
    *,
    camera=None,
    view_distance: float | None = None,
) -> None:  # pragma: no cover - visual
    groups: dict[int, list[Goblin]] = {}
    cam_pos = getattr(camera, "position", None)
    frustum_test = getattr(camera, "sphere_in_frustum", None)
    view_distance_sq = None
    if cam_pos is not None and view_distance is not None and not callable(frustum_test):
        view_distance_sq = float(view_distance) * float(view_distance)

    for goblin in goblins or ():
        if not getattr(goblin, "enabled", True) or not getattr(
            goblin,
            "visible",
            True,
        ):
            continue
        texture = int(getattr(goblin, "shadow_texture", 0) or 0)
        if not texture:
            continue
        position = getattr(goblin, "position", None)
        if position is None:
            continue
        if callable(frustum_test):
            width, depth = getattr(goblin, "shadow_size", GOBLIN_SHADOW_SIZE)
            radius = max(float(width), float(depth)) * 0.75
            ground_y = float(getattr(goblin, "_ground_y", position.y))
            center = (float(position.x), ground_y + GOBLIN_SHADOW_ELEVATION, float(position.z))
            if not frustum_test(center, radius, far_distance=view_distance):
                continue
        if view_distance_sq is not None:
            dx = float(position.x) - float(cam_pos.x)
            dy = float(position.y) - float(cam_pos.y)
            dz = float(position.z) - float(cam_pos.z)
            if (dx * dx + dy * dy + dz * dz) > view_distance_sq:
                continue
        groups.setdefault(texture, []).append(goblin)

    if not groups:
        return

    scratch_owner = camera if camera is not None else next(iter(groups.values()))[0]

    glDepthMask(False)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glDisable(GL_ALPHA_TEST)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(-1.0, -1.0)
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)

    try:
        for texture, group in groups.items():
            shadow_count = len(group)
            vertex_count = shadow_count * 6
            scratch = _shadow_batch_scratch(scratch_owner, shadow_count)
            vertices = scratch["vertices"][:vertex_count]
            texcoords = scratch["texcoords"][:vertex_count]

            texcoord_view = texcoords.reshape(shadow_count, 6, 2)
            texcoord_view[:, 0, :] = (0.0, 0.0)
            texcoord_view[:, 1, :] = (1.0, 0.0)
            texcoord_view[:, 2, :] = (1.0, 1.0)
            texcoord_view[:, 3, :] = (0.0, 0.0)
            texcoord_view[:, 4, :] = (1.0, 1.0)
            texcoord_view[:, 5, :] = (0.0, 1.0)

            vertex_view = vertices.reshape(shadow_count, 6, 3)
            for index, goblin in enumerate(group):
                position = goblin.position
                x = float(position.x)
                z = float(position.z)
                y = (
                    float(getattr(goblin, "_ground_y", position.y))
                    + GOBLIN_SHADOW_ELEVATION
                )
                width, depth = goblin.shadow_size
                half_w = float(width) * 0.5
                half_d = float(depth) * 0.5
                vertex_view[index, 0, :] = (x - half_w, y, z - half_d)
                vertex_view[index, 1, :] = (x + half_w, y, z - half_d)
                vertex_view[index, 2, :] = (x + half_w, y, z + half_d)
                vertex_view[index, 3, :] = (x - half_w, y, z - half_d)
                vertex_view[index, 4, :] = (x + half_w, y, z + half_d)
                vertex_view[index, 5, :] = (x - half_w, y, z + half_d)

            glBindTexture(GL_TEXTURE_2D, texture)
            glVertexPointer(3, GL_FLOAT, 0, vertices)
            glTexCoordPointer(2, GL_FLOAT, 0, texcoords)
            glDrawArrays(GL_TRIANGLES, 0, vertex_count)
    finally:
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDepthMask(True)
