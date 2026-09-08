"""Roaming goblins rendered as a dynamic batch of generated low-poly models."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable

import numpy as np
from pygame.math import Vector3
from OpenGL.GL import (
    GL_ALPHA_TEST,
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_COLOR_ARRAY,
    GL_FLOAT,
    GL_NORMAL_ARRAY,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POLYGON_OFFSET_FILL,
    GL_QUADS,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE0,
    GL_TEXTURE_COORD_ARRAY,
    GL_TEXTURE_ENV,
    GL_TEXTURE_ENV_MODE,
    GL_MODULATE,
    GL_VERTEX_ARRAY,
    glBindBuffer,
    glBindTexture,
    glBlendFunc,
    glClientActiveTexture,
    glColorPointer,
    glColor4f,
    glDepthMask,
    glDisable,
    glDisableClientState,
    glEnable,
    glEnableClientState,
    glDrawArrays,
    glNormalPointer,
    glPolygonOffset,
    glTexEnvi,
    glTexCoordPointer,
    glVertexPointer,
)

from engine.rendering.state import use_fixed_pipeline
from engine.rendering.mesh import BatchedMesh
from engine.sound.sound_utils import Sounds
from game.actors.creature import Creature
from game.actors.goblin_model import (
    GOBLIN_MODEL_POSE_COUNT,
    GOBLIN_MODEL_VARIANT_COUNT,
    cached_goblin_local_pose_vertex_data,
    goblin_model_pose_index,
    prewarm_goblin_pose_cache,
    transformed_cached_goblin_vertex_data,
)
from game.inventory import InventoryItem, goblin_equipment_set
from game.world.rendering.receivers import GOBLIN_LIGHTING_RECEIVER

GOBLIN_MODEL_HEIGHT = 42.0
# Source-compatible name for callers that used the old billboard height knob.
GOBLIN_SPRITE_HEIGHT = GOBLIN_MODEL_HEIGHT
GOBLIN_WALK_CYCLE_DISTANCE = 30.0
GOBLIN_ROAM_RADIUS = 145.0
GOBLIN_MOVE_SPEED = 32.0
GOBLIN_COLLISION_RADIUS = 18.0
GOBLIN_CHASE_RADIUS = 260.0
GOBLIN_CHASE_GIVE_UP_RADIUS = 380.0
GOBLIN_CHASE_STOP_DISTANCE = 34.0
GOBLIN_MAX_HP = 5
GOBLIN_TARGET_REACHED_DISTANCE = 7.0
GOBLIN_MIN_IDLE_SECONDS = 0.35
GOBLIN_MAX_IDLE_SECONDS = 1.45
GOBLIN_LOGIC_UPDATE_INTERVAL = 1.0 / 30.0
GOBLIN_DISTANT_LOGIC_UPDATE_INTERVAL = 1.0 / 15.0
GOBLIN_SHADOW_SIZE = (30.0, 22.0)
GOBLIN_SHADOW_ELEVATION = 0.35
GOBLIN_PATH_SAMPLE_SPACING = 12.0
GOBLIN_NAV_MIN_STEP = 0.15
GOBLIN_NAV_STEP_SCALES = (1.0, 0.55, 0.25)
GOBLIN_NAV_STEER_DEGREES = (30.0, 60.0, 90.0, 135.0)
GOBLIN_SOUND_KEYS = tuple(f"goblin_{i}" for i in range(1, 4))
GOBLIN_SOUND_MIN_INTERVAL = 4.0
GOBLIN_SOUND_MAX_INTERVAL = 11.0
GOBLIN_SOUND_AUDIBLE_RADIUS = 520.0
GOBLIN_SOUND_FULL_VOLUME_RADIUS = 70.0
GOBLIN_SOUND_MAX_VOLUME = 0.65
GOBLIN_SOUND_INDOOR_MULTIPLIER = 0.22
GOBLIN_STEP_SOUND_KEY = "step"
GOBLIN_STEP_INTERVAL = 0.48
GOBLIN_STEP_INTERVAL_JITTER = 0.12
GOBLIN_STEP_AUDIBLE_RADIUS = 280.0
GOBLIN_STEP_FULL_VOLUME_RADIUS = 35.0
GOBLIN_STEP_MAX_VOLUME = 0.16
GOBLIN_STEP_INDOOR_MULTIPLIER = 0.16

PositionBlockedFn = Callable[[float, float, float], bool]
PlayerInBuildingFn = Callable[[], bool]


def _shadow_batch_scratch(owner, shadow_count: int) -> dict:
    scratch = getattr(owner, "_goblin_shadow_batch_scratch", None)
    current_capacity = int(scratch.get("shadow_capacity", 0)) if scratch else 0
    if scratch is not None and current_capacity >= shadow_count:
        return scratch

    shadow_capacity = max(64, int(shadow_count), int(current_capacity * 1.5))
    vertex_capacity = shadow_capacity * 4
    scratch = {
        "shadow_capacity": shadow_capacity,
        "vertices": np.empty((vertex_capacity, 3), dtype=np.float32),
        "texcoords": np.empty((vertex_capacity, 2), dtype=np.float32),
    }
    setattr(owner, "_goblin_shadow_batch_scratch", scratch)
    return scratch


class Goblin(Creature):
    """A wandering forest creature with generated, poseable 3D geometry."""

    DEFAULT_HEIGHT = GOBLIN_SPRITE_HEIGHT
    DEFAULT_ROAM_RADIUS = GOBLIN_ROAM_RADIUS
    DEFAULT_MOVE_SPEED = GOBLIN_MOVE_SPEED
    DEFAULT_MAX_HP = GOBLIN_MAX_HP
    DISPLAY_NAME = "Goblin"
    DEFAULT_ATTACK_DAMAGE = 1
    MODEL_VARIANT_COUNT = GOBLIN_MODEL_VARIANT_COUNT
    WALK_CYCLE_DISTANCE = GOBLIN_WALK_CYCLE_DISTANCE
    transformed_pose_vertex_data = staticmethod(transformed_cached_goblin_vertex_data)
    pose_index = staticmethod(goblin_model_pose_index)
    packet_lighting_receiver = GOBLIN_LIGHTING_RECEIVER
    render_batched = True
    casts_shadows = False

    def __init__(
        self,
        position: Vector3,
        *,
        material_texture: Any | None = None,
        texture: Any | None = None,
        camera: object,
        ground_height_at: Callable[[float, float], float],
        position_blocked: PositionBlockedFn | None = None,
        player_in_building: PlayerInBuildingFn | None = None,
        model_height: float = GOBLIN_MODEL_HEIGHT,
        sprite_height: float | None = None,
        roam_radius: float = GOBLIN_ROAM_RADIUS,
        move_speed: float = GOBLIN_MOVE_SPEED,
        collision_radius: float = GOBLIN_COLLISION_RADIUS,
        chase_radius: float = GOBLIN_CHASE_RADIUS,
        chase_give_up_radius: float = GOBLIN_CHASE_GIVE_UP_RADIUS,
        chase_stop_distance: float = GOBLIN_CHASE_STOP_DISTANCE,
        frame_duration: float | None = None,
        shadow_texture: int | None = None,
        shadow_size: tuple[float, float] = GOBLIN_SHADOW_SIZE,
        sound_keys: Iterable[str] | None = None,
        max_hp: int = GOBLIN_MAX_HP,
        rng: random.Random | None = None,
    ) -> None:
        self.camera = camera
        self.ground_height_at = ground_height_at
        self.position_blocked = position_blocked
        self.player_in_building = player_in_building
        self.rng = rng or random.Random()
        self.material_texture = (
            material_texture if material_texture is not None else texture
        )
        keys = GOBLIN_SOUND_KEYS if sound_keys is None else tuple(sound_keys)
        self.sound_keys = tuple(str(key) for key in keys if key)
        self._sound_channel = None
        self._sound_key: str | None = None
        self._sound_timer = self._next_sound_delay(initial=True)
        self._step_sound_timer = self.rng.uniform(0.0, GOBLIN_STEP_INTERVAL)
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
        height_value = model_height if sprite_height is None else sprite_height
        self._model_height = max(1.0, float(height_value))
        self._target: Vector3 | None = None
        self._chasing_player = False
        self._idle_timer = self.rng.uniform(0.0, GOBLIN_MAX_IDLE_SECONDS)
        self._facing_xz = Vector3(0.0, 0.0, 1.0)
        self._avoidance_turn = -1.0 if self.rng.random() < 0.5 else 1.0
        self._model_seed = self.rng.randrange(1 << 30)
        self._model_variant = self._model_seed % self.MODEL_VARIANT_COUNT
        self._walk_phase = self.rng.uniform(0.0, math.tau)
        self._is_moving = False
        self._model_revision = 0
        self._model_vertex_cache_revision = -1
        self._model_vertex_cache: np.ndarray | None = None
        self._logic_update_elapsed = self.rng.uniform(
            0.0,
            GOBLIN_LOGIC_UPDATE_INTERVAL,
        )
        self.shadow_texture = int(shadow_texture or 0)
        self.shadow_size = (
            max(1.0, float(shadow_size[0])),
            max(1.0, float(shadow_size[1])),
        )
        self.shadow_render_batched = bool(self.shadow_texture)

        height = self._model_height
        y = float(ground_height_at(self.spawn_position.x, self.spawn_position.z))
        self._ground_y = y
        super().__init__(
            position=Vector3(
                self.spawn_position.x,
                y + height * 0.5,
                self.spawn_position.z,
            ),
            display_name=self.DISPLAY_NAME,
            max_hp=max_hp,
            attack_damage=self.DEFAULT_ATTACK_DAMAGE,
        )

    def get_sprites(self) -> Iterable[object]:
        return ()

    @property
    def model_revision(self) -> int:
        return self._model_revision

    def model_vertex_data(self) -> np.ndarray:
        """Return this actor's cached generated pose in world coordinates."""

        if (
            self._model_vertex_cache is None
            or self._model_vertex_cache_revision != self._model_revision
        ):
            self._model_vertex_cache = self.transformed_pose_vertex_data(
                (float(self.position.x), self._ground_y, float(self.position.z)),
                facing=(float(self._facing_xz.x), float(self._facing_xz.z)),
                height=self._model_height,
                walk_phase=self._walk_phase,
                moving=self._is_moving,
                variant=self._model_variant,
            )
            self._model_vertex_cache_revision = self._model_revision
        return self._model_vertex_cache

    @property
    def model_vertex_count(self) -> int:
        return int(self.model_vertex_data().shape[0])

    def get_bounding_sphere(self) -> tuple[tuple[float, float, float], float]:
        half_height = self._model_height * 0.5
        half_width = self._model_height * 0.34
        return (
            (
                float(self.position.x),
                float(self.position.y),
                float(self.position.z),
            ),
            math.hypot(half_height, half_width),
        )

    def _mark_model_dirty(self) -> None:
        self._model_revision += 1

    def _set_model_moving(self, moving: bool) -> None:
        moving_value = bool(moving)
        if self._is_moving == moving_value:
            return
        self._is_moving = moving_value
        self._mark_model_dirty()

    def draw(self, camera=None, **_kwargs) -> None:  # pragma: no cover - visual
        """The scene-owned :class:`GoblinRenderBatch` draws all goblin bodies."""

    def kill(self) -> None:
        self.enabled = False
        self.visible = False
        self._mark_model_dirty()
        channel = self._sound_channel
        if channel is not None:
            try:
                channel.stop()
            except Exception:
                pass
        self._sound_channel = None
        self._sound_key = None

    def dispose(self) -> None:
        self.kill()

    def combat_rewards(self, scene) -> tuple[InventoryItem, ...]:
        equipment = goblin_equipment_set()
        return (self.rng.choice(equipment),)

    def update(self, dt: float) -> None:
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return

        self._update_sound(dt)

        logic_interval = self._logic_update_interval()
        self._logic_update_elapsed += dt
        if self._logic_update_elapsed < logic_interval:
            return

        logic_dt = min(self._logic_update_elapsed, logic_interval * 2.0)
        self._logic_update_elapsed %= logic_interval
        moving = False
        try:
            chase_target = self._current_chase_target()
            if chase_target is not None:
                move_result = self._move_towards(
                    chase_target,
                    logic_dt,
                    clamp_to_roam=False,
                    stop_distance=self.chase_stop_distance,
                )
                moving = move_result == "moved"
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
            moving = move_result == "moved"
            if move_result in {"reached", "blocked"}:
                self._pause_before_next_roam()
        finally:
            self._set_model_moving(moving)

    def _logic_update_interval(self) -> float:
        """Keep nearby combat responsive while reducing distant roaming work."""

        camera_position = getattr(self.camera, "position", None)
        if camera_position is None:
            return GOBLIN_LOGIC_UPDATE_INTERVAL
        dx = float(camera_position.x) - float(self.position.x)
        dz = float(camera_position.z) - float(self.position.z)
        if dx * dx + dz * dz <= self.chase_give_up_radius**2:
            return GOBLIN_LOGIC_UPDATE_INTERVAL
        return GOBLIN_DISTANT_LOGIC_UPDATE_INTERVAL

    def _next_sound_delay(self, *, initial: bool = False) -> float:
        minimum = max(0.1, float(GOBLIN_SOUND_MIN_INTERVAL))
        maximum = max(minimum, float(GOBLIN_SOUND_MAX_INTERVAL))
        if initial:
            return self.rng.uniform(0.5, maximum)
        return self.rng.uniform(minimum, maximum)

    def _update_sound(self, dt: float) -> None:
        self._update_active_sound_volume()
        self._sound_timer -= dt
        if self._sound_timer > 0.0:
            return

        self._sound_timer = self._next_sound_delay()
        if self._sound_is_active():
            return

        keys = self._available_sound_keys()
        if not keys:
            return

        key = self.rng.choice(keys)
        channel = Sounds.play_at(
            key,
            self.position,
            self.camera,
            audible_radius=GOBLIN_SOUND_AUDIBLE_RADIUS,
            full_volume_radius=GOBLIN_SOUND_FULL_VOLUME_RADIUS,
            max_volume=self._indoor_scaled_volume(
                GOBLIN_SOUND_MAX_VOLUME,
                GOBLIN_SOUND_INDOOR_MULTIPLIER,
            ),
        )
        if channel is None:
            self._sound_channel = None
            self._sound_key = None
            return
        self._sound_channel = channel
        self._sound_key = key

    def _available_sound_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self.sound_keys if Sounds.is_loaded(key))

    def _sound_is_active(self) -> bool:
        channel = self._sound_channel
        key = self._sound_key
        if channel is None or key is None:
            return False

        try:
            if not channel.get_busy():
                self._sound_channel = None
                self._sound_key = None
                return False

            sound = Sounds.get(key)
            if sound is not None and channel.get_sound() != sound:
                self._sound_channel = None
                self._sound_key = None
                return False
        except Exception:
            self._sound_channel = None
            self._sound_key = None
            return False

        return True

    def _update_active_sound_volume(self) -> None:
        if not self._sound_is_active():
            return

        channel = self._sound_channel
        if channel is None:
            return

        try:
            if not Sounds.set_channel_spatial(
                channel,
                self.position,
                self.camera,
                audible_radius=GOBLIN_SOUND_AUDIBLE_RADIUS,
                full_volume_radius=GOBLIN_SOUND_FULL_VOLUME_RADIUS,
                max_volume=self._indoor_scaled_volume(
                    GOBLIN_SOUND_MAX_VOLUME,
                    GOBLIN_SOUND_INDOOR_MULTIPLIER,
                ),
            ):
                channel.stop()
                self._sound_channel = None
                self._sound_key = None
        except Exception:
            self._sound_channel = None
            self._sound_key = None

    def _next_step_delay(self) -> float:
        interval = max(0.1, float(GOBLIN_STEP_INTERVAL))
        jitter = max(0.0, float(GOBLIN_STEP_INTERVAL_JITTER))
        return max(0.08, interval + self.rng.uniform(-jitter, jitter))

    def _update_step_sound(self, dt: float) -> None:
        self._step_sound_timer -= max(0.0, float(dt))
        if self._step_sound_timer > 0.0:
            return

        self._step_sound_timer = self._next_step_delay()
        if not Sounds.is_loaded(GOBLIN_STEP_SOUND_KEY):
            return

        Sounds.play_at(
            GOBLIN_STEP_SOUND_KEY,
            self.position,
            self.camera,
            audible_radius=GOBLIN_STEP_AUDIBLE_RADIUS,
            full_volume_radius=GOBLIN_STEP_FULL_VOLUME_RADIUS,
            max_volume=self._indoor_scaled_volume(
                GOBLIN_STEP_MAX_VOLUME,
                GOBLIN_STEP_INDOOR_MULTIPLIER,
            ),
        )

    def _indoor_scaled_volume(self, volume: float, indoor_multiplier: float) -> float:
        if not self._player_is_in_building():
            return float(volume)
        return float(volume) * max(0.0, min(1.0, float(indoor_multiplier)))

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
        self._update_step_sound(logic_dt)
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
        old_x = float(self.position.x)
        old_z = float(self.position.z)
        self.position.x = float(x)
        self.position.z = float(z)
        self._ground_y = float(self.ground_height_at(self.position.x, self.position.z))
        self.position.y = self._ground_y + self._model_height * 0.5
        distance = math.hypot(self.position.x - old_x, self.position.z - old_z)
        if distance > 1e-8:
            self._walk_phase = (
                self._walk_phase
                + math.tau * distance / max(1.0, self.WALK_CYCLE_DISTANCE)
            ) % math.tau
            self._mark_model_dirty()

    def _set_facing(self, dx: float, dz: float) -> None:
        length_sq = dx * dx + dz * dz
        if length_sq <= 1e-8:
            return
        length = math.sqrt(length_sq)
        next_facing = Vector3(dx / length, 0.0, dz / length)
        if self._facing_xz.dot(next_facing) >= 0.99995:
            return
        self._facing_xz = next_facing
        self._mark_model_dirty()


class GoblinRenderBatch:
    """Render generated goblins from a static local-space pose atlas."""

    packet_lighting_receiver = GOBLIN_LIGHTING_RECEIVER
    casts_shadows = False
    visible = True
    pose_count = GOBLIN_MODEL_POSE_COUNT
    cached_pose_vertex_data = staticmethod(cached_goblin_local_pose_vertex_data)
    prewarm_pose_cache = staticmethod(prewarm_goblin_pose_cache)

    def __init__(
        self,
        goblins: Iterable[Goblin],
        *,
        material_texture: Any | None = None,
        exposure_baseline: float = 1.0,
    ) -> None:
        self.goblins = tuple(goblin for goblin in goblins if goblin is not None)
        texture = (
            material_texture
            if material_texture is not None
            else next(
                (
                    goblin.material_texture
                    for goblin in self.goblins
                    if goblin.material_texture is not None
                ),
                None,
            )
        )
        self.material_texture = (
            int(getattr(texture, "texture", texture)) if texture is not None else None
        )
        self.exposure_baseline = float(exposure_baseline)
        if not math.isfinite(self.exposure_baseline):
            raise ValueError("goblin exposure_baseline must be finite")
        variants_by_height: dict[float, set[int]] = {}
        for goblin in self.goblins:
            variants_by_height.setdefault(goblin._model_height, set()).add(
                goblin._model_variant
            )
        for height, variants in variants_by_height.items():
            self.prewarm_pose_cache(tuple(variants), height=height)
        self._pose_ranges: dict[tuple[float, int, bool, int], tuple[int, int]] = {}
        self._pose_mesh = self._build_pose_atlas(variants_by_height)
        self._mesh: BatchedMesh | None = None
        self._cache_key: tuple | None = None
        self.visible_count = 0
        self.visible_vertex_count = 0

    def _build_pose_atlas(
        self,
        variants_by_height: dict[float, set[int]],
    ) -> BatchedMesh | None:
        if self.material_texture is None:
            return None

        parts: list[np.ndarray] = []
        first_vertex = 0
        for height in sorted(variants_by_height):
            for variant in sorted(variants_by_height[height]):
                pose_states = [(False, 0), *(
                    (True, pose_index)
                    for pose_index in range(self.pose_count)
                )]
                for moving, pose_index in pose_states:
                    phase = math.tau * pose_index / self.pose_count
                    vertex_data = self.cached_pose_vertex_data(
                        height=height,
                        walk_phase=phase,
                        moving=moving,
                        variant=variant,
                    )
                    count = int(vertex_data.shape[0])
                    self._pose_ranges[(height, variant, moving, pose_index)] = (
                        first_vertex,
                        count,
                    )
                    parts.append(vertex_data)
                    first_vertex += count

        if not parts:
            return None
        vertex_data = np.ascontiguousarray(
            parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0),
            dtype=np.float32,
        )
        return BatchedMesh.from_vertex_data(
            vertex_data,
            texture=self.material_texture,
            alpha_test=False,
            casts_shadows=False,
            casts_sun_shadows=False,
            casts_point_shadows=False,
            exposure_baseline=self.exposure_baseline,
            keep_vertex_data=False,
            environment_lighting=True,
            shine_enabled=False,
            shader_lighting=True,
            lighting_receiver=self.packet_lighting_receiver,
        )

    @staticmethod
    def _pose_key(goblin: Goblin) -> tuple[float, int, bool, int]:
        moving = bool(goblin._is_moving)
        pose_index = goblin.pose_index(
            goblin._walk_phase,
            moving=moving,
        )
        return (
            goblin._model_height,
            goblin._model_variant,
            moving,
            pose_index,
        )

    @staticmethod
    def _model_matrix(goblin: Goblin) -> tuple[float, ...]:
        forward_x = float(goblin._facing_xz.x)
        forward_z = float(goblin._facing_xz.z)
        origin_x = float(goblin.position.x)
        origin_y = float(goblin._ground_y)
        origin_z = float(goblin.position.z)
        return (
            forward_z,
            0.0,
            forward_x,
            origin_x,
            0.0,
            1.0,
            0.0,
            origin_y,
            -forward_x,
            0.0,
            forward_z,
            origin_z,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    def _visible_goblins(
        self,
        camera=None,
        view_distance: float | None = None,
    ) -> tuple[Goblin, ...]:
        active = tuple(
            goblin
            for goblin in self.goblins
            if getattr(goblin, "enabled", True)
            and getattr(goblin, "visible", True)
        )
        if camera is None:
            return active

        frustum_test = getattr(camera, "sphere_in_frustum", None)
        if callable(frustum_test):
            visible = []
            for goblin in active:
                center, radius = goblin.get_bounding_sphere()
                if frustum_test(center, radius, far_distance=view_distance):
                    visible.append(goblin)
            return tuple(visible)

        camera_position = getattr(camera, "position", None)
        if camera_position is None or view_distance is None:
            return active
        visible = []
        for goblin in active:
            center, radius = goblin.get_bounding_sphere()
            dx = center[0] - float(camera_position.x)
            dy = center[1] - float(camera_position.y)
            dz = center[2] - float(camera_position.z)
            maximum = float(view_distance) + float(radius)
            if dx * dx + dy * dy + dz * dz <= maximum * maximum:
                visible.append(goblin)
        return tuple(visible)

    @staticmethod
    def _current_cache_key(goblins: Iterable[Goblin]) -> tuple:
        return tuple(
            (
                id(goblin),
                bool(getattr(goblin, "enabled", True)),
                bool(getattr(goblin, "visible", True)),
                int(goblin.model_revision),
            )
            for goblin in goblins
        )

    def _combined_vertex_data(
        self,
        goblins: Iterable[Goblin] | None = None,
    ) -> np.ndarray:
        source = self.goblins if goblins is None else tuple(goblins)
        parts = [
            goblin.model_vertex_data()
            for goblin in source
            if getattr(goblin, "enabled", True)
            and getattr(goblin, "visible", True)
        ]
        parts = [part for part in parts if part.size]
        if not parts:
            return np.empty((0, 11), dtype=np.float32)
        return np.ascontiguousarray(
            parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0),
            dtype=np.float32,
        )

    def _ensure_mesh(
        self,
        camera=None,
        view_distance: float | None = None,
    ) -> BatchedMesh | None:
        visible_goblins = self._visible_goblins(camera, view_distance)
        cache_key = self._current_cache_key(visible_goblins)
        if self._mesh is not None and cache_key == self._cache_key:
            return self._mesh

        vertex_data = self._combined_vertex_data(visible_goblins)
        if self._mesh is None:
            if not vertex_data.size:
                self._cache_key = cache_key
                return None
            self._mesh = BatchedMesh.from_vertex_data(
                vertex_data,
                texture=self.material_texture,
                alpha_test=False,
                casts_shadows=False,
                casts_sun_shadows=False,
                casts_point_shadows=False,
                exposure_baseline=self.exposure_baseline,
                keep_vertex_data=False,
                environment_lighting=True,
                shine_enabled=False,
                shader_lighting=True,
                lighting_receiver=self.packet_lighting_receiver,
            )
        elif vertex_data.size:
            self._mesh.update_vertex_data(vertex_data, keep_vertex_data=False)
        else:
            self._mesh.vertex_count = 0
            self._mesh.bounds_center = None
            self._mesh.bounds_radius = 0.0
        self._cache_key = cache_key
        return self._mesh

    def _draw_packet_instances(
        self,
        visible_goblins: tuple[Goblin, ...],
        *,
        lighting_packet,
        packet_shader,
    ) -> None:
        mesh = self._pose_mesh
        if mesh is None or not visible_goblins:
            return

        packet_shader.bind(lighting_packet, directional_normal_stream=False)
        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo_vertices)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, mesh._vertex_stride, None)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, mesh._vertex_stride, mesh._color_offset_ptr)
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, mesh._vertex_stride, mesh._normal_offset_ptr)
        glClientActiveTexture(GL_TEXTURE0)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glTexCoordPointer(2, GL_FLOAT, mesh._vertex_stride, mesh._uv_offset_ptr)
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glBindTexture(GL_TEXTURE_2D, int(mesh.texture or 0))

        try:
            for goblin in visible_goblins:
                first_vertex, vertex_count = self._pose_ranges[self._pose_key(goblin)]
                packet_shader.set_model_matrix(self._model_matrix(goblin))
                glDrawArrays(mesh.draw_mode, first_vertex, vertex_count)
        finally:
            packet_shader.reset_model_matrix()
            use_fixed_pipeline()
            glDisable(GL_TEXTURE_2D)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    @property
    def vertex_count(self) -> int:
        total = 0
        for goblin in self.goblins:
            if not getattr(goblin, "enabled", True) or not getattr(
                goblin,
                "visible",
                True,
            ):
                continue
            span = self._pose_ranges.get(self._pose_key(goblin))
            total += span[1] if span is not None else goblin.model_vertex_count
        return int(total)

    def shadow_meshes(self, camera=None) -> tuple:
        return ()

    def draw(
        self,
        camera=None,
        *,
        view_distance: float | None = None,
        lighting_packet=None,
        packet_shader=None,
    ) -> None:  # pragma: no cover - visual
        visible_goblins = self._visible_goblins(camera, view_distance)
        self.visible_count = len(visible_goblins)
        self.visible_vertex_count = sum(
            self._pose_ranges[self._pose_key(goblin)][1]
            for goblin in visible_goblins
            if self._pose_key(goblin) in self._pose_ranges
        )
        if (
            self._pose_mesh is not None
            and packet_shader is not None
            and lighting_packet is not None
        ):
            self._draw_packet_instances(
                visible_goblins,
                lighting_packet=lighting_packet,
                packet_shader=packet_shader,
            )
            return

        mesh = self._ensure_mesh(camera, view_distance)
        if mesh is None:
            return
        mesh.draw(
            camera=camera,
            view_distance=view_distance,
            lighting_packet=lighting_packet,
            packet_shader=packet_shader,
        )

    def dispose(self) -> None:
        if self._pose_mesh is not None:
            self._pose_mesh.dispose()
        self._pose_mesh = None
        self._pose_ranges.clear()
        if self._mesh is not None:
            self._mesh.dispose()
        self._mesh = None
        self._cache_key = None


def draw_goblin_shadows_batched(
    goblins,
    *,
    camera=None,
    view_distance: float | None = None,
) -> None:  # pragma: no cover - visual
    groups: dict[int, list[Goblin]] = {}
    cam_pos = getattr(camera, "position", None)
    frustum_test = getattr(camera, "sphere_in_frustum", None)
    use_distance_fallback = False
    if cam_pos is not None and view_distance is not None and not callable(frustum_test):
        use_distance_fallback = True

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
        width, shadow_depth = getattr(goblin, "shadow_size", GOBLIN_SHADOW_SIZE)
        radius = max(float(width), float(shadow_depth)) * 0.75
        if callable(frustum_test):
            ground_y = float(getattr(goblin, "_ground_y", position.y))
            center = (
                float(position.x),
                ground_y + GOBLIN_SHADOW_ELEVATION,
                float(position.z),
            )
            if not frustum_test(center, radius, far_distance=view_distance):
                continue
        if use_distance_fallback:
            dx = float(position.x) - float(cam_pos.x)
            dy = float(position.y) - float(cam_pos.y)
            dz = float(position.z) - float(cam_pos.z)
            max_distance = float(view_distance) + radius
            if (dx * dx + dy * dy + dz * dz) > max_distance * max_distance:
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
            vertex_count = shadow_count * 4
            scratch = _shadow_batch_scratch(scratch_owner, shadow_count)
            vertices = scratch["vertices"][:vertex_count]
            texcoords = scratch["texcoords"][:vertex_count]

            texcoord_view = texcoords.reshape(shadow_count, 4, 2)
            texcoord_view[:, 0, :] = (0.0, 0.0)
            texcoord_view[:, 1, :] = (1.0, 0.0)
            texcoord_view[:, 2, :] = (1.0, 1.0)
            texcoord_view[:, 3, :] = (0.0, 1.0)

            vertex_view = vertices.reshape(shadow_count, 4, 3)
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
                vertex_view[index, 3, :] = (x - half_w, y, z + half_d)

            glBindTexture(GL_TEXTURE_2D, texture)
            glVertexPointer(3, GL_FLOAT, 0, vertices)
            glTexCoordPointer(2, GL_FLOAT, 0, texcoords)
            glDrawArrays(GL_QUADS, 0, vertex_count)
    finally:
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDepthMask(True)
