"""Roaming goblin entity backed by a front-facing billboard animation."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable, Iterable

from pygame.math import Vector3

from engine.entity import Entity
from engine.rendering.sprite import AnimatedWorldSprite
from textures.resource_path import (
    GOBLIN_FRONT_TEXTURE_DIR_PATH,
)
from textures.texture_utils import get_texture_size, load_texture_atlas


GOBLIN_SPRITE_HEIGHT = 42.0
GOBLIN_ANIMATION_FPS = 8.0
GOBLIN_ANIMATION_FRAME_DURATION = 1.0 / GOBLIN_ANIMATION_FPS
GOBLIN_ROAM_RADIUS = 145.0
GOBLIN_MOVE_SPEED = 32.0
GOBLIN_COLLISION_RADIUS = 18.0
GOBLIN_TARGET_REACHED_DISTANCE = 7.0
GOBLIN_MIN_IDLE_SECONDS = 0.35
GOBLIN_MAX_IDLE_SECONDS = 1.45

PositionBlockedFn = Callable[[float, float, float], bool]


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.lower())


def _front_frame_paths() -> list[str]:
    frame_dir = Path(GOBLIN_FRONT_TEXTURE_DIR_PATH)
    if not frame_dir.is_dir():
        return []
    return [str(path) for path in sorted(frame_dir.glob("*.png"), key=_frame_sort_key)]


class Goblin(Entity):
    """A simple wandering forest creature.

    The animation is intentionally front-facing only for now. Movement-facing
    and camera-facing variants can slot in later by swapping the active frame
    set before the sprite animation update runs.
    """

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
        sprite_height: float = GOBLIN_SPRITE_HEIGHT,
        roam_radius: float = GOBLIN_ROAM_RADIUS,
        move_speed: float = GOBLIN_MOVE_SPEED,
        collision_radius: float = GOBLIN_COLLISION_RADIUS,
        frame_duration: float = GOBLIN_ANIMATION_FRAME_DURATION,
        rng: random.Random | None = None,
    ) -> None:
        frames = self.animation_frames(texture)
        if not frames:
            raise ValueError("Goblin requires at least one texture frame")

        self.ground_height_at = ground_height_at
        self.position_blocked = position_blocked
        self.rng = rng or random.Random()
        self.spawn_position = Vector3(float(position.x), 0.0, float(position.z))
        self.roam_radius = max(1.0, float(roam_radius))
        self.move_speed = max(0.0, float(move_speed))
        self.collision_radius = max(0.0, float(collision_radius))
        self._sprite_height = max(1.0, float(sprite_height))
        self._target: Vector3 | None = None
        self._idle_timer = self.rng.uniform(0.0, GOBLIN_MAX_IDLE_SECONDS)
        self._current_animation = "front"
        self._front_frames = frames

        height = self._sprite_height
        y = float(ground_height_at(self.spawn_position.x, self.spawn_position.z))
        super().__init__(
            position=Vector3(
                self.spawn_position.x,
                y + height * 0.5,
                self.spawn_position.z,
            )
        )

        self.sprite = AnimatedWorldSprite(
            position=self.position,
            size=self.sprite_size_for_texture(frames[0], sprite_height=height),
            texture=frames[0],
            camera=camera,
            frames=frames,
            frame_duration=frame_duration,
            frame_index=self.rng.randrange(len(frames)),
        )

    @classmethod
    def animation_frames(cls, texture: Any | None = None) -> tuple[Any, ...]:
        if isinstance(texture, (list, tuple)):
            return tuple(frame for frame in texture if frame)
        return (texture,) if texture else ()

    @classmethod
    def texture_or_load(cls, texture: Any | None = None) -> tuple[Any, ...]:
        frames = cls.animation_frames(texture)
        if frames:
            return frames

        frame_paths = _front_frame_paths()
        if frame_paths:
            frames = tuple(load_texture_atlas(frame_paths))
            if frames:
                return frames

        return ()

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

    def update(self, dt: float) -> None:
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return

        if self._idle_timer > 0.0:
            self._idle_timer = max(0.0, self._idle_timer - dt)
            return

        if self._target is None:
            self._target = self._pick_roam_target()

        dx = self._target.x - self.position.x
        dz = self._target.z - self.position.z
        distance_sq = dx * dx + dz * dz
        reached = GOBLIN_TARGET_REACHED_DISTANCE
        if distance_sq <= reached * reached:
            self._pause_before_next_roam()
            return

        distance = math.sqrt(distance_sq)
        if distance <= 1e-6:
            self._pause_before_next_roam()
            return

        step = min(distance, self.move_speed * dt)
        next_x = self.position.x + (dx / distance) * step
        next_z = self.position.z + (dz / distance) * step
        next_x, next_z = self._clamp_to_roam_radius(next_x, next_z)

        if not self._can_stand_at(next_x, next_z):
            self._pause_before_next_roam()
            return

        self._set_xz(next_x, next_z)

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
            if self._can_stand_at(x, z):
                return Vector3(x, 0.0, z)

        return Vector3(self.spawn_position.x, 0.0, self.spawn_position.z)

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
        self.position.y = (
            float(self.ground_height_at(self.position.x, self.position.z))
            + self._sprite_height * 0.5
        )
        self.sprite.position = self.position
