"""PlayerCameraController: handles mouse look, movement input, and wall blocking.

This isolates input and movement from WorldScene. It reads keys, updates
rotation smoothing toward targets set by mouse deltas, moves the camera via
Camera.move_camera(), and checks world bounds and wall collisions using hooks
on the scene.
"""

from __future__ import annotations

import math
import pygame
from pygame.math import Vector3
from typing import Tuple

from world.world_collision import movement_blocked_by_wall, player_support_height_at
from config import (
    MOUSE_SENSITIVITY,
    SPRINT_SPEED,
    BASE_SPEED,
    CAMERA_GROUND_OFFSET,
    JUMP_SPEED,
    PLAYER_HEAD_CLEARANCE,
    PLAYER_RADIUS,
)


class PlayerCameraController:
    def __init__(self, scene, camera, *, rot_smooth_hz: float = 4.0):
        self.scene = scene
        self.camera = camera
        self.rot_target_x = float(camera.rotation.x)
        self.rot_target_y = float(camera.rotation.y)
        self.rot_smooth_hz = float(rot_smooth_hz)

    def _eye_to_foot_offset(self) -> float:
        return CAMERA_GROUND_OFFSET + float(
            getattr(self.camera, "manual_height_offset", 0.0)
        )

    def _support_height_at_current_position(self) -> float:
        height_at = getattr(self.scene, "ground_height_at", None)
        if callable(height_at):
            ground_height = height_at(self.camera.position.x, self.camera.position.z)
        else:
            ground = getattr(self.scene, "ground_mesh", None)
            sampler = getattr(ground, "height_sampler", None)
            if sampler is None or not hasattr(sampler, "height_at"):
                return 0.0
            ground_height = sampler.height_at(self.camera.position.x, self.camera.position.z)

        player_radius = float(getattr(self.scene, "player_radius", PLAYER_RADIUS))
        wall_support = player_support_height_at(
            self._collision_meshes_at(
                self.camera.position.x,
                self.camera.position.z,
                player_radius,
                include_polygons=False,
            ),
            self.camera.position.x,
            self.camera.position.z,
            self.camera.position.y - self._eye_to_foot_offset(),
            player_radius,
        )
        if wall_support is None:
            return float(ground_height)
        return max(float(ground_height), float(wall_support))

    def _collision_meshes_at(
        self,
        x: float,
        z: float,
        radius: float,
        *,
        include_polygons: bool,
    ):
        get_candidates = getattr(self.scene, "collision_meshes_at", None)
        if callable(get_candidates):
            return get_candidates(
                x,
                z,
                radius,
                include_polygons=include_polygons,
            )
        if include_polygons:
            return (getattr(self.scene, "wall_tiles", None) or []) + (
                getattr(self.scene, "polygons", None) or []
            )
        return getattr(self.scene, "wall_tiles", None) or []

    def _collision_meshes_for_movement(
        self,
        old_position: Vector3,
        new_position: Vector3,
        player_radius: float,
    ):
        get_candidates = getattr(self.scene, "collision_meshes_for_bounds", None)
        if callable(get_candidates):
            return get_candidates(
                min(old_position.x, new_position.x) - player_radius,
                max(old_position.x, new_position.x) + player_radius,
                min(old_position.z, new_position.z) - player_radius,
                max(old_position.z, new_position.z) + player_radius,
                include_polygons=True,
            )
        return (getattr(self.scene, "wall_tiles", None) or []) + (
            getattr(self.scene, "polygons", None) or []
        )

    def _attempt_boundary_slide(self, old_position: Vector3) -> bool:
        """If current camera position is outside playable area, try axis-aligned
        slides (X-only, then Z-only). If neither works, revert to old_position.

        Returns True if the outside condition was handled (either slid or
        reverted). Returns False if the camera position was already inside.
        """
        if self.scene.contains_horizontal(self.camera.position):
            return False

        attempted = self.camera.position - old_position
        # Try sliding only along X (keep old Z), then only along Z.
        pos_x = Vector3(
            old_position.x + attempted.x,
            old_position.y + attempted.y,
            old_position.z,
        )
        if self.scene.contains_horizontal(pos_x):
            self.camera.position = pos_x
            return True

        pos_z = Vector3(
            old_position.x,
            old_position.y + attempted.y,
            old_position.z + attempted.z,
        )
        if self.scene.contains_horizontal(pos_z):
            self.camera.position = pos_z
            return True

        # Couldn't slide along axis; revert.
        self.camera.position = old_position
        return True

    def _attempt_wall_slide(self, old_position: Vector3) -> bool:

        #meshes = getattr(self.scene, "static_meshes", None) or []
        # Try to handle multiple, sequential wall collisions (e.g. at a 90deg
        # corner) by repeatedly querying for a blocking plane and projecting
        # the attempted movement onto that plane. Limit iterations to avoid
        # pathological loops.
        slid = False
        max_iters = 3
        eps = 1e-6
        player_radius = float(getattr(self.scene, "player_radius", PLAYER_RADIUS))
        col_meshes = self._collision_meshes_for_movement(
            old_position,
            self.camera.position,
            player_radius,
        )
        foot_offset = self._eye_to_foot_offset()
        player_bottom_y = min(old_position.y, self.camera.position.y) - foot_offset
        player_top_y = (
            max(old_position.y, self.camera.position.y)
            + float(getattr(self.scene, "player_head_clearance", PLAYER_HEAD_CLEARANCE))
        )
        #filter out WorldSprites
        for _ in range(max_iters):
            col_normal = movement_blocked_by_wall(
                col_meshes,
                old_position,
                self.camera.position,
                player_radius,
                player_bottom_y=player_bottom_y,
                player_top_y=player_top_y,
            )
            if col_normal is None:
                break

            attempted = self.camera.position - old_position
            # v_slide = attempted - (n dot attempted) * n
            n = col_normal
            dot = attempted.dot(n)
            slide = attempted - n * dot
            new_pos = old_position + slide
            # If projection made no meaningful progress, stop to avoid loop.
            if (new_pos - self.camera.position).length_squared() <= eps:
                # Ensure we don't leave the camera stuck inside the wall.
                self.camera.position = old_position
                slid = True
                break

            self.camera.position = new_pos
            slid = True

        return slid

    def _attempt_y_collision(self, old_position: Vector3) -> bool:
        neg_y_buff = self._eye_to_foot_offset()
        ground_height = self._support_height_at_current_position()

        min_y = ground_height + neg_y_buff
        if self.camera.position.y < min_y:
            self.camera.position.y = min_y
            return True

        return False

    def on_mouse_delta(self, dx: float, dy: float, dt: float | None = None) -> None:
        """Accept raw mouse delta and update rotation targets.

        If a frame delta `dt` is provided, normalize the raw per-frame deltas
        to a baseline 60Hz so mouse look feels consistent across varying FPS.
        """

        # Accept normalized (or raw) deltas
        sensitivity = float(getattr(self.scene, "mouse_sensitivity", MOUSE_SENSITIVITY))
        self.rot_target_y -= dx * sensitivity
        cand_x = self.rot_target_x - dy * sensitivity
        self.rot_target_x = max(-math.pi / 2 + 0.001, min(math.pi / 2 - 0.001, cand_x))
        # Forward mouse delta into sway controller if present
        sc = getattr(self.scene, "_sway_controller", None)
        if sc is not None:
            sc.on_mouse_delta(dx, dy)
        # Notify headbob controller that mouse moved so idle timer resets
        hb = getattr(self.scene, "_headbob", None)
        if hb is not None:
            # Only treat as mouse activity if there is an actual delta. Engine calls
            # on_mouse_delta each frame with (0,0) when mouse is still, which
            # would prevent idle from ever triggering.
            if dx != 0 or dy != 0:
                hb.notify_mouse_moved()

    def update(self, dt: float) -> Tuple[bool, bool]:
        """Update rotation smoothing, movement, and collision.

        Returns (moving, sprinting) booleans for callers (e.g., headbob).
        """
        # Rotation smoothing toward targets
        rotation_changed = False
        if self.rot_smooth_hz <= 0 or dt <= 0:
            if (
                self.camera.rotation.x != self.rot_target_x
                or self.camera.rotation.y != self.rot_target_y
            ):
                self.camera.rotation.x = self.rot_target_x
                self.camera.rotation.y = self.rot_target_y
                rotation_changed = True
        else:
            delta_x = self.rot_target_x - self.camera.rotation.x
            delta_y = self.rot_target_y - self.camera.rotation.y
            if abs(delta_x) > 1e-8 or abs(delta_y) > 1e-8:
                alpha = 1.0 - math.exp(-self.rot_smooth_hz * dt)
                self.camera.rotation.x += delta_x * alpha
                self.camera.rotation.y += delta_y * alpha
                rotation_changed = True
        if rotation_changed:
            self.camera.update_rotation(dt)

        # Read input and handle movement
        keys = pygame.key.get_pressed()
        # pygame ScancodeWrapper is not iterable in some builds; check relevant
        # keys explicitly instead of using any(keys).
        moving = bool(
            keys[pygame.K_w]
            or keys[pygame.K_a]
            or keys[pygame.K_s]
            or keys[pygame.K_d]
        )
        sprinting = bool(keys[pygame.K_LSHIFT])
        jump_pressed = bool(keys[pygame.K_SPACE])
        any_key_down = bool(moving or sprinting or jump_pressed)
        # Inform headbob controller about keyboard activity so it can manage idle sway
        hb = getattr(self.scene, "_headbob", None)
        if hb is not None:
            hb.notify_input_active(any_key_down)

        if not moving and not jump_pressed:
            return False, sprinting

        road_multi = float(getattr(self.scene, "road_speed_multiplier", 1.5))
        base_speed = float(getattr(self.scene, "walk_speed", BASE_SPEED))
        sprint_speed = float(getattr(self.scene, "sprint_speed", SPRINT_SPEED))
        speed = sprint_speed if sprinting else base_speed
        if moving and self.scene.is_on_road(self.camera.position.x, self.camera.position.z):
            speed *= road_multi
        # speed is in world units/sec; Camera.move_camera applies dt internally

        if jump_pressed and not self.camera.is_jumping:
            ground_y = self._support_height_at_current_position()
            desired_ground_y = (
                ground_y + self._eye_to_foot_offset()
            )
            if self.camera.position.y <= desired_ground_y + 0.1:
                self.camera.is_jumping = True
                self.camera.vertical_velocity = float(getattr(self.scene, "jump_speed", JUMP_SPEED))

        old_position = self.camera.position.copy()
        if moving:
            self.camera.move_camera(keys, speed, dt)
            if self._attempt_boundary_slide(old_position):
                pass
            else:
                self._attempt_wall_slide(old_position)

        self._attempt_y_collision(old_position)

        return moving, sprinting


CameraController = PlayerCameraController

__all__ = ["PlayerCameraController", "CameraController"]
