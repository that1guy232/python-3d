import math
import numpy as np
import pygame
from pygame.math import Vector3
from config import WIDTH, HEIGHT


class Camera:
    def __init__(self, position=None, rotation=None, width=800, height=600, fov=75):
        # keep external API types the same (pygame.Vector3)
        self.position = position or Vector3(0, 0, -300)
        self.rotation = rotation or Vector3(0, 0, 0)  # pitch (x), yaw (y), roll (z)
        self.speed = 5
        self.world_up = Vector3(0, 1, 0)

        # FOV scale (same formula you used)
        self._fov_scale = (height / 2) / math.tan(math.radians(fov / 2))

        # cached trig & direction vectors (kept as Vector3 where movement uses them)
        self._yaw_cos = 1.0
        self._yaw_sin = 0.0
        self._pitch_cos = 1.0
        self._pitch_sin = 0.0
        self._forward = Vector3(0, 0, -1)
        self._right = Vector3(1, 0, 0)
        self._ground_forward = -Vector3(0, 0, -1)

        # NumPy rotation matrix (world -> camera)
        self._R = np.eye(3, dtype=np.float64)

        # Manual height offset (adjustable by user for screenshots, added on top of
        # the terrain-following camera Y). WorldScene will add this when computing
        # the target camera height.
        self.manual_height_offset = 0.0

        # Speed used when adjusting manual height offset with Q/E (world units/sec).
        # Chosen to be relatively small so Q/E can be used for fine screenshot tweaks.
        self.height_adjust_speed = 50.0

        # initialize
        self.update_rotation()

    def update_rotation(self):
        """Precompute trig, direction vectors (Vector3) and rotation matrix (numpy)."""
        cp = math.cos(self.rotation.x)
        sp = math.sin(self.rotation.x)
        cy = math.cos(self.rotation.y)
        sy = math.sin(self.rotation.y)

        self._yaw_cos = cy
        self._yaw_sin = sy
        self._pitch_cos = cp
        self._pitch_sin = sp

        # Right vector (matches your previous expression)
        self._right = Vector3(self._yaw_cos, 0, -self._yaw_sin)

        # Full forward vector (with vertical component when pitched)
        self._forward = Vector3(
            -self._pitch_cos * self._yaw_sin,
            self._pitch_sin,
            -self._pitch_cos * self._yaw_cos,
        )

        # Ground forward (horizontal component only — keep same formula you used)
        self._ground_forward = -Vector3(-self._yaw_sin, 0, -self._yaw_cos)

        # Build rotation matrices with the same yaw-then-pitch order as your original transform:
        # yaw (around Y) then pitch (around X). Apply yaw first, then pitch.
        Ry = np.array(
            [[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]], dtype=np.float64
        )

        Rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64
        )

        # world -> camera matrix: pitch * yaw (so application is R @ vector)
        self._R = Rx @ Ry

    def transform_point(self, point):
        """
        Drop-in replacement for your original single-point transform.
        Accepts a pygame.Vector3 `point`. Returns (int_x, int_y) or None for clipped.
        Uses the precomputed numpy rotation matrix for speed/clarity.
        """
        # translate into camera-relative coordinates (world -> camera via matrix)
        rel = np.array(
            [
                point.x - self.position.x,
                point.y - self.position.y,
                point.z - self.position.z,
            ],
            dtype=np.float64,
        )

        # apply rotation matrix (yaw then pitch)
        cam = self._R @ rel  # cam[0]=x_cam, cam[1]=y_cam, cam[2]=z_cam

        z = cam[2]
        if z <= 1e-4:
            return None

        factor = self._fov_scale / z
        screen_x = int(cam[0] * factor + WIDTH // 2)
        screen_y = int(HEIGHT // 2 - cam[1] * factor)
        return (screen_x, screen_y)

    def move_camera(self, keys, speed, dt):
        """Movement using precomputed direction vectors (keeps same calls/semantics)."""
        # keys is expected to be pygame.key.get_pressed() or similar mapping
        if keys[pygame.K_w]:
            self.position -= self._ground_forward * speed * dt
        if keys[pygame.K_s]:
            self.position += self._ground_forward * speed * dt
        if keys[pygame.K_a]:
            self.position -= self._right * speed * dt
        if keys[pygame.K_d]:
            self.position += self._right * speed * dt

        # Q/E adjust manual height offset (used on top of terrain-follow Y).
        # Use a dedicated, smaller speed for fine control instead of the movement speed.
        if keys[pygame.K_q]:
            self.manual_height_offset += self.height_adjust_speed * dt
        if keys[pygame.K_e]:
            self.manual_height_offset -= self.height_adjust_speed * dt

        return self.position

    def is_visible(self, obj):
        # as long as 1 vertex is in front of the near plane, the object is visible
        for v in obj.get_world_vertices():
            if self.is_point_visible(v):
                return True
        return False

    def is_point_visible(self, point):
        point = self.transform_point(point)
        if point is None:
            return False
        # Check if the point is within the screen bounds
        x, y = point
        return True
