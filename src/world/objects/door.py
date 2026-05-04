"""Interactive door entity backed by a fixed world-space textured slab."""

from __future__ import annotations

import math
from typing import Any, Callable

from pygame.math import Vector3
from OpenGL.GL import (
    GL_ALPHA_TEST,
    GL_BLEND,
    GL_GREATER,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    glAlphaFunc,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glTexCoord2f,
    glVertex3f,
)

from engine.entity import Entity
from engine.rendering.lighting import (
    INDOOR_LIGHT_FACTOR,
    INDOOR_NORMAL,
    sunlight_factor_for_normal,
)
from textures.resource_path import DOOR_TEXTURE_PATH
from textures.texture_utils import load_texture


DOOR_INTERACTION_DISTANCE = 95.0
DOOR_SWING_RADIANS = math.radians(90.0)
DOOR_SWING_SPEED = 3.8
DOOR_THICKNESS = 4.0
DOOR_EDGE_UV_FRACTION = 0.08
DOOR_FRAME_OVERLAP = .5
DOOR_TEXTURE_WIDTH = 32.0
DOOR_TEXTURE_HEIGHT = 48.0
DOOR_TEXTURE_ASPECT = DOOR_TEXTURE_WIDTH / DOOR_TEXTURE_HEIGHT
DOOR_DEFAULT_HEIGHT = 48.0
DOOR_DEFAULT_WIDTH = DOOR_DEFAULT_HEIGHT * DOOR_TEXTURE_ASPECT
DOOR_EDGE_SHADE = 0.72
DOOR_TOP_SHADE = 0.82
DOOR_BOTTOM_SHADE = 0.48

_SIDE_NORMALS = {
    "north": Vector3(0.0, 0.0, 1.0),
    "east": Vector3(1.0, 0.0, 0.0),
    "south": Vector3(0.0, 0.0, -1.0),
    "west": Vector3(-1.0, 0.0, 0.0),
}


def _normal_for_side(side: str) -> Vector3:
    return _SIDE_NORMALS.get(str(side).lower(), _SIDE_NORMALS["south"]).copy()


def _door_tangent(normal: Vector3) -> Vector3:
    tangent = Vector3(-normal.z, 0.0, normal.x)
    if tangent.length_squared() <= 1e-8:
        return Vector3(1.0, 0.0, 0.0)
    return tangent.normalize()


def _smooth01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


class Door(Entity):
    """A toggled door with hinge animation and collision when closed."""

    TEXTURE_ASPECT = DOOR_TEXTURE_ASPECT
    DEFAULT_WIDTH = DOOR_DEFAULT_WIDTH
    DEFAULT_HEIGHT = DOOR_DEFAULT_HEIGHT

    faces = [
        (0, 1, 2, 3),  # front
        (5, 4, 7, 6),  # back
        (4, 0, 3, 7),  # hinge edge
        (1, 5, 6, 2),  # latch edge
        (3, 2, 6, 7),  # top
        (4, 5, 1, 0),  # bottom
    ]

    def __init__(
        self,
        position: Vector3,
        *,
        camera: object,
        texture: Any,
        lighting: Any | None = None,
        sun_direction: Any | None = None,
        width: float = DOOR_DEFAULT_WIDTH,
        height: float = DOOR_DEFAULT_HEIGHT,
        side: str = "south",
        thickness: float = DOOR_THICKNESS,
        collision_width: float | None = None,
        collision_height: float | None = None,
        collision_thickness: float | None = None,
        swing_radians: float = DOOR_SWING_RADIANS,
        swing_speed: float = DOOR_SWING_SPEED,
        interaction_distance: float = DOOR_INTERACTION_DISTANCE,
    ) -> None:
        super().__init__(position=position.copy())
        self.closed_center = position.copy()
        self.side = str(side).lower()
        self.normal = _normal_for_side(self.side)
        self.tangent = _door_tangent(self.normal)
        self.open_normal = -self.normal
        self.width = max(1.0, float(width))
        self.height = max(1.0, float(height))
        self.thickness = max(0.1, float(thickness))
        self.collision_width = max(1.0, float(collision_width or self.width))
        self.collision_height = max(1.0, float(collision_height or self.height))
        self.collision_thickness = max(
            0.1,
            float(collision_thickness or self.thickness),
        )
        self.swing_radians = max(0.0, float(swing_radians))
        self.swing_speed = max(0.0, float(swing_speed))
        self.interaction_distance = max(0.0, float(interaction_distance))
        self.open_amount = 0.0
        self.target_open = False
        self.collision_enabled = True
        self.texture = int(getattr(texture, "texture", texture) or 0)
        self.uv_rect = getattr(texture, "uv_rect", (0.0, 0.0, 1.0, 1.0))
        self.lighting = lighting
        self.sun_direction = sun_direction
        self._doorway_light_region: dict[str, Any] | None = None
        self._doorway_light_open_factor = 1.0
        self._doorway_light_closed_factor: float | None = None
        self._last_doorway_light_factor: float | None = None
        self.hinge = self.closed_center - self.tangent * (self.width * 0.5)
        self.panel_axis = self.tangent.copy()
        self.depth_axis = self.normal.copy()
        self._bounds_cache: tuple[float, float, float, float] | None = None
        self._sync_visual()

    @classmethod
    def texture_or_load(cls, texture: Any | None = None) -> Any:
        if texture:
            return texture
        return load_texture(DOOR_TEXTURE_PATH)

    @classmethod
    def from_building_spec(
        cls,
        spec: dict,
        *,
        texture: Any,
        camera: object,
        ground_height_at: Callable[[float, float], float],
        lighting: Any | None = None,
        sun_direction: Any | None = None,
        wall_thickness: float = 2.5,
    ) -> "Door":
        side = str(spec.get("doorway_side", "south")).lower()
        wall_thickness = float(spec.get("wall_thickness", wall_thickness))
        normal = _normal_for_side(side)
        center = spec["position"]
        half_x = float(spec["width"]) * 0.5
        half_z = float(spec["depth"]) * 0.5
        wall_half = max(0.0, float(wall_thickness)) * 0.5

        x = float(center.x)
        z = float(center.z)
        if abs(normal.x) > 0.0:
            x += normal.x * max(0.0, half_x - wall_half)
        else:
            z += normal.z * max(0.0, half_z - wall_half)

        doorway_height_value = spec.get("doorway_height", None)
        doorway_width_value = spec.get("doorway_width", None)
        if doorway_height_value is not None:
            doorway_height = max(8.0, float(doorway_height_value))
            doorway_width = doorway_height * DOOR_TEXTURE_ASPECT
        elif doorway_width_value is not None:
            doorway_width = max(8.0, float(doorway_width_value))
            doorway_height = doorway_width / DOOR_TEXTURE_ASPECT
        else:
            doorway_width = DOOR_DEFAULT_WIDTH
            doorway_height = DOOR_DEFAULT_HEIGHT

        visual_width = max(8.0, doorway_width + DOOR_FRAME_OVERLAP * 2.0)
        visual_height = max(8.0, visual_width / DOOR_TEXTURE_ASPECT)
        base_y = spec.get("base_y", None)
        if base_y is None:
            base_y = ground_height_at(float(center.x), float(center.z))
        base_y = float(base_y)
        position = Vector3(x, base_y + visual_height * 0.5, z)

        return cls(
            position,
            camera=camera,
            texture=texture,
            lighting=lighting,
            sun_direction=sun_direction,
            width=visual_width,
            height=visual_height,
            side=side,
            thickness=max(DOOR_THICKNESS, wall_thickness),
            collision_width=visual_width,
            collision_height=visual_height,
            collision_thickness=max(DOOR_THICKNESS, wall_thickness),
        )

    @property
    def is_open(self) -> bool:
        return self.open_amount >= 1.0 - 1e-4 and self.target_open

    def get_collision_meshes(self):
        return (self,)

    def get_interaction_position(self) -> Vector3:
        return self.closed_center

    def open(self) -> None:
        self.target_open = True
        self.collision_enabled = False

    def close(self) -> None:
        self.target_open = False
        self.collision_enabled = False

    def toggle(self) -> None:
        if self.target_open:
            self.close()
        else:
            self.open()

    def interact(self, actor=None, scene=None) -> bool:
        self.toggle()
        return True

    def update(self, dt: float) -> None:
        target = 1.0 if self.target_open else 0.0
        if self.open_amount == target:
            return

        step = self.swing_speed * max(0.0, float(dt))
        if step <= 0.0:
            self.open_amount = target
        elif self.open_amount < target:
            self.open_amount = min(target, self.open_amount + step)
        else:
            self.open_amount = max(target, self.open_amount - step)

        if not self.target_open and self.open_amount <= 0.0:
            self.collision_enabled = True

        self._bounds_cache = None
        self._sync_visual()
        self._sync_doorway_light()

    def _sync_visual(self) -> None:
        amount = _smooth01(self.open_amount)
        angle = self.swing_radians * amount
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        half_width = self.width * 0.5
        self.panel_axis = self.tangent * cos_a + self.open_normal * sin_a
        self.depth_axis = self.normal * cos_a + self.tangent * sin_a
        self.position = self.hinge + self.panel_axis * half_width

    def bind_doorway_light(self, region: object) -> None:
        if not isinstance(region, dict):
            return

        doorway = region.get("doorway")
        if not isinstance(doorway, dict):
            return

        self._doorway_light_region = region
        self._doorway_light_open_factor = max(
            0.0,
            min(1.0, float(doorway.get("open_edge_factor", 1.0))),
        )
        closed_factor = doorway.get("closed_edge_factor", region.get("factor"))
        try:
            self._doorway_light_closed_factor = max(0.0, min(1.0, float(closed_factor)))
        except (TypeError, ValueError):
            self._doorway_light_closed_factor = None
        self._sync_doorway_light(force=True)

    def _sync_doorway_light(self, *, force: bool = False) -> None:
        region = self._doorway_light_region
        if not isinstance(region, dict):
            return

        doorway = region.get("doorway")
        if not isinstance(doorway, dict):
            return

        try:
            region_factor = max(
                0.0,
                min(1.0, float(region.get("factor", INDOOR_LIGHT_FACTOR))),
            )
        except (TypeError, ValueError):
            region_factor = INDOOR_LIGHT_FACTOR
        closed_factor = (
            region_factor
            if self._doorway_light_closed_factor is None
            else self._doorway_light_closed_factor
        )
        amount = _smooth01(self.open_amount)
        edge_factor = closed_factor + (
            self._doorway_light_open_factor - closed_factor
        ) * amount
        if (
            not force
            and self._last_doorway_light_factor is not None
            and abs(edge_factor - self._last_doorway_light_factor) <= 1e-4
        ):
            return

        doorway["edge_factor"] = edge_factor
        self._last_doorway_light_factor = edge_factor

    def _face_normal(self, face_idx: int) -> Vector3:
        if face_idx == 0:
            return self.depth_axis
        if face_idx == 1:
            return -self.depth_axis
        if face_idx == 2:
            return -self.panel_axis
        if face_idx == 3:
            return self.panel_axis
        if face_idx == 4:
            return Vector3(0.0, 1.0, 0.0)
        return Vector3(0.0, -1.0, 0.0)

    def _sunlight_factor(self, normal) -> float:
        lighting = getattr(self, "lighting", None)
        sun_direction = getattr(
            lighting,
            "sun_direction",
            getattr(self, "sun_direction", None),
        )
        if lighting is None and sun_direction is None:
            return 1.0
        return sunlight_factor_for_normal(
            normal,
            lighting=lighting,
            sun_direction=sun_direction,
        )

    def _face_shade(self, face_idx: int) -> float:
        if face_idx == 0:
            base = 1.0
            normal = self._face_normal(face_idx)
        elif face_idx == 1:
            base = INDOOR_LIGHT_FACTOR
            normal = INDOOR_NORMAL
        elif face_idx == 4:
            base = DOOR_TOP_SHADE
            normal = self._face_normal(face_idx)
        elif face_idx == 5:
            base = DOOR_BOTTOM_SHADE
            normal = self._face_normal(face_idx)
        else:
            base = DOOR_EDGE_SHADE
            normal = self._face_normal(face_idx)
        return max(0.0, min(1.0, base * self._sunlight_factor(normal)))

    def _visual_vertices(self) -> list[Vector3]:
        return self._box_vertices(
            self.position,
            self.panel_axis,
            self.depth_axis,
            width=self.width,
            height=self.height,
            thickness=self.thickness,
        )

    @staticmethod
    def _box_vertices(
        center: Vector3,
        width_axis: Vector3,
        depth_axis: Vector3,
        *,
        width: float,
        height: float,
        thickness: float,
    ) -> list[Vector3]:
        half_w = float(width) * 0.5
        half_h = float(height) * 0.5
        half_t = float(thickness) * 0.5
        up = Vector3(0.0, 1.0, 0.0)
        width_offset = width_axis * half_w
        height_offset = up * half_h
        depth_offset = depth_axis * half_t

        return [
            center - width_offset - height_offset + depth_offset,
            center + width_offset - height_offset + depth_offset,
            center + width_offset + height_offset + depth_offset,
            center - width_offset + height_offset + depth_offset,
            center - width_offset - height_offset - depth_offset,
            center + width_offset - height_offset - depth_offset,
            center + width_offset + height_offset - depth_offset,
            center - width_offset + height_offset - depth_offset,
        ]

    def _face_uvs(self, face_idx: int) -> list[tuple[float, float]]:
        u0, v0, u1, v1 = self.uv_rect
        u_span = u1 - u0
        v_span = v1 - v0
        strip_u = u_span * DOOR_EDGE_UV_FRACTION
        strip_v = v_span * DOOR_EDGE_UV_FRACTION

        if face_idx == 0:
            return [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]
        if face_idx == 1:
            return [(u1, v1), (u0, v1), (u0, v0), (u1, v0)]
        if face_idx == 2:
            return [(u0, v1), (u0 + strip_u, v1), (u0 + strip_u, v0), (u0, v0)]
        if face_idx == 3:
            return [(u1 - strip_u, v1), (u1, v1), (u1, v0), (u1 - strip_u, v0)]
        if face_idx == 4:
            return [(u0, v0 + strip_v), (u1, v0 + strip_v), (u1, v0), (u0, v0)]
        return [(u0, v1), (u1, v1), (u1, v1 - strip_v), (u0, v1 - strip_v)]

    def draw(self, camera=None) -> None:  # pragma: no cover - visual
        if not self.visible or not self.texture:
            return

        verts = self._visual_vertices()
        if not verts:
            return

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glEnable(GL_ALPHA_TEST)
        glAlphaFunc(GL_GREATER, 0.01)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        glBegin(GL_QUADS)
        try:
            for face_idx, face in enumerate(self.faces):
                shade = self._face_shade(face_idx)
                glColor4f(shade, shade, shade, 1.0)
                for vertex_idx, uv in zip(face, self._face_uvs(face_idx)):
                    vertex = verts[vertex_idx]
                    glTexCoord2f(uv[0], uv[1])
                    glVertex3f(vertex.x, vertex.y, vertex.z)
        finally:
            glEnd()
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glDisable(GL_ALPHA_TEST)
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)

    def get_world_vertices(self):
        if not self.collision_enabled:
            return []

        return self._box_vertices(
            self.closed_center,
            self.tangent,
            self.normal,
            width=self.collision_width,
            height=self.collision_height,
            thickness=self.collision_thickness,
        )

    def get_bounding_box(self):
        if not self.collision_enabled:
            return None
        if self._bounds_cache is not None:
            return self._bounds_cache

        verts = self.get_world_vertices()
        if not verts:
            return None

        min_x = min(v.x for v in verts)
        max_x = max(v.x for v in verts)
        min_z = min(v.z for v in verts)
        max_z = max(v.z for v in verts)
        self._bounds_cache = (min_x, max_x, min_z, max_z)
        return self._bounds_cache
