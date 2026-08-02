"""Interactive opening chest entity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pygame.math import Vector3

from engine.core.mesh import BatchedMesh
from engine.entity import Entity
from engine.rendering.lighting import sunlight_factor_for_normal
from engine.textures.texture_utils import load_texture
from game.resources.paths import CHEST_TEXTURE_PATH
from game.world.lighting_receivers import (
    CPU_BAKED_OBJECT_LIGHTING_RECEIVER,
    DYNAMIC_OBJECT_LIGHTING_RECEIVER,
)

from .slab import (
    SLAB_BOX_FACES,
    normal_for_side,
    signed_wall_tangent_for_normal,
    sphere_for_vertices,
    texture_id,
    texture_uv_rect,
    xz_bounds_for_vertices,
)

CHEST_INTERACTION_DISTANCE = 95.0
CHEST_OPEN_RADIANS = math.radians(82.0)
CHEST_OPEN_SPEED = 3.6
CHEST_DEFAULT_WIDTH = 52.0
CHEST_DEFAULT_DEPTH = 34.0
CHEST_DEFAULT_BODY_HEIGHT = 22.0
CHEST_DEFAULT_LID_HEIGHT = 14.0
CHEST_WALL_THICKNESS = 2.5
CHEST_LID_OVERHANG = 2.0
CHEST_LID_SEGMENTS = 7
CHEST_FRAME_THICKNESS = 3.2
CHEST_LID_SHELL_THICKNESS = 2.4
CHEST_LID_BAND_WIDTH = 3.4
CHEST_FRONT_SHADE = 0.96
CHEST_BACK_SHADE = 0.82
CHEST_EDGE_SHADE = 0.74
CHEST_TOP_SHADE = 0.94
CHEST_BOTTOM_SHADE = 0.48

# The texture is a tiny grayscale atlas.  The left twelve columns provide
# pixel-art wood grain; the right four are neutral so vertex colors can create
# iron, gold, and dark recessed details without inheriting the wood pattern.
CHEST_WOOD_UV_RECT = (0.5 / 16.0, 0.5 / 8.0, 11.5 / 16.0, 7.5 / 8.0)
CHEST_SOLID_UV_RECT = (
    13.5 / 16.0,
    4.5 / 8.0,
    13.5 / 16.0,
    4.5 / 8.0,
)

CHEST_WOOD_COLORS = (
    (1.00, 0.58, 0.27),
    (0.90, 0.48, 0.21),
    (1.08, 0.64, 0.31),
)
CHEST_WOOD_DARK = (0.52, 0.25, 0.10)
CHEST_INTERIOR_COLOR = (0.22, 0.10, 0.035)
CHEST_IRON_COLOR = (0.54, 0.61, 0.72)
CHEST_IRON_DARK = (0.31, 0.36, 0.44)
CHEST_GOLD_COLOR = (1.34, 1.00, 0.16)
CHEST_KEYHOLE_COLOR = (0.13, 0.10, 0.055)
CHEST_ALL_FACES = (0, 1, 2, 3, 4, 5)
CHEST_SHELL_FACES = (0, 1, 4, 5)
CHEST_WALL_FACES = (0, 1, 4)
CHEST_FLOOR_FACES = (4,)

ChestColor = tuple[float, float, float]
UvRect = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _ChestPart:
    """One colored, textured oriented box in the assembled chest."""

    center: Vector3
    width_axis: Vector3
    height_axis: Vector3
    depth_axis: Vector3
    width: float
    height: float
    depth: float
    color: ChestColor
    uv_rect: UvRect = CHEST_SOLID_UV_RECT
    face_indices: tuple[int, ...] = CHEST_ALL_FACES

    def box_spec(
        self,
    ) -> tuple[Vector3, Vector3, Vector3, Vector3, float, float, float]:
        return (
            self.center,
            self.width_axis,
            self.height_axis,
            self.depth_axis,
            self.width,
            self.height,
            self.depth,
        )


def _smooth01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _normalized(vector: Vector3, fallback: Vector3) -> Vector3:
    if vector.length_squared() <= 1e-8:
        return fallback.copy()
    return vector.normalize()


def _rotate_around_axis(vector: Vector3, axis: Vector3, angle: float) -> Vector3:
    axis = _normalized(axis, Vector3(1.0, 0.0, 0.0))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        vector * cos_a
        + axis.cross(vector) * sin_a
        + axis * (axis.dot(vector) * (1.0 - cos_a))
    )


def _oriented_box_vertices(
    center: Vector3,
    width_axis: Vector3,
    height_axis: Vector3,
    depth_axis: Vector3,
    *,
    width: float,
    height: float,
    depth: float,
) -> list[Vector3]:
    half_w = float(width) * 0.5
    half_h = float(height) * 0.5
    half_d = float(depth) * 0.5
    width_offset = width_axis * half_w
    height_offset = height_axis * half_h
    depth_offset = depth_axis * half_d

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


class Chest(Entity):
    """A stylized wooden coffer with a hinged, barrel-vaulted lid."""

    lighting_receiver = CPU_BAKED_OBJECT_LIGHTING_RECEIVER
    packet_lighting_receiver = DYNAMIC_OBJECT_LIGHTING_RECEIVER
    faces = SLAB_BOX_FACES

    def __init__(
        self,
        position: Vector3,
        *,
        texture: Any,
        lighting: Any | None = None,
        sun_direction: Any | None = None,
        width: float = CHEST_DEFAULT_WIDTH,
        depth: float = CHEST_DEFAULT_DEPTH,
        body_height: float = CHEST_DEFAULT_BODY_HEIGHT,
        lid_height: float = CHEST_DEFAULT_LID_HEIGHT,
        side: str = "south",
        wall_thickness: float = CHEST_WALL_THICKNESS,
        open_radians: float = CHEST_OPEN_RADIANS,
        open_speed: float = CHEST_OPEN_SPEED,
        interaction_distance: float = CHEST_INTERACTION_DISTANCE,
    ) -> None:
        base_position = position.copy()
        center = base_position + Vector3(0.0, body_height * 0.5, 0.0)
        super().__init__(position=center)
        self.base_position = base_position
        self.side = str(side).lower()
        self.front_axis = normal_for_side(self.side)
        self.width_axis = signed_wall_tangent_for_normal(self.front_axis)
        self.up_axis = Vector3(0.0, 1.0, 0.0)
        self.width = max(4.0, float(width))
        self.depth = max(4.0, float(depth))
        self.body_height = max(4.0, float(body_height))
        self.lid_height = max(1.0, float(lid_height))
        self.wall_thickness = max(1.0, float(wall_thickness))
        self.open_radians = max(0.0, float(open_radians))
        self.open_speed = max(0.0, float(open_speed))
        self.interaction_distance = max(0.0, float(interaction_distance))
        self.open_amount = 0.0
        self.target_open = False
        self.runtime_update_enabled = False
        self.collision_enabled = True
        self.texture = texture_id(texture)
        self.uv_rect = texture_uv_rect(texture)
        self.lighting = lighting
        self.sun_direction = None if lighting is not None else sun_direction
        self._mesh: BatchedMesh | None = None
        self._mesh_key = None
        self._bounds_cache: tuple[float, float, float, float] | None = None
        self._body_parts_cache: tuple[_ChestPart, ...] | None = None
        self._lid_parts_cache_key: float | None = None
        self._lid_parts_cache: tuple[_ChestPart, ...] | None = None
        self._visual_parts_cache_key: float | None = None
        self._visual_parts_cache: tuple[_ChestPart, ...] | None = None
        self._visual_vertices_cache_key: float | None = None
        self._visual_vertices_cache: list[Vector3] | None = None
        self._render_sphere_cache_key: float | None = None
        self._render_sphere_cache = None
        self._body_vertex_data_cache: dict[tuple[Any, ...], np.ndarray] = {}

    @classmethod
    def texture_or_load(cls, texture: Any | None = None) -> Any:
        if texture:
            return texture
        return load_texture(CHEST_TEXTURE_PATH)

    @property
    def is_open(self) -> bool:
        return self.open_amount >= 1.0 - 1e-4 and self.target_open

    def get_interaction_position(self) -> Vector3:
        return self.base_position + Vector3(0.0, self.body_height * 0.6, 0.0)

    def open(self) -> None:
        self.target_open = True
        self.runtime_update_enabled = self.open_amount < 1.0 - 1e-4

    def close(self) -> None:
        self.target_open = False
        self.runtime_update_enabled = self.open_amount > 1e-4

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
            self.runtime_update_enabled = False
            return

        step = self.open_speed * max(0.0, float(dt))
        if step <= 0.0:
            self.open_amount = target
        elif self.open_amount < target:
            self.open_amount = min(target, self.open_amount + step)
        else:
            self.open_amount = max(target, self.open_amount - step)

        if abs(self.open_amount - target) <= 1e-4:
            self.open_amount = target
            self.runtime_update_enabled = False

        self._mark_mesh_dirty()

    def dispose(self) -> None:
        self._dispose_mesh()

    def _dispose_mesh(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.dispose()
            except Exception:
                pass
        self._mesh = None
        self._mesh_key = None

    def _mark_mesh_dirty(self) -> None:
        self._mesh_key = None
        self._lid_parts_cache_key = None
        self._lid_parts_cache = None
        self._visual_parts_cache_key = None
        self._visual_parts_cache = None
        self._visual_vertices_cache_key = None
        self._visual_vertices_cache = None
        self._render_sphere_cache_key = None
        self._render_sphere_cache = None

    def _geometry_pose_key(self) -> float:
        return round(float(self.open_amount), 5)

    def _sunlight_factor(self, normal: Vector3) -> float:
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

    def _light_cache_key(self):
        lighting = getattr(self, "lighting", None)
        sun_direction = getattr(
            lighting,
            "sun_direction",
            getattr(self, "sun_direction", None),
        )
        try:
            sun_key = (
                round(float(sun_direction.x), 6),
                round(float(sun_direction.y), 6),
                round(float(sun_direction.z), 6),
            )
        except Exception:
            sun_key = None

        if lighting is None:
            return sun_key

        return (
            sun_key,
            round(float(getattr(lighting, "ambient", 0.72)), 6),
            round(float(getattr(lighting, "diffuse", 0.48)), 6),
            round(float(getattr(lighting, "max_factor", 1.15)), 6),
        )

    def _face_normal(
        self,
        face_idx: int,
        width_axis: Vector3,
        height_axis: Vector3,
        depth_axis: Vector3,
    ) -> Vector3:
        if face_idx == 0:
            return depth_axis
        if face_idx == 1:
            return -depth_axis
        if face_idx == 2:
            return -width_axis
        if face_idx == 3:
            return width_axis
        if face_idx == 4:
            return height_axis
        return -height_axis

    @staticmethod
    def _face_base_shade(face_idx: int) -> float:
        if face_idx == 0:
            return CHEST_FRONT_SHADE
        if face_idx == 1:
            return CHEST_BACK_SHADE
        if face_idx == 4:
            return CHEST_TOP_SHADE
        if face_idx == 5:
            return CHEST_BOTTOM_SHADE
        return CHEST_EDGE_SHADE

    def _face_shade(
        self,
        face_idx: int,
        width_axis: Vector3,
        height_axis: Vector3,
        depth_axis: Vector3,
    ) -> float:
        normal = self._face_normal(face_idx, width_axis, height_axis, depth_axis)
        base = self._base_shade_for_normal(normal)
        return max(0.0, min(1.0, base * self._sunlight_factor(normal)))

    def _base_shade_for_normal(self, normal: Vector3) -> float:
        """Return hand-painted shading from the actual world-space normal."""

        normal = _normalized(normal, self.up_axis)
        vertical = max(-1.0, min(1.0, float(normal.dot(self.up_axis))))
        vertical_weight = abs(vertical)
        horizontal_length = max(0.0, 1.0 - vertical * vertical) ** 0.5
        if horizontal_length <= 1e-6:
            horizontal_shade = CHEST_EDGE_SHADE
        else:
            front_amount = max(
                -1.0,
                min(1.0, float(normal.dot(self.front_axis)) / horizontal_length),
            )
            if front_amount >= 0.0:
                horizontal_shade = CHEST_EDGE_SHADE + front_amount * (
                    CHEST_FRONT_SHADE - CHEST_EDGE_SHADE
                )
            else:
                horizontal_shade = CHEST_EDGE_SHADE + (-front_amount) * (
                    CHEST_BACK_SHADE - CHEST_EDGE_SHADE
                )

        vertical_shade = CHEST_TOP_SHADE if vertical >= 0.0 else CHEST_BOTTOM_SHADE
        return (
            horizontal_shade * (1.0 - vertical_weight)
            + vertical_shade * vertical_weight
        )

    def _face_uvs(
        self,
        local_rect: UvRect | None = None,
    ) -> tuple[tuple[float, float], ...]:
        local_u0, local_v0, local_u1, local_v1 = local_rect or (
            0.0,
            0.0,
            1.0,
            1.0,
        )
        atlas_u0, atlas_v0, atlas_u1, atlas_v1 = self.uv_rect
        atlas_u_span = atlas_u1 - atlas_u0
        atlas_v_span = atlas_v1 - atlas_v0
        u0 = atlas_u0 + atlas_u_span * local_u0
        v0 = atlas_v0 + atlas_v_span * local_v0
        u1 = atlas_u0 + atlas_u_span * local_u1
        v1 = atlas_v0 + atlas_v_span * local_v1
        return ((u0, v1), (u1, v1), (u1, v0), (u0, v0))

    def _append_box_vertex_data(
        self,
        rows: list[tuple[float, ...]],
        verts: Sequence[Vector3],
        width_axis: Vector3,
        height_axis: Vector3,
        depth_axis: Vector3,
        *,
        color: ChestColor = (1.0, 1.0, 1.0),
        uv_rect: UvRect | None = None,
        face_indices: Sequence[int] = CHEST_ALL_FACES,
        dynamic_lighting: bool = False,
    ) -> None:
        uv_values = self._face_uvs(uv_rect)
        for face_idx in face_indices:
            face = self.faces[face_idx]
            normal = self._face_normal(
                face_idx,
                width_axis,
                height_axis,
                depth_axis,
            )
            shade = self._base_shade_for_normal(normal)
            if not dynamic_lighting:
                shade = max(
                    0.0,
                    min(1.0, shade * self._sunlight_factor(normal)),
                )
            red, green, blue = (
                max(0.0, float(channel) * shade) for channel in color
            )
            for vertex_idx, uv in (
                (face[0], uv_values[0]),
                (face[1], uv_values[1]),
                (face[2], uv_values[2]),
                (face[0], uv_values[0]),
                (face[2], uv_values[2]),
                (face[3], uv_values[3]),
            ):
                vertex = verts[vertex_idx]
                row = (
                    vertex.x,
                    vertex.y,
                    vertex.z,
                    red,
                    green,
                    blue,
                )
                if dynamic_lighting:
                    row += (normal.x, normal.y, normal.z)
                rows.append(row + (uv[0], uv[1]))

    def _body_boxes(self):
        thickness = min(
            self.wall_thickness,
            self.width * 0.45,
            self.depth * 0.45,
            self.body_height * 0.45,
        )
        body_center_y = self.body_height * 0.5
        bottom_center_y = thickness * 0.5
        side_depth = max(1.0, self.depth - thickness * 2.0)
        bottom_width = max(1.0, self.width - thickness * 2.0)
        bottom_depth = max(1.0, self.depth - thickness * 2.0)

        base = self.base_position
        front = self.front_axis
        width_axis = self.width_axis
        up = self.up_axis

        return (
            (
                base
                + up * body_center_y
                + front * (self.depth * 0.5 - thickness * 0.5),
                width_axis,
                up,
                front,
                self.width,
                self.body_height,
                thickness,
            ),
            (
                base
                + up * body_center_y
                - front * (self.depth * 0.5 - thickness * 0.5),
                width_axis,
                up,
                front,
                self.width,
                self.body_height,
                thickness,
            ),
            (
                base
                + up * body_center_y
                - width_axis * (self.width * 0.5 - thickness * 0.5),
                front,
                up,
                width_axis,
                side_depth,
                self.body_height,
                thickness,
            ),
            (
                base
                + up * body_center_y
                + width_axis * (self.width * 0.5 - thickness * 0.5),
                front,
                up,
                width_axis,
                side_depth,
                self.body_height,
                thickness,
            ),
            (
                base + up * bottom_center_y,
                width_axis,
                up,
                front,
                bottom_width,
                thickness,
                bottom_depth,
            ),
        )

    def _frame_size(self) -> float:
        return max(
            0.8,
            min(
                CHEST_FRAME_THICKNESS,
                self.width * 0.11,
                self.depth * 0.15,
                self.body_height * 0.22,
            ),
        )

    def _lid_shell_size(self) -> float:
        return max(
            0.65,
            min(
                CHEST_LID_SHELL_THICKNESS,
                self.depth * 0.10,
                self.lid_height * 0.24,
            ),
        )

    def _body_parts(self) -> tuple[_ChestPart, ...]:
        if self._body_parts_cache is not None:
            return self._body_parts_cache

        parts: list[_ChestPart] = []
        for index, box in enumerate(self._body_boxes()):
            color = (
                CHEST_INTERIOR_COLOR
                if index == 4
                else CHEST_WOOD_COLORS[index % len(CHEST_WOOD_COLORS)]
            )
            parts.append(
                _ChestPart(
                    *box,
                    color=color,
                    uv_rect=(
                        CHEST_SOLID_UV_RECT
                        if index == 4
                        else CHEST_WOOD_UV_RECT
                    ),
                    face_indices=(
                        CHEST_FLOOR_FACES
                        if index == 4
                        else CHEST_WALL_FACES
                    ),
                )
            )

        base = self.base_position
        front = self.front_axis
        width_axis = self.width_axis
        up = self.up_axis
        frame = self._frame_size()
        rail_depth = frame * 0.78
        rail_lift = rail_depth * 0.18
        surface_gap = max(0.04, min(0.12, frame * 0.025))
        front_rail_width = max(
            frame,
            self.width + rail_lift * 2.0 - frame - surface_gap * 2.0,
        )
        side_rail_width = max(
            frame,
            self.depth + rail_lift * 2.0 - rail_depth - surface_gap * 2.0,
        )

        # Rails stop just short of the posts instead of occupying the same
        # coplanar corner surfaces.  The top rail is also lifted off the wooden
        # shell by a tiny, visible-scale-safe amount.
        rail_heights = (
            frame * 0.5 + surface_gap,
            self.body_height - frame * 0.5 + surface_gap,
        )
        for y in rail_heights:
            for front_sign in (-1.0, 1.0):
                outward = front * front_sign
                parts.append(
                    _ChestPart(
                        base
                        + up * y
                        + outward * (self.depth * 0.5 + rail_lift),
                        width_axis,
                        up,
                        outward,
                        front_rail_width,
                        frame,
                        rail_depth,
                        CHEST_IRON_COLOR,
                    )
                )
            for width_sign in (-1.0, 1.0):
                outward = width_axis * width_sign
                parts.append(
                    _ChestPart(
                        base
                        + up * y
                        + outward * (self.width * 0.5 + rail_lift),
                        front,
                        up,
                        outward,
                        side_rail_width,
                        frame,
                        rail_depth,
                        CHEST_IRON_DARK,
                    )
                )

        post_bottom = frame + surface_gap * 2.0
        post_top = self.body_height - frame
        post_height = max(0.5, post_top - post_bottom)
        post_center_y = (post_bottom + post_top) * 0.5
        for width_sign in (-1.0, 1.0):
            for front_sign in (-1.0, 1.0):
                outward_width = width_axis * width_sign
                outward_front = front * front_sign
                parts.append(
                    _ChestPart(
                        base
                        + up * post_center_y
                        + outward_width * (self.width * 0.5 + rail_lift)
                        + outward_front * (self.depth * 0.5 + rail_lift),
                        outward_width,
                        up,
                        outward_front,
                        frame,
                        post_height,
                        rail_depth,
                        CHEST_IRON_COLOR,
                    )
                )

        # Recessed seams make the lower front read as separate vertical boards.
        groove_height = max(1.0, self.body_height - frame * 2.0)
        groove_depth = max(0.35, frame * 0.16)
        for divider in range(1, 5):
            x = -self.width * 0.5 + self.width * divider / 5.0
            parts.append(
                _ChestPart(
                    base
                    + up * (self.body_height * 0.5)
                    + width_axis * x
                    + front * (self.depth * 0.5 + groove_depth * 0.72),
                    width_axis,
                    up,
                    front,
                    max(0.35, frame * 0.14),
                    groove_height,
                    groove_depth,
                    CHEST_WOOD_DARK,
                )
            )

        # The lower half of the clasp remains fixed when the lid opens.
        latch_depth = max(1.0, frame * 0.55)
        parts.append(
            _ChestPart(
                base
                + up * (self.body_height - frame * 0.95 - surface_gap)
                + front * (self.depth * 0.5 + latch_depth * 0.72),
                width_axis,
                up,
                front,
                frame * 1.65,
                frame * 1.9,
                latch_depth,
                CHEST_GOLD_COLOR,
            )
        )
        parts.append(
            _ChestPart(
                base
                + up * (self.body_height - frame * 1.12)
                + front * (self.depth * 0.5 + latch_depth * 1.28),
                width_axis,
                up,
                front,
                max(0.75, frame * 0.38),
                max(1.3, frame * 0.70),
                max(0.3, frame * 0.16),
                CHEST_KEYHOLE_COLOR,
            )
        )
        self._body_parts_cache = tuple(parts)
        return self._body_parts_cache

    def _lid_pose(
        self,
    ) -> tuple[Vector3, Vector3, Vector3, float, float]:
        amount = _smooth01(self.open_amount)
        angle = self.open_radians * amount
        lid_width = self.width + CHEST_LID_OVERHANG * 2.0
        lid_depth = self.depth + CHEST_LID_OVERHANG * 2.0
        hinge = (
            self.base_position
            + self.up_axis * self.body_height
            - self.front_axis * (lid_depth * 0.5)
        )
        depth_axis = _rotate_around_axis(self.front_axis, self.width_axis, angle)
        height_axis = _rotate_around_axis(self.up_axis, self.width_axis, angle)
        return hinge, depth_axis, height_axis, lid_width, lid_depth

    def _lid_box(self):
        """Return the animated lid's envelope for compatibility/debugging."""

        hinge, depth_axis, height_axis, lid_width, lid_depth = self._lid_pose()
        center = (
            hinge
            + depth_axis * (lid_depth * 0.5)
            + height_axis * (self.lid_height * 0.5)
        )
        return (
            center,
            self.width_axis,
            height_axis,
            depth_axis,
            lid_width,
            self.lid_height,
            lid_depth,
        )

    def _lid_arch_sections(self):
        hinge, depth_axis, height_axis, _, lid_depth = self._lid_pose()
        radius = lid_depth * 0.5
        section_count = max(3, int(CHEST_LID_SEGMENTS))
        sections = []
        for index in range(section_count):
            theta0 = -math.pi * 0.5 + math.pi * index / section_count
            theta1 = -math.pi * 0.5 + math.pi * (index + 1) / section_count
            d0 = radius * (math.sin(theta0) + 1.0)
            d1 = radius * (math.sin(theta1) + 1.0)
            h0 = self.lid_height * math.cos(theta0)
            h1 = self.lid_height * math.cos(theta1)
            edge0 = hinge + depth_axis * d0 + height_axis * h0
            edge1 = hinge + depth_axis * d1 + height_axis * h1
            delta = edge1 - edge0
            length = max(0.1, delta.length())
            tangent = delta / length
            tangent_depth = float(tangent.dot(depth_axis))
            tangent_height = float(tangent.dot(height_axis))
            normal = _normalized(
                depth_axis * (-tangent_height) + height_axis * tangent_depth,
                height_axis,
            )
            sections.append(
                (
                    edge0,
                    edge1,
                    tangent,
                    normal,
                    length,
                    d0,
                    d1,
                    h0,
                    h1,
                )
            )
        return tuple(sections)

    def _lid_parts(self) -> tuple[_ChestPart, ...]:
        pose_key = self._geometry_pose_key()
        if (
            self._lid_parts_cache_key == pose_key
            and self._lid_parts_cache is not None
        ):
            return self._lid_parts_cache

        parts: list[_ChestPart] = []
        hinge, depth_axis, height_axis, lid_width, lid_depth = self._lid_pose()
        shell = self._lid_shell_size()
        frame = self._frame_size()
        band_width = max(
            0.8,
            min(CHEST_LID_BAND_WIDTH, lid_width * 0.12),
        )
        surface_gap = max(0.04, min(0.12, frame * 0.025))
        sections = self._lid_arch_sections()

        # Slightly overlapping tangent boxes form a clean low-poly barrel vault.
        for index, section in enumerate(sections):
            edge0, edge1, tangent, normal, length, *_ = section
            parts.append(
                _ChestPart(
                    (edge0 + edge1) * 0.5 - normal * (shell * 0.5),
                    self.width_axis,
                    tangent,
                    normal,
                    lid_width,
                    length * 1.045,
                    shell,
                    CHEST_WOOD_COLORS[index % len(CHEST_WOOD_COLORS)],
                    CHEST_WOOD_UV_RECT,
                    face_indices=CHEST_SHELL_FACES,
                )
            )

        # Two shallow segmented hoops follow the arch without penetrating the
        # wood shell; the previous coincident inner faces caused Z-fighting
        # when the open lid was viewed from inside.
        strap_depth = max(0.48, shell * 0.28)
        band_offset = (
            lid_width * 0.5
            - band_width * 0.5
            + surface_gap * 2.0
        )
        for width_sign in (-1.0, 1.0):
            for section in sections:
                edge0, edge1, tangent, normal, length, *_ = section
                parts.append(
                    _ChestPart(
                        (edge0 + edge1) * 0.5
                        + normal * (surface_gap + strap_depth * 0.5)
                        + self.width_axis * (width_sign * band_offset),
                        self.width_axis,
                        tangent,
                        normal,
                        band_width,
                        length * 0.99,
                        strap_depth,
                        CHEST_IRON_COLOR,
                    )
                )

        # Narrow stepped end panels close the sides under the curved shell.
        for width_sign in (-1.0, 1.0):
            outward = self.width_axis * width_sign
            for section in sections:
                _, _, _, _, _, d0, d1, h0, h1 = section
                fill_height = min(h0, h1) - frame * 0.30
                if fill_height <= 0.5:
                    continue
                parts.append(
                    _ChestPart(
                        hinge
                        + depth_axis * ((d0 + d1) * 0.5)
                        + height_axis * (fill_height * 0.5 + frame * 0.28)
                        + outward
                        * (lid_width * 0.5 - shell * 0.5 + surface_gap),
                        depth_axis,
                        height_axis,
                        outward,
                        abs(d1 - d0) * 0.965,
                        fill_height,
                        shell,
                        CHEST_WOOD_DARK,
                        CHEST_WOOD_UV_RECT,
                    )
                )

        rail_depth = frame * 0.82
        rail_lift = rail_depth * 0.16
        front_rail_width = max(
            frame,
            lid_width + rail_lift * 2.0 - rail_depth - surface_gap * 2.0,
        )
        side_rail_width = max(
            frame,
            lid_depth + rail_lift * 2.0 - rail_depth - surface_gap * 2.0,
        )
        for front_sign, d in ((-1.0, 0.0), (1.0, lid_depth)):
            outward = depth_axis * front_sign
            parts.append(
                _ChestPart(
                    hinge
                    + depth_axis * d
                    + height_axis * (frame * 0.5)
                    + outward * rail_lift,
                    self.width_axis,
                    height_axis,
                    outward,
                    front_rail_width,
                    frame,
                    rail_depth,
                    CHEST_IRON_COLOR,
                )
            )
        for width_sign in (-1.0, 1.0):
            outward = self.width_axis * width_sign
            parts.append(
                _ChestPart(
                    hinge
                    + depth_axis * (lid_depth * 0.5)
                    + height_axis * (frame * 0.5)
                    + outward * (lid_width * 0.5 + rail_lift),
                    depth_axis,
                    height_axis,
                    outward,
                    side_rail_width,
                    frame,
                    rail_depth,
                    CHEST_IRON_DARK,
                )
            )

        # The upper clasp is part of the lid and peels away naturally on open.
        latch_depth = max(1.0, frame * 0.58)
        parts.append(
            _ChestPart(
                hinge
                + depth_axis * (lid_depth + latch_depth * 0.62)
                + height_axis * (frame * 0.75),
                self.width_axis,
                height_axis,
                depth_axis,
                frame * 2.25,
                frame * 1.45,
                latch_depth,
                CHEST_GOLD_COLOR,
            )
        )
        self._lid_parts_cache_key = pose_key
        self._lid_parts_cache = tuple(parts)
        return self._lid_parts_cache

    def _visual_parts(self) -> tuple[_ChestPart, ...]:
        pose_key = self._geometry_pose_key()
        if (
            self._visual_parts_cache_key == pose_key
            and self._visual_parts_cache is not None
        ):
            return self._visual_parts_cache
        self._visual_parts_cache_key = pose_key
        self._visual_parts_cache = (*self._body_parts(), *self._lid_parts())
        return self._visual_parts_cache

    def _visual_box_specs(self):
        """Return geometry-only tuples retained for existing debug callers."""

        return tuple(part.box_spec() for part in self._visual_parts())

    def _visual_vertices(self) -> list[Vector3]:
        pose_key = self._geometry_pose_key()
        if (
            self._visual_vertices_cache_key == pose_key
            and self._visual_vertices_cache is not None
        ):
            return self._visual_vertices_cache

        verts: list[Vector3] = []
        for part in self._visual_parts():
            verts.extend(
                _oriented_box_vertices(
                    part.center,
                    part.width_axis,
                    part.height_axis,
                    part.depth_axis,
                    width=part.width,
                    height=part.height,
                    depth=part.depth,
                )
            )
        self._visual_vertices_cache_key = pose_key
        self._visual_vertices_cache = verts
        return self._visual_vertices_cache

    def _parts_vertex_data(
        self,
        parts: Sequence[_ChestPart],
        *,
        dynamic_lighting: bool,
    ) -> np.ndarray:
        rows: list[tuple[float, ...]] = []
        for part in parts:
            verts = _oriented_box_vertices(
                part.center,
                part.width_axis,
                part.height_axis,
                part.depth_axis,
                width=part.width,
                height=part.height,
                depth=part.depth,
            )
            self._append_box_vertex_data(
                rows,
                verts,
                part.width_axis,
                part.height_axis,
                part.depth_axis,
                color=part.color,
                uv_rect=part.uv_rect,
                face_indices=part.face_indices,
                dynamic_lighting=dynamic_lighting,
            )

        if not rows:
            return np.zeros(
                (0, 11 if dynamic_lighting else 8),
                dtype=np.float32,
            )
        return np.array(rows, dtype=np.float32)

    def _body_vertex_data(self, *, dynamic_lighting: bool) -> np.ndarray:
        cache_key = (
            bool(dynamic_lighting),
            None if dynamic_lighting else self._light_cache_key(),
        )
        cached = self._body_vertex_data_cache.get(cache_key)
        if cached is not None:
            return cached
        vertex_data = self._parts_vertex_data(
            self._body_parts(),
            dynamic_lighting=dynamic_lighting,
        )
        if len(self._body_vertex_data_cache) >= 2:
            self._body_vertex_data_cache.clear()
        self._body_vertex_data_cache[cache_key] = vertex_data
        return vertex_data

    def _vertex_data(self, *, dynamic_lighting: bool = False) -> np.ndarray:
        body_data = self._body_vertex_data(dynamic_lighting=dynamic_lighting)
        lid_data = self._parts_vertex_data(
            self._lid_parts(),
            dynamic_lighting=dynamic_lighting,
        )
        if body_data.size == 0:
            return lid_data
        if lid_data.size == 0:
            return body_data
        return np.ascontiguousarray(
            np.concatenate((body_data, lid_data), axis=0),
            dtype=np.float32,
        )

    def _mesh_cache_key(self, *, dynamic_lighting: bool = False):
        return (
            int(self.texture or 0),
            round(float(self.open_amount), 5),
            None if dynamic_lighting else self._light_cache_key(),
            bool(dynamic_lighting),
        )

    def get_render_bounding_sphere(self):
        pose_key = self._geometry_pose_key()
        if (
            self._render_sphere_cache_key == pose_key
            and self._render_sphere_cache is not None
        ):
            return self._render_sphere_cache
        self._render_sphere_cache_key = pose_key
        self._render_sphere_cache = sphere_for_vertices(self._visual_vertices())
        return self._render_sphere_cache

    def _collision_vertices(self) -> list[Vector3]:
        center = self.base_position + self.up_axis * (self.body_height * 0.5)
        return _oriented_box_vertices(
            center,
            self.width_axis,
            self.up_axis,
            self.front_axis,
            width=self.width,
            height=self.body_height,
            depth=self.depth,
        )

    def get_world_vertices(self):
        if not self.collision_enabled:
            return []
        return self._collision_vertices()

    def get_bounding_box(self):
        if not self.collision_enabled:
            return None
        if self._bounds_cache is None:
            self._bounds_cache = xz_bounds_for_vertices(self._collision_vertices())
        return self._bounds_cache

    def get_collision_meshes(self):
        return (self,)

    def _ensure_render_mesh(self, *, dynamic_lighting: bool) -> BatchedMesh | None:
        mesh_key = self._mesh_cache_key(dynamic_lighting=dynamic_lighting)
        if self._mesh is not None and mesh_key == self._mesh_key:
            return self._mesh
        self._dispose_mesh()
        vertex_data = self._vertex_data(dynamic_lighting=dynamic_lighting)
        if vertex_data.size == 0:
            return None
        self._mesh = BatchedMesh.from_vertex_data(
            vertex_data,
            texture=self.texture,
            alpha_test=False,
            exposure_baseline=1.0,
            environment_lighting=False,
            keep_vertex_data=False,
            lighting_receiver=(
                DYNAMIC_OBJECT_LIGHTING_RECEIVER
                if dynamic_lighting
                else CPU_BAKED_OBJECT_LIGHTING_RECEIVER
            ),
        )
        if self._mesh.bounds_center is not None:
            self._render_sphere_cache_key = self._geometry_pose_key()
            self._render_sphere_cache = (
                self._mesh.bounds_center,
                self._mesh.bounds_radius,
            )
            self._visual_vertices_cache_key = None
            self._visual_vertices_cache = None
        self._mesh_key = mesh_key
        return self._mesh

    def shadow_meshes(self, camera=None) -> tuple[BatchedMesh, ...]:
        """Expose current lid/body geometry before the color pass."""

        if not self.visible or not self.texture:
            return ()
        mesh = self._ensure_render_mesh(dynamic_lighting=True)
        return mesh.shadow_meshes() if mesh is not None else ()

    def draw(
        self,
        camera=None,
        *,
        lighting_packet=None,
        packet_shader=None,
    ) -> None:  # pragma: no cover - visual
        if not self.visible or not self.texture:
            return

        mesh = self._ensure_render_mesh(dynamic_lighting=packet_shader is not None)
        if mesh is None:
            return
        mesh.draw(
            camera=camera,
            lighting_packet=lighting_packet,
            packet_shader=packet_shader,
        )
