"""World-space minimap billboard for world-space features."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pygame
from pygame.math import Vector3
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_DEPTH_TEST,
    GL_FLOAT,
    GL_LINEAR,
    GL_QUADS,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_ARRAY,
    glBegin,
    glBindBuffer,
    glBindTexture,
    glBlendFunc,
    glColor4f,
    glDeleteTextures,
    glDepthMask,
    glDisable,
    glDisableClientState,
    glDrawArrays,
    glEnable,
    glEnableClientState,
    glEnd,
    glGenTextures,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex3f,
    glVertexPointer,
)

from engine.core.consts import FORWARD, RIGHT, WORLD_UP


_EPS2 = 1e-12


@dataclass(frozen=True)
class MiniMapContext:
    """Resolved local map geometry passed to minimap layer callbacks."""

    scene: object
    panel_x: float
    panel_y: float
    panel_size: int
    map_x: float
    map_y: float
    map_width: float
    map_height: float
    scale: float
    bounds: tuple[float, float, float, float]

    def world_to_map(self, x: float, z: float) -> tuple[float, float]:
        min_x, _max_x, min_z, _max_z = self.bounds
        return (
            self.map_x + (float(x) - min_x) * self.scale,
            self.map_y + (float(z) - min_z) * self.scale,
        )

    def world_length_to_map(self, length: float) -> float:
        return float(length) * self.scale

    def contains_world_point(self, x: float, z: float) -> bool:
        min_x, max_x, min_z, max_z = self.bounds
        return min_x <= float(x) <= max_x and min_z <= float(z) <= max_z


@dataclass
class MiniMapLayer:
    key: str
    draw: Callable[[MiniMapContext], None] | None = None
    visible: bool = True


class MiniMapOverlay:
    """Draw a compact top-down world map as a camera-facing billboard.

    The built-in roads/buildings are cached into one small texture. Live
    markers are drawn on the same billboard plane so the minimap behaves like
    the compass/sword HUD sprites instead of a hard-pinned screen overlay.
    """

    _DEFAULT_LAYER_KEYS = frozenset(("roads", "buildings", "goblins", "player"))

    def __init__(
        self,
        scene,
        *,
        size: float = 230.0,
        padding: float = 12.0,
        world_size: tuple[float, float] = (1.75, 1.75),
    ) -> None:
        self.scene = scene
        self.size = float(size)
        self.padding = float(padding)
        self.world_size = (float(world_size[0]), float(world_size[1]))
        self.camera = getattr(scene, "camera", None)
        self.position = Vector3(0.0, 0.0, 0.0)
        self._layers: dict[str, MiniMapLayer] = {
            "roads": MiniMapLayer("roads"),
            "buildings": MiniMapLayer("buildings"),
            "goblins": MiniMapLayer("goblins"),
            "player": MiniMapLayer("player"),
        }
        self._texture_id = 0
        self._static_surface: pygame.Surface | None = None
        self._static_cache_key = None
        self._goblin_marker_vertices = np.empty((0, 3), dtype=np.float32)

    def update(self, dt: float) -> None:
        return None

    def add_layer(
        self,
        key: str,
        draw: Callable[[MiniMapContext], None],
        *,
        visible: bool = True,
    ) -> MiniMapLayer:
        layer = MiniMapLayer(str(key), draw, bool(visible))
        self._layers[layer.key] = layer
        return layer

    def remove_layer(self, key: str) -> None:
        key = str(key)
        self._layers.pop(key, None)
        if key in {"roads", "buildings"}:
            self.invalidate_static_cache()

    def set_layer_visible(self, key: str, visible: bool) -> None:
        key = str(key)
        layer = self._layers.get(key)
        if layer is None:
            return
        was_visible = layer.visible
        layer.visible = bool(visible)
        if was_visible == layer.visible:
            return
        if key in {"roads", "buildings"}:
            self.invalidate_static_cache()

    def invalidate_static_cache(self) -> None:
        self._static_surface = None
        self._static_cache_key = None

    def dispose(self) -> None:
        if self._texture_id:
            try:
                glDeleteTextures([self._texture_id])
            except Exception:
                pass
            self._texture_id = 0
        self._static_surface = None
        self._static_cache_key = None

    @property
    def layers(self) -> tuple[MiniMapLayer, ...]:
        return tuple(self._layers.values())

    def draw(self, pitch_effect: bool = True) -> None:  # pragma: no cover - visual
        self._count("minimap.draw_calls")
        context = self._build_context()
        if context is None:
            self._count("minimap.skipped_context")
            return

        axes = self._billboard_axes(pitch_effect=pitch_effect)
        if axes is None:
            self._count("minimap.skipped_axes")
            return
        right, up = axes

        with self._profile("minimap.ensure_texture"):
            self._ensure_texture(context)

        self._begin_billboard()
        try:
            with self._profile("minimap.draw_texture"):
                self._draw_texture(context, right, up)
            glDisable(GL_TEXTURE_2D)
            if self._layer_visible("goblins"):
                with self._profile("minimap.draw_goblin_markers"):
                    self._draw_goblin_markers(context, right, up)
            if self._layer_visible("player"):
                with self._profile("minimap.draw_player_marker"):
                    self._draw_player_marker(context, right, up)
            with self._profile("minimap.custom_layers"):
                self._draw_custom_layers(context)
        finally:
            self._end_billboard()

    def _profile(self, name: str):
        profiler = getattr(self.scene, "profiler", None)
        if profiler is None or not getattr(profiler, "enabled", False):
            return nullcontext()
        return profiler.section(name)

    def _count(self, name: str, amount: float = 1.0) -> None:
        profiler = getattr(self.scene, "profiler", None)
        if profiler is not None and getattr(profiler, "enabled", False):
            profiler.count(name, amount)

    def _build_context(self) -> MiniMapContext | None:
        try:
            min_x, max_x, min_z, max_z = self.scene.ground_bounds
        except (TypeError, ValueError, AttributeError):
            return None

        world_w = max(1.0, float(max_x) - float(min_x))
        world_h = max(1.0, float(max_z) - float(min_z))
        panel_size = int(round(max(120.0, self.size)))
        map_size = max(1.0, panel_size - self.padding * 2)
        scale = min(map_size / world_w, map_size / world_h)
        map_w = world_w * scale
        map_h = world_h * scale
        panel_x = 0.0
        panel_y = 0.0
        map_x = panel_x + (panel_size - map_w) * 0.5
        map_y = panel_y + (panel_size - map_h) * 0.5

        return MiniMapContext(
            scene=self.scene,
            panel_x=panel_x,
            panel_y=panel_y,
            panel_size=panel_size,
            map_x=map_x,
            map_y=map_y,
            map_width=map_w,
            map_height=map_h,
            scale=scale,
            bounds=(float(min_x), float(max_x), float(min_z), float(max_z)),
        )

    def _billboard_axes(self, *, pitch_effect: bool) -> tuple[Vector3, Vector3] | None:
        camera = self.camera
        right = getattr(camera, "_right", RIGHT)
        forward = getattr(camera, "_forward", FORWARD)
        if pitch_effect:
            r = right.normalize() if right.length_squared() > _EPS2 else None
            f = forward.normalize() if forward.length_squared() > _EPS2 else None
            if not r or not f:
                return None
            up = r.cross(f)
            up = up.normalize() if up.length_squared() > _EPS2 else WORLD_UP
            return r, up

        r_flat = Vector3(right.x, 0.0, right.z)
        if r_flat.length_squared() <= _EPS2:
            f_flat = Vector3(forward.x, 0.0, forward.z)
            if f_flat.length_squared() <= _EPS2:
                return None
            r_flat = WORLD_UP.cross(f_flat)
        return r_flat.normalize(), WORLD_UP

    @staticmethod
    def _begin_billboard() -> None:
        glDisable(GL_DEPTH_TEST)
        glDepthMask(False)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    @staticmethod
    def _end_billboard() -> None:
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glDepthMask(True)
        glEnable(GL_DEPTH_TEST)

    def _layer_visible(self, key: str) -> bool:
        layer = self._layers.get(key)
        return bool(layer is not None and layer.visible)

    def _draw_custom_layers(self, context: MiniMapContext) -> None:
        for layer in self._layers.values():
            if layer.key in self._DEFAULT_LAYER_KEYS:
                continue
            if layer.visible and layer.draw is not None:
                layer.draw(context)

    def _ensure_texture(self, context: MiniMapContext) -> None:
        static_key = self._make_static_cache_key(context)
        if self._texture_id and static_key == self._static_cache_key:
            self._count("minimap.texture_cache_hits")
            return

        if self._static_surface is None or static_key != self._static_cache_key:
            with self._profile("minimap.build_static_surface"):
                self._static_surface = self._build_static_surface(context)
            self._static_cache_key = static_key
            self._count("minimap.static_rebuilds")

        texture_id = self._texture_id or int(glGenTextures(1))
        with self._profile("minimap.upload_texture"):
            data = pygame.image.tostring(self._static_surface, "RGBA", True)
            size = context.panel_size

            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                size,
                size,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                data,
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        self._texture_id = texture_id
        self._count("minimap.texture_uploads")

    def _make_static_cache_key(self, context: MiniMapContext):
        roads = getattr(self.scene, "roads", ()) or ()
        buildings = getattr(self.scene, "buildings", ()) or ()
        return (
            context.panel_size,
            self._rounded_tuple(context.bounds),
            bool(self._layer_visible("roads")),
            bool(self._layer_visible("buildings")),
            id(roads),
            len(roads),
            id(buildings),
            len(buildings),
        )

    @staticmethod
    def _rounded(value, *, digits: int = 3):
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _rounded_tuple(cls, values, *, digits: int = 3):
        try:
            return tuple(cls._rounded(value, digits=digits) for value in values)
        except TypeError:
            return ()

    def _build_static_surface(self, context: MiniMapContext) -> pygame.Surface:
        size = context.panel_size
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))

        pygame.draw.rect(surface, (6, 9, 10, 184), (0, 0, size, size))

        map_rect = self._local_map_rect(context)
        pygame.draw.rect(surface, (18, 31, 28, 158), map_rect)

        if self._layer_visible("roads"):
            self._draw_static_roads(surface, context)
        if self._layer_visible("buildings"):
            self._draw_static_buildings(surface, context)

        pygame.draw.rect(surface, (173, 199, 199, 148), map_rect, width=1)
        pygame.draw.rect(surface, (230, 242, 235, 92), (0, 0, size, size), width=1)
        return surface

    def _draw_texture(
        self,
        context: MiniMapContext,
        right: Vector3,
        up: Vector3,
    ) -> None:
        if not self._texture_id:
            return

        w, h = self.world_size
        hw = w * 0.5
        hh = h * 0.5
        center = self.position
        tl = center - right * hw + up * hh
        tr = center + right * hw + up * hh
        br = center + right * hw - up * hh
        bl = center - right * hw - up * hh

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        try:
            glTexCoord2f(0.0, 1.0)
            glVertex3f(tl.x, tl.y, tl.z)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(tr.x, tr.y, tr.z)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(br.x, br.y, br.z)
            glTexCoord2f(0.0, 0.0)
            glVertex3f(bl.x, bl.y, bl.z)
        finally:
            glEnd()

    def _draw_static_roads(
        self,
        surface: pygame.Surface,
        context: MiniMapContext,
    ) -> None:
        for road in getattr(context.scene, "roads", ()) or ():
            points = getattr(road, "points", None) or ()
            if len(points) < 2:
                continue

            line_width = max(
                2,
                min(
                    8,
                    int(round(context.world_length_to_map(getattr(road, "width", 1.0)))),
                ),
            )
            local_points = [
                self._world_to_local(context, point.x, point.z)
                for point in points
            ]
            pygame.draw.lines(
                surface,
                (148, 140, 125, 230),
                False,
                local_points,
                width=line_width,
            )

    def _draw_static_buildings(
        self,
        surface: pygame.Surface,
        context: MiniMapContext,
    ) -> None:
        for building in getattr(context.scene, "buildings", ()) or ():
            try:
                min_x, max_x, min_z, max_z = building.bounds
            except (TypeError, ValueError, AttributeError):
                continue

            x0, y0 = self._world_to_local(context, min_x, min_z)
            x1, y1 = self._world_to_local(context, max_x, max_z)
            x = int(round(min(x0, x1)))
            y = int(round(min(y0, y1)))
            w = max(3, int(round(abs(x1 - x0))))
            h = max(3, int(round(abs(y1 - y0))))
            rect = (x, y, w, h)
            pygame.draw.rect(surface, (224, 184, 107, 188), rect)
            pygame.draw.rect(surface, (255, 230, 158, 235), rect, width=1)

    def _draw_goblin_markers(
        self,
        context: MiniMapContext,
        right: Vector3,
        up: Vector3,
    ) -> None:
        marker_radius = 3.5
        drawn = 0
        goblins = getattr(context.scene, "goblins", ()) or ()
        vertices = self._goblin_vertices_for_capacity(len(goblins))
        radius_right = right * ((marker_radius / context.panel_size) * self.world_size[0])
        radius_up = up * ((marker_radius / context.panel_size) * self.world_size[1])

        for goblin in goblins:
            if not getattr(goblin, "enabled", True):
                continue
            position = getattr(goblin, "position", None)
            if position is None or not context.contains_world_point(
                position.x,
                position.z,
            ):
                continue

            x, y = context.world_to_map(position.x, position.z)
            center = self._panel_to_world(context, x, y, right, up)
            start = drawn * 4
            vertices[start + 0] = self._vector_to_tuple(center + radius_up)
            vertices[start + 1] = self._vector_to_tuple(center + radius_right)
            vertices[start + 2] = self._vector_to_tuple(center - radius_up)
            vertices[start + 3] = self._vector_to_tuple(center - radius_right)
            drawn += 1

        if drawn <= 0:
            self._count("minimap.goblins_drawn", 0)
            return

        vertex_count = drawn * 4
        glColor4f(0.34, 0.95, 0.43, 0.95)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glEnableClientState(GL_VERTEX_ARRAY)
        try:
            glVertexPointer(3, GL_FLOAT, 0, vertices[:vertex_count])
            glDrawArrays(GL_QUADS, 0, vertex_count)
        finally:
            glDisableClientState(GL_VERTEX_ARRAY)
            self._count("minimap.goblins_drawn", drawn)

    def _goblin_vertices_for_capacity(self, goblin_count: int) -> np.ndarray:
        vertex_count = max(0, int(goblin_count)) * 4
        if self._goblin_marker_vertices.shape[0] >= vertex_count:
            return self._goblin_marker_vertices

        capacity = max(64, vertex_count, int(self._goblin_marker_vertices.shape[0] * 1.5))
        self._goblin_marker_vertices = np.empty((capacity, 3), dtype=np.float32)
        return self._goblin_marker_vertices

    def _draw_player_marker(
        self,
        context: MiniMapContext,
        right_axis: Vector3,
        up_axis: Vector3,
    ) -> None:
        camera = getattr(context.scene, "camera", None)
        position = getattr(camera, "position", None)
        if position is None:
            return

        x, y = context.world_to_map(position.x, position.z)
        forward = getattr(camera, "_forward", None)
        fx = float(getattr(forward, "x", 0.0)) if forward is not None else 0.0
        fz = float(getattr(forward, "z", -1.0)) if forward is not None else -1.0
        length = max(0.001, (fx * fx + fz * fz) ** 0.5)
        fx /= length
        fz /= length

        radius = 8.0
        right_x = fz
        right_y = -fx
        tip_2d = (x + fx * radius, y + fz * radius)
        left_2d = (
            x - fx * radius * 0.65 - right_x * radius * 0.55,
            y - fz * radius * 0.65 - right_y * radius * 0.55,
        )
        right_2d = (
            x - fx * radius * 0.65 + right_x * radius * 0.55,
            y - fz * radius * 0.65 + right_y * radius * 0.55,
        )
        tip = self._panel_to_world(context, tip_2d[0], tip_2d[1], right_axis, up_axis)
        left = self._panel_to_world(context, left_2d[0], left_2d[1], right_axis, up_axis)
        right = self._panel_to_world(context, right_2d[0], right_2d[1], right_axis, up_axis)
        glColor4f(0.28, 0.76, 1.0, 0.98)
        glBegin(GL_TRIANGLES)
        try:
            glVertex3f(tip.x, tip.y, tip.z)
            glVertex3f(left.x, left.y, left.z)
            glVertex3f(right.x, right.y, right.z)
        finally:
            glEnd()

    def _panel_to_world(
        self,
        context: MiniMapContext,
        x: float,
        y: float,
        right: Vector3,
        up: Vector3,
    ) -> Vector3:
        px = ((float(x) / context.panel_size) - 0.5) * self.world_size[0]
        py = (0.5 - (float(y) / context.panel_size)) * self.world_size[1]
        return self.position + right * px + up * py

    @staticmethod
    def _vector_to_tuple(value: Vector3) -> tuple[float, float, float]:
        return (float(value.x), float(value.y), float(value.z))

    def _world_to_local(
        self,
        context: MiniMapContext,
        x: float,
        z: float,
    ) -> tuple[int, int]:
        screen_x, screen_y = context.world_to_map(x, z)
        return (
            int(round(screen_x - context.panel_x)),
            int(round(screen_y - context.panel_y)),
        )

    def _local_map_rect(self, context: MiniMapContext) -> pygame.Rect:
        return pygame.Rect(
            int(round(context.map_x - context.panel_x)),
            int(round(context.map_y - context.panel_y)),
            int(round(context.map_width)),
            int(round(context.map_height)),
        )
