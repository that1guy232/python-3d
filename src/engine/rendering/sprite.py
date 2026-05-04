"""World-space sprite (camera-facing billboard).

Draws a textured quad in 3D that always faces the camera. Uses the
fixed-function pipeline for simplicity to match the current codebase.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
from pygame.math import Vector3
from OpenGL.GL import (
    GL_ALPHA_TEST,
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_COLOR_ARRAY,
    GL_FLOAT,
    GL_MODULATE,
    GL_NORMAL_ARRAY,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_COORD_ARRAY,
    GL_TEXTURE_ENV,
    GL_TEXTURE_ENV_MODE,
    GL_TRIANGLES,
    GL_VERTEX_ARRAY,
    glBegin,
    glBindBuffer,
    glBindTexture,
    glBlendFunc,
    glColor3f,
    glColor4f,
    glColorPointer,
    glDepthMask,
    glDisable,
    glDisableClientState,
    glEnable,
    glEnableClientState,
    glEnd,
    glDrawArrays,
    glNormalPointer,
    glTexEnvi,
    glTexCoord2f,
    glNormal3f,
    glTexCoordPointer,
    glVertexPointer,
    glVertex3f,
)
from config import WIDTH, HEIGHT, VIEWDISTANCE
from core.consts import FORWARD, RIGHT, WORLD_UP
from core.compat_shader import get_texture_color_exposure_shader, use_fixed_pipeline
from engine.rendering.lighting import sprite_light_factor


# Internal helpers for billboard math
_EPS = 1e-6
_EPS2 = _EPS * _EPS


def _profile(profiler, name: str):
    if profiler is None or not getattr(profiler, "enabled", False):
        return nullcontext()
    return profiler.section(name)


def _count(profiler, name: str, amount: float = 1.0) -> None:
    if profiler is not None and getattr(profiler, "enabled", False):
        profiler.count(name, amount)


def _sprite_array_scratch(owner, sprite_count: int) -> dict:
    scratch = getattr(owner, "_sprite_array_scratch", None)
    current_capacity = int(scratch.get("sprite_capacity", 0)) if scratch else 0
    if scratch is not None and current_capacity >= sprite_count:
        return scratch

    sprite_capacity = max(64, int(sprite_count), int(current_capacity * 1.5))
    vertex_capacity = sprite_capacity * 6
    scratch = {
        "sprite_capacity": sprite_capacity,
        "vertex_capacity": vertex_capacity,
        "positions": np.empty((sprite_capacity, 3), dtype=np.float32),
        "sizes": np.empty((sprite_capacity, 2), dtype=np.float32),
        "uvs": np.empty((sprite_capacity, 4), dtype=np.float32),
        "rgb": np.empty((sprite_capacity, 3), dtype=np.float32),
        "vertices": np.empty((vertex_capacity, 3), dtype=np.float32),
        "colors": np.empty((vertex_capacity, 4), dtype=np.float32),
        "texcoords": np.empty((vertex_capacity, 2), dtype=np.float32),
        "normals": np.empty((vertex_capacity, 3), dtype=np.float32),
    }
    setattr(owner, "_sprite_array_scratch", scratch)
    return scratch


def _sprite_data_cache(owner, sprites: list["WorldSprite"], *, static_data: bool) -> dict:
    sprite_count = len(sprites)
    cache = getattr(owner, "_sprite_cull_cache", None)
    cache_valid = (
        cache is not None
        and cache.get("sprites") is sprites
        and cache.get("count") == sprite_count
        and cache.get("static_data") == bool(static_data)
        and (
            sprite_count == 0
            or (cache.get("first") is sprites[0] and cache.get("last") is sprites[-1])
        )
    )

    if cache_valid:
        if static_data:
            return cache
        _refresh_sprite_data_cache(cache, sprites)
        return cache

    positions = np.empty((sprite_count, 3), dtype=np.float32)
    sizes = np.empty((sprite_count, 2), dtype=np.float32)
    uvs = np.empty((sprite_count, 4), dtype=np.float32)
    rgb = np.empty((sprite_count, 3), dtype=np.float32)
    textures = np.empty(sprite_count, dtype=np.int64)

    cache = {
        "sprites": sprites,
        "count": sprite_count,
        "static_data": bool(static_data),
        "first": sprites[0] if sprite_count else None,
        "last": sprites[-1] if sprite_count else None,
        "positions": positions,
        "sizes": sizes,
        "uvs": uvs,
        "rgb": rgb,
        "textures": textures,
        "rel_x": np.empty(sprite_count, dtype=np.float32),
        "rel_y": np.empty(sprite_count, dtype=np.float32),
        "rel_z": np.empty(sprite_count, dtype=np.float32),
        "depths": np.empty(sprite_count, dtype=np.float32),
        "x_cam": np.empty(sprite_count, dtype=np.float32),
        "y_cam": np.empty(sprite_count, dtype=np.float32),
        "half_v": np.empty(sprite_count, dtype=np.float32),
        "half_h": np.empty(sprite_count, dtype=np.float32),
        "side_extra": np.empty(sprite_count, dtype=np.float32),
        "back_extra": np.empty(sprite_count, dtype=np.float32),
        "limit": np.empty(sprite_count, dtype=np.float32),
        "mask": np.empty(sprite_count, dtype=bool),
        "mask_tmp": np.empty(sprite_count, dtype=bool),
    }
    _refresh_sprite_data_cache(cache, sprites)
    setattr(owner, "_sprite_cull_cache", cache)
    return cache


def _refresh_sprite_data_cache(cache: dict, sprites: list["WorldSprite"]) -> None:
    positions = cache["positions"]
    sizes = cache["sizes"]
    uvs = cache["uvs"]
    rgb = cache["rgb"]
    textures = cache["textures"]

    for i, sprite in enumerate(sprites):
        texture = getattr(sprite, "texture", None)
        textures[i] = int(texture or 0)

        pos = sprite.position
        positions[i, 0] = pos.x
        positions[i, 1] = pos.y
        positions[i, 2] = pos.z

        size = sprite.size
        sizes[i, 0] = float(size[0])
        sizes[i, 1] = float(size[1])

        uv = sprite.uv_rect
        uvs[i, 0] = float(uv[0])
        uvs[i, 1] = float(uv[1])
        uvs[i, 2] = float(uv[2])
        uvs[i, 3] = float(uv[3])

        color = sprite.color
        rgb[i, 0] = float(color[0])
        rgb[i, 1] = float(color[1])
        rgb[i, 2] = float(color[2])


def _length_sq(v: Vector3) -> float:
    # pygame's Vector3 has length_squared; fall back if missing
    try:
        return v.length_squared()  # type: ignore[attr-defined]
    except Exception:
        return v.x * v.x + v.y * v.y + v.z * v.z


@dataclass
class WorldSprite:
    position: Vector3
    size: tuple[float, float]
    texture: int
    camera: Any  # expects Camera with _right and _forward vectors updated
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    uv_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        texture_id = getattr(self.texture, "texture", None)
        if texture_id is not None:
            self.uv_rect = getattr(self.texture, "uv_rect", self.uv_rect)
            self.texture = int(texture_id)

    @property
    def faces(self):
        return [[0, 1, 2, 3]]

    def _billboard_axes(
        self, camera, pitch_effect: bool
    ) -> tuple[Vector3, Vector3] | None:
        """Compute right/up axes for a billboard."""
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
        r = r_flat.normalize()
        return r, WORLD_UP

    def draw_untextured(self) -> None:  # keep interface parity
        self.draw()

    def draw(self, pitch_effect: bool = False) -> None:  # pragma: no cover - visual
        if not self.texture:
            return

        axes = self._billboard_axes(self.camera, pitch_effect)
        if not axes:
            return
        right, up = axes

        w, h = self.size
        hw, hh = w * 0.5, h * 0.5

        # Four corners in world space (counter-clockwise)
        center = self.position
        tl = center - right * hw + up * hh
        tr = center + right * hw + up * hh
        br = center + right * hw - up * hh
        bl = center - right * hw - up * hh

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glDisable(GL_ALPHA_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        r, g, b = self.color
        u0, v0, u1, v1 = self.uv_rect
        shader = get_texture_color_exposure_shader()

        if shader is not None:
            forward = getattr(self.camera, "_forward", FORWARD)
            normal = Vector3(-forward.x * 0.35, 1.0, -forward.z * 0.35)
            normal = normal.normalize() if _length_sq(normal) > _EPS2 else WORLD_UP
            shader.bind(
                scene_lighting_enabled=True,
                directional_enabled=True,
                environment_enabled=False,
            )
            glNormal3f(normal.x, normal.y, normal.z)
            glColor4f(r, g, b, 1.0)
        else:
            glColor3f(r, g, b)

        glBegin(GL_QUADS)
        try:
            # t,v ordering: (u, v) then vertex
            glTexCoord2f(u0, v1)
            glVertex3f(tl.x, tl.y, tl.z)
            glTexCoord2f(u1, v1)
            glVertex3f(tr.x, tr.y, tr.z)
            glTexCoord2f(u1, v0)
            glVertex3f(br.x, br.y, br.z)
            glTexCoord2f(u0, v0)
            glVertex3f(bl.x, bl.y, bl.z)
        finally:
            glEnd()
            if shader is not None:
                use_fixed_pipeline()
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)

    def get_world_vertices(self):
        axes = self._billboard_axes(self.camera, pitch_effect=False)
        if not axes:
            return None
        right, up = axes

        w, h = self.size
        hw, hh = w * 0.5, h * 0.5

        # Four corners in world space (counter-clockwise)
        center = self.position
        tl = center - right * hw + up * hh
        tr = center + right * hw + up * hh
        br = center + right * hw - up * hh
        bl = center - right * hw - up * hh

        return [tl, tr, br, bl]


def draw_sprites_batched(
    sprites: list["WorldSprite"],
    camera,
    ground_height_fn=None,
    *,
    lighting=None,
    sun_direction=None,
    profiler=None,
    static_data: bool = False,
) -> None:
    """Draw visible sprites in texture batches while keeping fixed-function GL."""
    if not sprites:
        return

    # camera validity
    if not hasattr(camera, "_forward") or not hasattr(camera, "_right"):
        return

    # hoist GL functions + constants locally (faster attribute lookup)
    glBindTexture_local = glBindTexture
    glColor4f_local = glColor4f
    glEnable_local = glEnable
    glDisable_local = glDisable
    glBlendFunc_local = glBlendFunc
    glDepthMask_local = glDepthMask
    glTexEnvi_local = glTexEnvi
    glBindBuffer_local = glBindBuffer
    glEnableClientState_local = glEnableClientState
    glDisableClientState_local = glDisableClientState
    glVertexPointer_local = glVertexPointer
    glColorPointer_local = glColorPointer
    glTexCoordPointer_local = glTexCoordPointer
    glNormalPointer_local = glNormalPointer
    glDrawArrays_local = glDrawArrays

    # hoist camera / config values
    forward = camera._forward
    cam_pos = camera.position
    fx, fy, fz = forward.x, forward.y, forward.z
    cx, cy, cz = cam_pos.x, cam_pos.y, cam_pos.z

    # small, allocation-free precomputed numpy-matrix fallback:
    use_numpy_transform = False
    R0x = R0y = R0z = R1x = R1y = R1z = R2x = R2y = R2z = None
    if hasattr(camera, "_R"):
        R = camera._R
        # try to extract rows as plain floats to avoid per-sprite numpy ops
        try:
            # works with numpy arrays or nested sequences
            R0x, R0y, R0z = float(R[0][0]), float(R[0][1]), float(R[0][2])
            R1x, R1y, R1z = float(R[1][0]), float(R[1][1]), float(R[1][2])
            R2x, R2y, R2z = float(R[2][0]), float(R[2][1]), float(R[2][2])
            use_numpy_transform = True
        except Exception:
            use_numpy_transform = False

    # frustum/culling params
    fov_scale = getattr(camera, "_fov_scale", (HEIGHT * 0.5) / 1.0)
    tan_half_fov = (HEIGHT * 0.5) / fov_scale
    aspect = WIDTH / HEIGHT

    brightness_default = getattr(camera, "brightness_default", 0.0)
    brightness_areas = getattr(camera, "brightness_areas", [])
    has_brightness_areas = bool(brightness_areas)
    get_brightness_at = getattr(camera, "get_brightness_at", None)
    sun_factor = (
        sprite_light_factor(lighting=lighting, sun_direction=sun_direction)
        if lighting is not None or sun_direction is not None
        else 1.0
    )
    shader = get_texture_color_exposure_shader()
    use_shader_lighting = shader is not None

    # cylindrical billboard axes (render)
    world_up = Vector3(0.0, 1.0, 0.0)
    right_render = Vector3(camera._right.x, 0.0, camera._right.z)
    if _length_sq(right_render) < _EPS:
        f_flat = Vector3(forward.x, 0.0, forward.z)
        if _length_sq(f_flat) < _EPS:
            return
        right_render = world_up.cross(f_flat)
    right_render = right_render.normalize()
    up_render = world_up

    rx_x, rx_y, rx_z = right_render.x, right_render.y, right_render.z
    ux_x, ux_y, ux_z = up_render.x, up_render.y, up_render.z
    fake_normal = Vector3(-forward.x * 0.35, 1.0, -forward.z * 0.35)
    if _length_sq(fake_normal) <= _EPS2:
        fake_normal = WORLD_UP
    else:
        fake_normal = fake_normal.normalize()
    fn_x, fn_y, fn_z = fake_normal.x, fake_normal.y, fake_normal.z

    sprite_side_cull_extra = getattr(camera, "sprite_side_cull_extra", 0.0)
    sprite_back_cull_extra = getattr(camera, "sprite_back_cull_extra", 0.0)

    view_distance = VIEWDISTANCE
    tan_h = tan_half_fov
    asp = aspect

    # Prepare cull fallback axes if numpy-transform not used
    if not use_numpy_transform:
        right_cull = camera._right.normalize()
        up_cull = right_cull.cross(forward).normalize()
        if _length_sq(right_cull) == 0 or _length_sq(up_cull) == 0:
            return
        rcx, rcy, rcz = right_cull.x, right_cull.y, right_cull.z
        ucx, ucy, ucz = up_cull.x, up_cull.y, up_cull.z

    _count(profiler, "sprites.total", len(sprites))

    data_cache = None
    visible_indices = None
    with _profile(profiler, "sprites.cull"):
        data_cache = _sprite_data_cache(
            camera,
            sprites,
            static_data=static_data,
        )
        positions = data_cache["positions"]
        sizes = data_cache["sizes"]
        textures = data_cache["textures"]
        widths = sizes[:, 0]
        heights = sizes[:, 1]

        rel_x = data_cache["rel_x"]
        rel_y = data_cache["rel_y"]
        rel_z = data_cache["rel_z"]
        depths = data_cache["depths"]
        x_cam = data_cache["x_cam"]
        y_cam = data_cache["y_cam"]
        half_v = data_cache["half_v"]
        half_h = data_cache["half_h"]
        side_extra = data_cache["side_extra"]
        back_extra = data_cache["back_extra"]
        limit = data_cache["limit"]
        visible_mask = data_cache["mask"]
        mask_tmp = data_cache["mask_tmp"]

        np.subtract(positions[:, 0], cx, out=rel_x)
        np.subtract(positions[:, 1], cy, out=rel_y)
        np.subtract(positions[:, 2], cz, out=rel_z)

        np.multiply(rel_x, fx, out=depths)
        np.multiply(rel_y, fy, out=x_cam)
        np.add(depths, x_cam, out=depths)
        np.multiply(rel_z, fz, out=x_cam)
        np.add(depths, x_cam, out=depths)

        np.multiply(widths, 0.75, out=back_extra)
        np.maximum(back_extra, sprite_back_cull_extra, out=back_extra)
        np.negative(back_extra, out=limit)
        np.not_equal(textures, 0, out=visible_mask)
        np.greater(depths, limit, out=mask_tmp)
        visible_mask &= mask_tmp
        np.less_equal(depths, view_distance, out=mask_tmp)
        visible_mask &= mask_tmp

        if use_numpy_transform:
            np.multiply(rel_x, R0x, out=x_cam)
            np.multiply(rel_y, R0y, out=limit)
            np.add(x_cam, limit, out=x_cam)
            np.multiply(rel_z, R0z, out=limit)
            np.add(x_cam, limit, out=x_cam)

            np.multiply(rel_x, R1x, out=y_cam)
            np.multiply(rel_y, R1y, out=limit)
            np.add(y_cam, limit, out=y_cam)
            np.multiply(rel_z, R1z, out=limit)
            np.add(y_cam, limit, out=y_cam)
        else:
            np.multiply(rel_x, rcx, out=x_cam)
            np.multiply(rel_y, rcy, out=limit)
            np.add(x_cam, limit, out=x_cam)
            np.multiply(rel_z, rcz, out=limit)
            np.add(x_cam, limit, out=x_cam)

            np.multiply(rel_x, ucx, out=y_cam)
            np.multiply(rel_y, ucy, out=limit)
            np.add(y_cam, limit, out=y_cam)
            np.multiply(rel_z, ucz, out=limit)
            np.add(y_cam, limit, out=y_cam)

        np.multiply(depths, tan_h, out=half_v)
        np.multiply(half_v, asp, out=half_h)

        np.multiply(widths, 0.9, out=side_extra)
        np.maximum(side_extra, sprite_side_cull_extra, out=side_extra)
        np.multiply(widths, 0.75, out=limit)
        np.add(limit, side_extra, out=limit)
        np.add(limit, half_h, out=limit)
        np.abs(x_cam, out=x_cam)
        np.less_equal(x_cam, limit, out=mask_tmp)
        visible_mask &= mask_tmp

        np.multiply(heights, 0.75, out=limit)
        np.add(limit, half_v, out=limit)
        np.abs(y_cam, out=y_cam)
        np.less_equal(y_cam, limit, out=mask_tmp)
        visible_mask &= mask_tmp

        visible_indices = np.flatnonzero(visible_mask)

    if visible_indices is None or len(visible_indices) == 0 or data_cache is None:
        return

    _count(profiler, "sprites.visible", len(visible_indices))

    # back-to-front sort
    with _profile(profiler, "sprites.sort"):
        depths = data_cache["depths"]
        visible_indices = visible_indices[np.argsort(depths[visible_indices])[::-1]]

    # Group adjacent depth-sorted sprites by texture. If callers provide atlas
    # regions, many logical sprites share the same underlying GL texture.
    batches = []
    with _profile(profiler, "sprites.batch_build"):
        textures = data_cache["textures"]
        visible_textures = textures[visible_indices]
        split_points = np.flatnonzero(visible_textures[1:] != visible_textures[:-1]) + 1
        start = 0
        for end in split_points:
            batches.append((int(visible_textures[start]), visible_indices[start:end]))
            start = int(end)
        batches.append((int(visible_textures[start]), visible_indices[start:]))
    _count(profiler, "sprites.batches", len(batches))

    with _profile(profiler, "sprites.draw_gl"):
        # Client arrays cut thousands of per-vertex PyOpenGL calls down to one
        # draw call per texture batch while keeping dynamic billboard vertices.
        glDepthMask_local(False)
        glEnable_local(GL_BLEND)
        glBlendFunc_local(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable_local(GL_TEXTURE_2D)
        glDisable_local(GL_ALPHA_TEST)
        glTexEnvi_local(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glBindBuffer_local(GL_ARRAY_BUFFER, 0)
        glEnableClientState_local(GL_VERTEX_ARRAY)
        glEnableClientState_local(GL_COLOR_ARRAY)
        glEnableClientState_local(GL_TEXTURE_COORD_ARRAY)
        if use_shader_lighting:
            glEnableClientState_local(GL_NORMAL_ARRAY)
            shader.bind(
                scene_lighting_enabled=True,
                directional_enabled=True,
                environment_enabled=False,
            )

        source_positions = data_cache["positions"]
        source_sizes = data_cache["sizes"]
        source_uvs = data_cache["uvs"]
        source_rgb = data_cache["rgb"]

        try:
            for tex, batch_indices in batches:
                sprite_count = len(batch_indices)
                vertex_count = sprite_count * 6
                _count(profiler, "sprites.vertices", vertex_count)

                with _profile(profiler, "sprites.build_arrays"):
                    scratch = _sprite_array_scratch(camera, sprite_count)
                    positions = scratch["positions"][:sprite_count]
                    sizes = scratch["sizes"][:sprite_count]
                    uvs = scratch["uvs"][:sprite_count]
                    rgb = scratch["rgb"][:sprite_count]
                    vertices = scratch["vertices"][:vertex_count]
                    colors = scratch["colors"][:vertex_count]
                    texcoords = scratch["texcoords"][:vertex_count]
                    normals = scratch["normals"][:vertex_count] if use_shader_lighting else None

                    np.take(source_positions, batch_indices, axis=0, out=positions)
                    np.take(source_sizes, batch_indices, axis=0, out=sizes)
                    np.take(source_uvs, batch_indices, axis=0, out=uvs)

                    if use_shader_lighting:
                        np.take(source_rgb, batch_indices, axis=0, out=rgb)
                    else:
                        np.take(source_rgb, batch_indices, axis=0, out=rgb)
                        if has_brightness_areas and get_brightness_at is not None:
                            for out_i, sprite_idx in enumerate(batch_indices):
                                brightness = get_brightness_at(
                                    sprites[int(sprite_idx)].position
                                )
                                if brightness < brightness_default:
                                    brightness = brightness_default
                                rgb[out_i, 0] *= brightness * sun_factor
                                rgb[out_i, 1] *= brightness * sun_factor
                                rgb[out_i, 2] *= brightness * sun_factor
                        else:
                            rgb *= brightness_default * sun_factor
                        np.clip(rgb, 0.0, 1.0, out=rgb)

                    hw = sizes[:, 0] * 0.5
                    hh = sizes[:, 1] * 0.5

                    vertex_view = vertices.reshape(sprite_count, 6, 3)
                    vertex_view[:, 0, :] = positions
                    vertex_view[:, 0, 0] += -rx_x * hw + ux_x * hh
                    vertex_view[:, 0, 1] += -rx_y * hw + ux_y * hh
                    vertex_view[:, 0, 2] += -rx_z * hw + ux_z * hh

                    vertex_view[:, 1, :] = positions
                    vertex_view[:, 1, 0] += rx_x * hw + ux_x * hh
                    vertex_view[:, 1, 1] += rx_y * hw + ux_y * hh
                    vertex_view[:, 1, 2] += rx_z * hw + ux_z * hh

                    vertex_view[:, 2, :] = positions
                    vertex_view[:, 2, 0] += rx_x * hw - ux_x * hh
                    vertex_view[:, 2, 1] += rx_y * hw - ux_y * hh
                    vertex_view[:, 2, 2] += rx_z * hw - ux_z * hh

                    vertex_view[:, 3, :] = vertex_view[:, 0, :]
                    vertex_view[:, 4, :] = vertex_view[:, 2, :]

                    vertex_view[:, 5, :] = positions
                    vertex_view[:, 5, 0] += -rx_x * hw - ux_x * hh
                    vertex_view[:, 5, 1] += -rx_y * hw - ux_y * hh
                    vertex_view[:, 5, 2] += -rx_z * hw - ux_z * hh

                    texcoord_view = texcoords.reshape(sprite_count, 6, 2)
                    texcoord_view[:, 0, 0] = uvs[:, 0]
                    texcoord_view[:, 0, 1] = uvs[:, 3]
                    texcoord_view[:, 1, 0] = uvs[:, 2]
                    texcoord_view[:, 1, 1] = uvs[:, 3]
                    texcoord_view[:, 2, 0] = uvs[:, 2]
                    texcoord_view[:, 2, 1] = uvs[:, 1]
                    texcoord_view[:, 3, :] = texcoord_view[:, 0, :]
                    texcoord_view[:, 4, :] = texcoord_view[:, 2, :]
                    texcoord_view[:, 5, 0] = uvs[:, 0]
                    texcoord_view[:, 5, 1] = uvs[:, 1]

                    color_view = colors.reshape(sprite_count, 6, 4)
                    color_view[:, :, 0:3] = rgb[:, np.newaxis, :]
                    color_view[:, :, 3] = 1.0

                    if normals is not None:
                        normal_view = normals.reshape(sprite_count, 6, 3)
                        normal_view[:, :, 0] = fn_x
                        normal_view[:, :, 1] = fn_y
                        normal_view[:, :, 2] = fn_z

                with _profile(profiler, "sprites.submit_arrays"):
                    glBindTexture_local(GL_TEXTURE_2D, tex)
                    glVertexPointer_local(3, GL_FLOAT, 0, vertices)
                    glColorPointer_local(4, GL_FLOAT, 0, colors)
                    glTexCoordPointer_local(2, GL_FLOAT, 0, texcoords)
                    if normals is not None:
                        glNormalPointer_local(GL_FLOAT, 0, normals)
                    glDrawArrays_local(GL_TRIANGLES, 0, vertex_count)
        finally:
            if shader is not None:
                use_fixed_pipeline()
            if use_shader_lighting:
                glDisableClientState_local(GL_NORMAL_ARRAY)
            glDisableClientState_local(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState_local(GL_COLOR_ARRAY)
            glDisableClientState_local(GL_VERTEX_ARRAY)
            glColor4f_local(1.0, 1.0, 1.0, 1.0)
            glDisable_local(GL_TEXTURE_2D)
            glDisable_local(GL_BLEND)
            glDepthMask_local(True)
