"""Batched pixel-art cloud rendering for the sky pass."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import random
import time

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_FLOAT,
    GL_MODULATE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_COORD_ARRAY,
    GL_TEXTURE_ENV,
    GL_TEXTURE_ENV_MODE,
    GL_TRIANGLES,
    GL_VERTEX_ARRAY,
    glBindBuffer,
    glBindTexture,
    glBlendFunc,
    glBufferData,
    glColor4f,
    glDisableClientState,
    glDrawArrays,
    glEnable,
    glEnableClientState,
    glGenBuffers,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTexCoordPointer,
    glTexEnvi,
    glVertexPointer,
)

from engine.textures.texture_utils import create_pixel_cloud_atlas


@dataclass(frozen=True)
class _Cloud:
    base_az_deg: float
    distance: float
    sky_y: float
    width: float
    height: float
    variant_index: int


class PixelCloudRenderer:
    """Draws distant pixel clouds as a mostly static sky-space VBO."""

    def __init__(
        self,
        *,
        max_clouds: int = 128,
        base_clouds: int = 64,
        seed: int = 7349,
    ) -> None:
        self.max_clouds = max(1, int(max_clouds))
        self.base_clouds = max(1, min(int(base_clouds), self.max_clouds))
        self._regions = create_pixel_cloud_atlas(variant_count=8)
        self._clouds = self._build_clouds(seed)
        self._vertex_count = len(self._clouds) * 6
        self._vbo = self._create_vbo()
        self._start_time_s = time.perf_counter()

    def _build_clouds(self, seed: int) -> list[_Cloud]:
        rng = random.Random(seed)
        clouds: list[_Cloud] = []
        az_step = 360.0 / self.max_clouds
        variant_count = max(1, len(self._regions))

        for index in range(self.max_clouds):
            base_az = index * az_step + rng.uniform(-az_step * 0.42, az_step * 0.42)
            layer = rng.random()
            distance = rng.uniform(36000.0, 57000.0)
            sky_y = rng.uniform(8500.0, 18000.0) + layer * rng.uniform(1200.0, 8200.0)
            width = rng.uniform(4400.0, 10400.0) * (0.86 + layer * 0.32)
            height = width * rng.uniform(0.27, 0.41)
            clouds.append(
                _Cloud(
                    base_az_deg=base_az % 360.0,
                    distance=distance,
                    sky_y=sky_y,
                    width=width,
                    height=height,
                    variant_index=rng.randrange(variant_count),
                )
            )

        rng.shuffle(clouds)
        return clouds

    def _create_vbo(self) -> int:
        vertex_data = np.ascontiguousarray(
            self._build_vertex_data(),
            dtype=np.float32,
        )
        vbo = glGenBuffers(1)
        vbo_id = int(np.asarray(vbo).reshape(-1)[0])
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        return vbo_id

    def _build_vertex_data(self) -> np.ndarray:
        vertex_data = np.empty((self._vertex_count, 5), dtype=np.float32)
        for cloud_index, cloud in enumerate(self._clouds):
            az = math.radians(cloud.base_az_deg)
            sin_az = math.sin(az)
            cos_az = math.cos(az)
            cx = -sin_az * cloud.distance
            cz = -cos_az * cloud.distance
            cy = cloud.sky_y

            rx = cos_az
            rz = -sin_az
            hw = cloud.width * 0.5
            hh = cloud.height * 0.5

            left_x = cx - rx * hw
            left_z = cz - rz * hw
            right_x = cx + rx * hw
            right_z = cz + rz * hw
            top_y = cy + hh
            bottom_y = cy - hh

            region = self._regions[cloud.variant_index % len(self._regions)]
            u0, v0, u1, v1 = region.uv_rect
            start = cloud_index * 6
            vertex_data[start + 0] = (left_x, top_y, left_z, u0, v1)
            vertex_data[start + 1] = (right_x, top_y, right_z, u1, v1)
            vertex_data[start + 2] = (right_x, bottom_y, right_z, u1, v0)
            vertex_data[start + 3] = vertex_data[start + 0]
            vertex_data[start + 4] = vertex_data[start + 2]
            vertex_data[start + 5] = (left_x, bottom_y, left_z, u0, v0)
        return vertex_data

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _lighting_tint(
        lighting,
        sun_tint=None,
    ) -> tuple[float, float, float]:
        if sun_tint is not None:
            try:
                return (
                    float(sun_tint[0]),
                    float(sun_tint[1]),
                    float(sun_tint[2]),
                )
            except Exception:
                pass
        if lighting is None:
            return (1.0, 1.0, 1.0)
        try:
            tint = getattr(lighting, "sun_tint")
            return (float(tint[0]), float(tint[1]), float(tint[2]))
        except Exception:
            return (1.0, 1.0, 1.0)

    def draw(
        self,
        *,
        brightness: float,
        density: float = 1.0,
        speed: float = 1.0,
        opacity: float = 1.0,
        lighting=None,
        sun_tint=None,
    ) -> None:  # pragma: no cover - visual
        if not self._regions or not self._clouds:
            return

        density = max(0.0, float(density))
        opacity = self._clamp01(opacity)
        if density <= 0.0 or opacity <= 0.0:
            return

        cloud_count = min(
            len(self._clouds),
            max(0, int(round(self.base_clouds * density))),
        )
        if cloud_count <= 0:
            return

        texture = int(self._regions[0].texture or 0)
        if texture == 0:
            return

        speed_scale = max(0.0, float(speed))
        brightness = self._clamp01(brightness)
        sr, sg, sb = self._lighting_tint(lighting, sun_tint)
        tint_r = brightness * (0.78 + 0.22 * sr)
        tint_g = brightness * (0.80 + 0.20 * sg)
        tint_b = brightness * (0.83 + 0.17 * sb)
        drift_deg = (time.perf_counter() - self._start_time_s) * 0.16 * speed_scale
        if speed_scale > 0.0:
            drift_deg = round(drift_deg * 24.0) / 24.0
        vertex_count = cloud_count * 6

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glBindTexture(GL_TEXTURE_2D, texture)
        glColor4f(
            self._clamp01(tint_r),
            self._clamp01(tint_g),
            self._clamp01(tint_b),
            opacity,
        )
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glPushMatrix()
        try:
            if drift_deg != 0.0:
                glRotatef(drift_deg, 0.0, 1.0, 0.0)
            stride = 5 * 4
            glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
            glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(3 * 4))
            glDrawArrays(GL_TRIANGLES, 0, vertex_count)
        finally:
            glPopMatrix()
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glColor4f(1.0, 1.0, 1.0, 1.0)
