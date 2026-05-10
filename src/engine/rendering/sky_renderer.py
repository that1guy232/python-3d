"""Sky rendering utilities.

Contains the SkyRenderer that draws billboarded sky elements. Ensure an active
OpenGL context exists before instantiating this class.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional
import math

from OpenGL.GL import (
    glBegin,
    glBindTexture,
    glBlendFunc,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTexCoord2f,
    glVertex3f,
    GL_BLEND,
    GL_DEPTH_TEST,
    GL_FOG,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
)

from engine.camera import Camera
from engine.textures.texture_utils import load_texture
from engine.rendering.cloud_renderer import PixelCloudRenderer
from engine.rendering.lighting import SceneLighting, sky_sun_y


class SkyRenderer:
    """Draws sky elements billboarded to the camera."""

    def __init__(
        self,
        *,
        sun_texture_path: str | None = None,
        moon_texture_path: str | None = None,
    ) -> None:
        self._sun_tex = load_texture(sun_texture_path) if sun_texture_path else 0
        self._moon_tex = load_texture(moon_texture_path) if moon_texture_path else 0
        self._clouds = PixelCloudRenderer()
        self.moon_offset_deg = 134.0
        self.sun_half_size = 6000.0
        self.moon_half_size = 2200.0

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _sun_azimuth_deg(sun_direction: Optional[object]) -> float | None:
        if sun_direction is None:
            return None

        try:
            sx = float(
                sun_direction.x if hasattr(sun_direction, "x") else sun_direction[0]
            )
            sz = float(
                sun_direction.z if hasattr(sun_direction, "z") else sun_direction[2]
            )
        except Exception:
            return None

        if abs(sx) < 1e-9 and abs(sz) < 1e-9:
            return None
        return math.degrees(math.atan2(sx, sz)) % 360.0

    def draw(
        self,
        camera: Camera,
        sun_direction: Optional[object] = None,
        *,
        lighting: SceneLighting | None = None,
        fog_enabled: bool = True,
        clouds_enabled: bool = True,
        cloud_density: float = 1.0,
        cloud_speed: float = 1.0,
        cloud_opacity: float = 1.0,
        profiler=None,
    ) -> None:  # pragma: no cover - visual
        """Draw the sky with minimal GL state churn."""

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_FOG)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        glPushMatrix()
        glRotatef(math.degrees(-camera.rotation.x), 1, 0, 0)
        glRotatef(math.degrees(-camera.rotation.y), 0, 1, 0)

        brightness = self._clamp01(getattr(camera, "brightness_default", 1.0))
        active_sun_direction = (
            lighting.sun_direction if lighting is not None else sun_direction
        )
        az = self._sun_azimuth_deg(active_sun_direction)
        sun_y = sky_sun_y(
            sun_direction=active_sun_direction,
            lighting=lighting,
            xz_distance=50000.0,
            fallback_y=20000.0,
        )
        if lighting is not None:
            sr, sg, sb = lighting.sun_tint
        else:
            sr, sg, sb = (1.0, 1.0, 1.0)

        glBindTexture(GL_TEXTURE_2D, self._sun_tex)
        glPushMatrix()
        if az is not None:
            glRotatef(az, 0.0, 1.0, 0.0)
        self._draw_sky_quad(
            dist=50000.0,
            half=self.sun_half_size,
            y=sun_y,
            color=(brightness * sr, brightness * sg, brightness * sb, 1.0),
        )
        glPopMatrix()

        if clouds_enabled:
            cloud_context = (
                profiler.section("render.clouds")
                if profiler is not None and getattr(profiler, "enabled", False)
                else nullcontext()
            )
            with cloud_context:
                self._clouds.draw(
                    brightness=brightness,
                    density=cloud_density,
                    speed=cloud_speed,
                    opacity=cloud_opacity,
                    lighting=lighting,
                )

        # glBindTexture(GL_TEXTURE_2D, self._moon_tex)
        # glPushMatrix()
        # moon_az = self.moon_offset_deg if az is None else (az + self.moon_offset_deg)
        # glRotatef(moon_az % 360.0, 0.0, 1.0, 0.0)
        # self._draw_sky_quad(
        #     dist=48000.0,
        #     half=self.moon_half_size,
        #     y=13500.0,
        #     color=(
        #         min(1.0, brightness * 1.08),
        #         min(1.0, brightness * 1.10),
        #         min(1.0, brightness * 1.15),
        #         0.92,
        #     ),
        # )
        # glPopMatrix()

        glPopMatrix()

        glColor4f(1.0, 1.0, 1.0, 1.0)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        if fog_enabled:
            glEnable(GL_FOG)
        else:
            glDisable(GL_FOG)
        glEnable(GL_DEPTH_TEST)

    def _draw_sky_quad(
        self,
        *,
        dist: float,
        half: float,
        y: float,
        color: tuple[float, float, float, float],
    ) -> None:
        glColor4f(*color)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-half, y - half, -dist)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(half, y - half, -dist)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(half, y + half, -dist)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-half, y + half, -dist)
        glEnd()
