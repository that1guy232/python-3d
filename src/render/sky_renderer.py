"""Sky rendering utilities.

Contains the SkyRenderer that draws billboarded sky elements (star and moon).
Ensure an active OpenGL context exists before instantiating this class.
"""

from __future__ import annotations

from camera import Camera
from textures.texture_utils import load_texture
from textures.resoucepath import STAR_TEXTURE_PATH, MOON_TEXTURE_PATH
from typing import Optional
import time

class SkyRenderer:
    """Draws sky elements (star and moon) billboarded to the camera.

    Textures are created lazily at construction time. Ensure an active GL
    context exists before instantiating this class.
    """

    def __init__(self) -> None:
        # Load textures using the active GL context
        self._star_tex = load_texture(STAR_TEXTURE_PATH)
        self._moon_tex = load_texture(MOON_TEXTURE_PATH)
        # Moon placement offset (degrees) relative to star azimuth. Change
        # this if you want the moon to appear at a different azimuth than the
        # star/sun. Default 134 preserves legacy behavior.
        self.moon_offset_deg = 134.0

    def draw(self, camera: Camera, sun_direction: Optional[object] = None) -> None:  # pragma: no cover - visual
        """Draw both star and moon with minimal GL state churn.

        If `sun_direction` is provided (vector pointing FROM sun -> world)
        the sky quads will be rotated so their apparent azimuth matches
        the sun direction (world->sun azimuth = negate sun_direction).
        """
        from OpenGL.GL import (
            glPushMatrix,
            glPopMatrix,
            glBegin,
            glEnd,
            glVertex3f,
            glTexCoord2f,
            glBindTexture,
            glEnable,
            glDisable,
            glBlendFunc,
            glRotatef,
            glColor3f,
            GL_TEXTURE_2D,
            GL_QUADS,
            GL_BLEND,
            GL_SRC_ALPHA,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_DEPTH_TEST,
            GL_FOG,
        )
        import math as _math


        # Set render state ONCE for all sky objects
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_FOG)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        glPushMatrix()
        # Lock to camera orientation (shared for all sky objects)
        glRotatef(_math.degrees(-camera.rotation.x), 1, 0, 0)
        glRotatef(_math.degrees(-camera.rotation.y), 0, 1, 0)


        # Draw star and moon aligned with decal azimuth convention when possible.
        az = None
        if sun_direction is not None:
            try:
                sx = float(getattr(sun_direction, "x", sun_direction[0]))
                sz = float(getattr(sun_direction, "z", sun_direction[2]))
                # Match decal math: compute angle = degrees(atan2(proj_x, proj_z))
                # then add 270 deg to match art orientation.
                angle_deg = _math.degrees(_math.atan2(sx, sz))
                az = (angle_deg) % 360.0
            except Exception:
                az = None

        # Draw star
        glBindTexture(GL_TEXTURE_2D, self._star_tex)
        glPushMatrix()
        if az is not None:
            glRotatef(az, 0.0, 1.0, 0.0)
        self._draw_sky_quad(camera=camera,dist=50000.0, half=6000.0, y=20000.0)
        glPopMatrix()

        # Draw moon (apply optional offset so it doesn't occupy same azimuth)
        glBindTexture(GL_TEXTURE_2D, self._moon_tex)
        glPushMatrix()
        if az is not None:
            glRotatef((az + self.moon_offset_deg) % 360.0, 0.0, 1.0, 0.0)
        else:
            glRotatef(134.0, 0.0, 1.0, 0.0)
        self._draw_sky_quad(camera=camera, dist=80000.0, half=4000.0, y=20000.0)
        glPopMatrix()

        glPopMatrix()

        # Restore state ONCE after all objects
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_FOG)
        glEnable(GL_DEPTH_TEST)


    def _draw_sky_quad(self, *, dist: float, half: float, y: float, camera: Camera) -> None:
        from OpenGL.GL import (
            glBegin,
            glEnd,
            glVertex3f,
            glTexCoord2f,
            glColor3f,
            GL_QUADS,
        )

        brightness = camera.brightness_default
        glColor3f(brightness, brightness, brightness)

        glBegin(GL_QUADS)
        # Counter-clockwise winding, centered at (0, y, -dist)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-half, y - half, -dist)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(half, y - half, -dist)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(half, y + half, -dist)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-half, y + half, -dist)
        glEnd()
