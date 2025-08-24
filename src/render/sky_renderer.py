"""Sky rendering utilities.

Contains the SkyRenderer that draws billboarded sky elements (star and moon).
Ensure an active OpenGL context exists before instantiating this class.
"""

from __future__ import annotations

from camera import Camera
from textures.texture_utils import load_texture
from textures.resoucepath import STAR_TEXTURE_PATH, MOON_TEXTURE_PATH


class SkyRenderer:
    """Draws sky elements (star and moon) billboarded to the camera.

    Textures are created lazily at construction time. Ensure an active GL
    context exists before instantiating this class.
    """

    def __init__(self) -> None:
        # Load textures using the active GL context
        self._star_tex = load_texture(STAR_TEXTURE_PATH)
        self._moon_tex = load_texture(MOON_TEXTURE_PATH)

    def draw(self, camera: Camera) -> None:  # pragma: no cover - visual
        """Draw both star and moon with minimal GL state churn."""
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

        # Draw star
        glBindTexture(GL_TEXTURE_2D, self._star_tex)
        self._draw_sky_quad(dist=50000.0, half=6000.0, y=20000.0)

        # Draw moon (with rotation offset)
        glBindTexture(GL_TEXTURE_2D, self._moon_tex)
        glPushMatrix()
        glRotatef(134.0, 0.0, 1.0, 0.0)  # Corrected Y-axis rotation
        self._draw_sky_quad(dist=80000.0, half=4000.0, y=20000.0)
        glPopMatrix()

        glPopMatrix()

        # Restore state ONCE after all objects
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_FOG)
        glEnable(GL_DEPTH_TEST)

    def _draw_sky_quad(self, *, dist: float, half: float, y: float) -> None:
        from OpenGL.GL import (
            glBegin,
            glEnd,
            glVertex3f,
            glTexCoord2f,
            glColor3f,
            GL_QUADS,
        )

        # White to use texture as-is
        glColor3f(1.0, 1.0, 1.0)
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
