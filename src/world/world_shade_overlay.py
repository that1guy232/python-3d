"""2D full-screen shading overlay drawn after the 3D world and before HUD.

Extracted from engine.py to keep responsibilities separated.
"""

from __future__ import annotations

from config import WIDTH, HEIGHT


class WorldShadeOverlay:
    """Simple full-screen shade to darken/tint the 3D world.

    Drawn in screen space with alpha blending after the world and before HUD.
    """

    def __init__(
        self, opacity: float = 0.3, color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.opacity = max(0.0, min(1.0, opacity))
        self.color = color  # RGB in 0..1

    def draw(self):  # pragma: no cover - visual
        # Local imports to avoid polluting module scope
        from OpenGL.GL import (
            glPushMatrix,
            glPopMatrix,
            glBegin,
            glEnd,
            glOrtho,
            glLoadIdentity,
            glMatrixMode,
            glDisable,
            glEnable,
            glBlendFunc,
            glColor4f,
            glVertex2f,
            glIsEnabled,
            GL_PROJECTION,
            GL_MODELVIEW,
            GL_BLEND,
            GL_SRC_ALPHA,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_DEPTH_TEST,
            GL_QUADS,
            GL_TEXTURE_2D,
            GL_FOG,
        )

        # Setup 2D orthographic projection
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Render state for translucent overlay
        was_fog = glIsEnabled(GL_FOG)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        r, g, b = self.color
        glColor4f(r, g, b, self.opacity)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(WIDTH, 0)
        glVertex2f(WIDTH, HEIGHT)
        glVertex2f(0, HEIGHT)
        glEnd()

        # Restore state
        glDisable(GL_BLEND)
        if was_fog:
            glEnable(GL_FOG)
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
