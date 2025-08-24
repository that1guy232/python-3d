"""Simple text rendering for OpenGL with pygame fonts.

Provides a small API to draw 2D text in screen space on top of a 3D scene.
Uses a lightweight texture cache and supports dynamic labels (e.g., FPS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pygame
from OpenGL.GL import (
    glGenTextures,
    glBindTexture,
    glTexImage2D,
    glTexParameteri,
    glPushMatrix,
    glPopMatrix,
    glBegin,
    glEnd,
    glOrtho,
    glLoadIdentity,
    glTexCoord2f,
    glVertex2f,
    glColor4f,
    glBlendFunc,
    glEnable,
    glDisable,
    glMatrixMode,
    GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_LINEAR,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PROJECTION,
    GL_MODELVIEW,
    GL_QUADS,
    GL_DEPTH_TEST,
)


@dataclass
class _TexSlot:
    id: int
    size: Tuple[int, int]
    last_text: str | None = None


class TextRenderer:
    """2D text renderer for OpenGL using pygame.font.

    - Call begin() once before drawing multiple labels; call end() after.
    - draw_text() can take a `key` to reuse a texture slot for dynamic text (FPS).
    - Without a key, content is cached by (text, color) and reused.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        font: Optional[pygame.font.Font] = None,
        size: int = 24,
    ) -> None:
        self.width = screen_width
        self.height = screen_height
        self.font = font or pygame.font.Font(None, size)
        self._cache: Dict[Tuple[str, Tuple[int, int, int, int]], _TexSlot] = {}
        self._slots: Dict[str, _TexSlot] = {}
        self._in_overlay = False

    # --------------------------- overlay state ---------------------------
    def begin(self) -> None:  # pragma: no cover - visual
        if self._in_overlay:
            return
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        self._in_overlay = True

    def end(self) -> None:  # pragma: no cover - visual
        if not self._in_overlay:
            return
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        self._in_overlay = False

    # --------------------------- rendering ------------------------------
    def _upload_surface(self, slot: _TexSlot, surf: pygame.Surface) -> None:
        data = pygame.image.tostring(surf, "RGBA", True)
        w, h = surf.get_width(), surf.get_height()
        glBindTexture(GL_TEXTURE_2D, slot.id)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            w,
            h,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            data,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        slot.size = (w, h)

    def _get_slot_for_key(self, key: str) -> _TexSlot:
        slot = self._slots.get(key)
        if slot is None:
            tex_id = glGenTextures(1)
            slot = _TexSlot(id=tex_id, size=(0, 0), last_text=None)
            self._slots[key] = slot
        return slot

    def _get_slot_for_static(
        self, text: str, color: Tuple[int, int, int, int]
    ) -> _TexSlot:
        cache_key = (text, color)
        slot = self._cache.get(cache_key)
        if slot is None:
            tex_id = glGenTextures(1)
            slot = _TexSlot(id=tex_id, size=(0, 0), last_text=text)
            # pre-upload
            surf = self.font.render(text, True, color)
            self._upload_surface(slot, surf)
            self._cache[cache_key] = slot
        return slot

    def draw_text(
        self,
        text: str,
        x: float,
        y: float,
        color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        *,
        key: Optional[str] = None,
        align: str = "topleft",
    ) -> Tuple[int, int]:  # returns (w, h)
        """Draw a single-line text at screen coords.

        key: supply for dynamic text; same key will reuse texture and only re-upload
             when the text changes.
        align: 'topleft' | 'topright' | 'bottomleft' | 'bottomright' | 'center'
        """
        if key is not None:
            slot = self._get_slot_for_key(key)
            if slot.last_text != text:
                surf = self.font.render(text, True, color)
                self._upload_surface(slot, surf)
                slot.last_text = text
        else:
            slot = self._get_slot_for_static(text, color)

        w, h = slot.size
        # alignment
        if align == "topright":
            draw_x, draw_y = x - w, y
        elif align == "bottomleft":
            draw_x, draw_y = x, y - h
        elif align == "bottomright":
            draw_x, draw_y = x - w, y - h
        elif align == "center":
            draw_x, draw_y = x - w / 2, y - h / 2
        else:  # topleft
            draw_x, draw_y = x, y

        glBindTexture(GL_TEXTURE_2D, slot.id)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        # Note: pygame.image.tostring with True gives origin at top-left, so v coords flipped
        glTexCoord2f(0.0, 1.0)
        glVertex2f(draw_x, draw_y)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(draw_x + w, draw_y)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(draw_x + w, draw_y + h)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(draw_x, draw_y + h)
        glEnd()
        return w, h

    def draw_text_multiline(
        self,
        text: str,
        x: float,
        y: float,
        color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        *,
        align: str = "topleft",
        line_spacing: float = 1.2,
    ) -> Tuple[int, int]:
        """Draw multi-line text; returns (total_w, total_h)."""
        lines = text.splitlines() if "\n" in text else [text]
        if not lines:
            return 0, 0

        # Measure line height (approximate using metrics if available)
        metrics = self.font.metrics("Ag")
        if metrics and metrics[0]:
            heights = [m[3] - m[2] if m else self.font.get_height() for m in metrics]
            line_h = max(heights) or self.font.get_height()
        else:
            line_h = self.font.get_height()

        # Measure widths without uploading textures
        line_widths = [self.font.size(line)[0] for line in lines]
        max_w = max(line_widths) if line_widths else 0

        # Compute total block height for n lines with spacing
        n = len(lines)
        total_h = int(line_h if n == 1 else line_h + (n - 1) * line_h * line_spacing)

        # Treat 'align' as block alignment; compute top-left of the block
        if align == "center":
            start_x = x - max_w / 2
            start_y = y - total_h / 2
        else:
            # Horizontal
            if align.endswith("right"):
                start_x = x - max_w
            else:  # left or unspecified
                start_x = x
            # Vertical
            if align.startswith("bottom"):
                start_y = y - total_h
            else:  # top or unspecified
                start_y = y

        # Draw each line top-to-bottom using topleft alignment for lines
        for i, line in enumerate(lines):
            line_y = start_y + int(i * line_h * line_spacing)
            self.draw_text(line, start_x, line_y, color, key=None, align="topleft")

        return int(max_w), int(total_h)
