"""Simple text rendering for OpenGL with pygame fonts.

Provides a small API to draw 2D text in screen space on top of a 3D scene.
Uses a lightweight texture cache and supports dynamic labels (e.g., FPS).
"""

from __future__ import annotations

from collections.abc import Callable
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


def fit_scale(
    content_width: float,
    content_height: float,
    max_width: float | None = None,
    max_height: float | None = None,
) -> float:
    """Return a uniform, never-upscaling factor that fits both bounds."""

    width = max(0.0, float(content_width))
    height = max(0.0, float(content_height))
    scale = 1.0
    if max_width is not None and width > 0.0:
        scale = min(scale, max(0.0, float(max_width)) / width)
    if max_height is not None and height > 0.0:
        scale = min(scale, max(0.0, float(max_height)) / height)
    return max(0.0, min(1.0, scale))


def wrap_text(
    value: str,
    measure_width: Callable[[str], float],
    max_width: float,
) -> tuple[str, ...]:
    """Greedily wrap complete text, splitting overlong tokens when required."""

    text = str(value or "")
    limit = max(0.0, float(max_width))
    if limit <= 0.0:
        return tuple(text.split("\n"))

    def width(label: str) -> float:
        measured = measure_width(label)
        if isinstance(measured, (tuple, list)):
            measured = measured[0] if measured else 0.0
        return max(0.0, float(measured))

    def token_chunks(token: str) -> list[str]:
        if not token or width(token) <= limit:
            return [token]
        chunks: list[str] = []
        current = ""
        for character in token:
            candidate = current + character
            if current and width(candidate) > limit:
                chunks.append(current)
                current = character
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or [token]

    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue

        current = ""
        for word in words:
            chunks = token_chunks(word)
            for chunk_index, chunk in enumerate(chunks):
                separator = " " if current and chunk_index == 0 else ""
                candidate = f"{current}{separator}{chunk}"
                if not current or width(candidate) <= limit:
                    current = candidate
                    continue
                lines.append(current)
                current = chunk
                if chunk_index < len(chunks) - 1:
                    lines.append(current)
                    current = ""
        if current:
            lines.append(current)

    return tuple(lines)


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

    @staticmethod
    def _aligned_origin(
        x: float,
        y: float,
        width: float,
        height: float,
        align: str,
    ) -> tuple[float, float]:
        if align == "topright":
            return x - width, y
        if align == "bottomleft":
            return x, y - height
        if align == "bottomright":
            return x - width, y - height
        if align == "center":
            return x - width * 0.5, y - height * 0.5
        return x, y

    @staticmethod
    def _draw_slot_at(
        slot: _TexSlot,
        x: float,
        y: float,
        scale: float,
    ) -> tuple[int, int]:
        width = float(slot.size[0]) * scale
        height = float(slot.size[1]) * scale
        glBindTexture(GL_TEXTURE_2D, slot.id)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x, y)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x + width, y)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x + width, y + height)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x, y + height)
        glEnd()
        return int(round(width)), int(round(height))

    def draw_text(
        self,
        text: str,
        x: float,
        y: float,
        color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        *,
        key: Optional[str] = None,
        align: str = "topleft",
        max_width: float | None = None,
        max_height: float | None = None,
    ) -> Tuple[int, int]:  # returns (w, h)
        """Draw a single-line text at screen coords.

        key: supply for dynamic text; same key will reuse texture and only re-upload
             when the text changes.
        align: 'topleft' | 'topright' | 'bottomleft' | 'bottomright' | 'center'
        """
        # Ensure color is hashable for use in cache keys (callers may pass list)
        try:
            color = tuple(color)
        except Exception:
            # Fallback: coerce elements to int then tuple
            color = tuple(int(c) for c in color)

        if key is not None:
            slot = self._get_slot_for_key(key)
            if slot.last_text != text:
                surf = self.font.render(text, True, color)
                self._upload_surface(slot, surf)
                slot.last_text = text
        else:
            slot = self._get_slot_for_static(text, color)

        scale = fit_scale(slot.size[0], slot.size[1], max_width, max_height)
        draw_width = float(slot.size[0]) * scale
        draw_height = float(slot.size[1]) * scale
        draw_x, draw_y = self._aligned_origin(
            x,
            y,
            draw_width,
            draw_height,
            align,
        )
        return self._draw_slot_at(slot, draw_x, draw_y, scale)

    def draw_text_multiline(
        self,
        text: str,
        x: float,
        y: float,
        color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        *,
        align: str = "topleft",
        line_spacing: float = 1.2,
        max_width: float | None = None,
        max_height: float | None = None,
        wrap: bool = False,
    ) -> Tuple[int, int]:
        """Draw multi-line text; returns (total_w, total_h)."""
        if wrap and max_width is not None:
            lines = list(
                wrap_text(
                    text,
                    lambda line: self.font.size(line)[0],
                    max_width,
                )
            )
        else:
            lines = text.split("\n")
        if not lines:
            return 0, 0

        try:
            color = tuple(color)
        except Exception:
            color = tuple(int(component) for component in color)

        slots = [
            None if not line else self._get_slot_for_static(line, color)
            for line in lines
        ]
        line_widths = [0 if slot is None else slot.size[0] for slot in slots]
        max_width_native = max(line_widths, default=0)
        line_height = max(
            [self.font.get_height()]
            + [slot.size[1] for slot in slots if slot is not None]
        )
        line_step = float(line_height) * max(0.0, float(line_spacing))
        total_height_native = float(line_height)
        if len(lines) > 1:
            total_height_native += (len(lines) - 1) * line_step
        scale = fit_scale(
            max_width_native,
            total_height_native,
            max_width,
            max_height,
        )
        block_width = float(max_width_native) * scale
        block_height = total_height_native * scale

        if align == "center":
            start_y = y - block_height * 0.5
        elif align.startswith("bottom"):
            start_y = y - block_height
        else:
            start_y = y

        for index, slot in enumerate(slots):
            if slot is None:
                continue
            line_width = float(slot.size[0]) * scale
            if align == "center":
                line_x = x - line_width * 0.5
            elif align.endswith("right"):
                line_x = x - line_width
            else:
                line_x = x
            line_y = start_y + index * line_step * scale
            self._draw_slot_at(slot, line_x, line_y, scale)

        return int(round(block_width)), int(round(block_height))
