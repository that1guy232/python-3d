"""Screen-space battle cards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from OpenGL.GL import (
    glBegin,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glVertex2f,
    GL_QUADS,
    GL_TEXTURE_2D,
)

CardHandler = Callable[[object], None]
CardPredicate = Callable[[object], bool]


@dataclass(frozen=True)
class CardStyle:
    shadow: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.30)
    border: tuple[float, float, float, float] = (0.055, 0.045, 0.038, 0.98)
    border_hover: tuple[float, float, float, float] = (0.72, 0.44, 0.18, 1.0)
    face: tuple[float, float, float, float] = (0.16, 0.12, 0.09, 0.96)
    face_hover: tuple[float, float, float, float] = (0.22, 0.16, 0.10, 0.98)
    panel: tuple[float, float, float, float] = (0.055, 0.052, 0.055, 0.92)
    accent: tuple[float, float, float, float] = (0.58, 0.16, 0.10, 0.96)
    accent_hover: tuple[float, float, float, float] = (0.78, 0.23, 0.12, 1.0)
    disabled_tint: tuple[float, float, float, float] = (0.07, 0.07, 0.075, 0.66)


class Card:
    """A draggable UI-layer card drawn as simple quads."""

    def __init__(
        self,
        action: str,
        title: str,
        detail: str,
        on_play: CardHandler,
        *,
        footer: str = "",
        can_play: CardPredicate | None = None,
        width: float = 112.0,
        height: float = 154.0,
        style: CardStyle | None = None,
    ) -> None:
        self.action = str(action)
        self.title = str(title)
        self.detail = str(detail)
        self.footer = str(footer)
        self.on_play = on_play
        self.can_play = can_play
        self.style = style or CardStyle()

        self.home_rect = (0.0, 0.0, float(width), float(height))
        self.rect = self.home_rect
        self.hovered = False
        self.dragging = False
        self._drag_offset = (float(width) * 0.5, float(height) * 0.5)

    @staticmethod
    def _draw_rect(
        x: float,
        y: float,
        w: float,
        h: float,
        color: tuple[float, float, float, float],
    ) -> None:
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    @staticmethod
    def _contains(rect: tuple[float, float, float, float], pos) -> bool:
        x, y, w, h = rect
        px, py = pos
        return x <= px <= x + w and y <= py <= y + h

    @staticmethod
    def _center_in_rect(
        rect: tuple[float, float, float, float],
        target: tuple[float, float, float, float],
    ) -> bool:
        x, y, w, h = rect
        tx, ty, tw, th = target
        cx = x + w * 0.5
        cy = y + h * 0.5
        return tx <= cx <= tx + tw and ty <= cy <= ty + th

    @staticmethod
    def _coerce_rect(rect) -> tuple[float, float, float, float]:
        x, y, w, h = rect
        return float(x), float(y), float(w), float(h)

    def enabled_for(self, scene) -> bool:
        if self.can_play is None:
            return True
        try:
            return bool(self.can_play(scene))
        except Exception:
            return False

    def set_home_center(
        self,
        x: float,
        y: float,
        *,
        size: tuple[float, float] | None = None,
    ) -> None:
        if size is None:
            _old_x, _old_y, w, h = self.home_rect
        else:
            w, h = float(size[0]), float(size[1])
        self.home_rect = (float(x) - w * 0.5, float(y) - h * 0.5, w, h)
        if not self.dragging:
            self.rect = self.home_rect

    def reset_to_home(self) -> None:
        self.rect = self.home_rect
        self.dragging = False
        self._drag_offset = (self.home_rect[2] * 0.5, self.home_rect[3] * 0.5)

    def update_hover(self, pos, scene=None) -> bool:
        self.hovered = (
            (scene is None or self.enabled_for(scene))
            and not self.dragging
            and self._contains(self.rect, pos)
        )
        return self.hovered

    def handle_mouse_down(self, pos, scene) -> bool:
        if not self.enabled_for(scene) or not self._contains(self.rect, pos):
            self.update_hover(pos, scene)
            return False

        x, y, _w, _h = self.rect
        self.dragging = True
        self.hovered = False
        self._drag_offset = (float(pos[0]) - x, float(pos[1]) - y)
        return True

    def handle_mouse_motion(self, pos, scene) -> bool:
        if self.dragging:
            _x, _y, w, h = self.rect
            ox, oy = self._drag_offset
            self.rect = (float(pos[0]) - ox, float(pos[1]) - oy, w, h)
            return True

        self.update_hover(pos, scene)
        return self.hovered

    def handle_mouse_up(
        self,
        pos,
        scene,
        *,
        play_rect: tuple[float, float, float, float],
    ) -> bool:
        if not self.dragging:
            self.update_hover(pos, scene)
            return False

        self.handle_mouse_motion(pos, scene)
        should_play = self.enabled_for(scene) and self._center_in_rect(
            self.rect,
            self._coerce_rect(play_rect),
        )
        self.reset_to_home()

        if should_play:
            self.on_play(scene)
        return True

    def draw(self, text, *, enabled: bool = True) -> None:  # pragma: no cover - visual
        x, y, w, h = self.rect
        style = self.style
        raised = self.hovered or self.dragging
        border = style.border_hover if raised and enabled else style.border
        face = style.face_hover if raised and enabled else style.face
        accent = style.accent_hover if raised and enabled else style.accent
        alpha_scale = 1.0 if enabled else 0.55

        def scaled(color):
            return (color[0], color[1], color[2], color[3] * alpha_scale)

        glDisable(GL_TEXTURE_2D)
        self._draw_rect(x + 6.0, y + 8.0, w, h, scaled(style.shadow))
        self._draw_rect(x, y, w, h, scaled(border))
        self._draw_rect(x + 4.0, y + 4.0, w - 8.0, h - 8.0, scaled(face))
        self._draw_rect(x + 10.0, y + 34.0, w - 20.0, h * 0.44, scaled(style.panel))
        self._draw_rect(x + 10.0, y + h - 32.0, w - 20.0, 18.0, scaled(accent))
        if not enabled:
            self._draw_rect(x + 4.0, y + 4.0, w - 8.0, h - 8.0, style.disabled_tint)

        glEnable(GL_TEXTURE_2D)
        text_alpha = 255 if enabled else 150
        text.draw_text(
            self.title,
            x + w * 0.5,
            y + 19.0,
            color=(255, 242, 220, text_alpha),
            align="center",
        )
        text.draw_text(
            self.detail,
            x + w * 0.5,
            y + h * 0.52,
            color=(245, 238, 224, text_alpha),
            align="center",
        )
        if self.footer:
            text.draw_text(
                self.footer,
                x + w * 0.5,
                y + h - 23.0,
                color=(255, 248, 232, text_alpha),
                align="center",
            )
