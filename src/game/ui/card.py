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

from game.ui.theme import (
    BLUE,
    CREAM_TEXT,
    FOREST,
    FOREST_LIGHT,
    GOLD,
    GOLD_DARK,
    GOLD_LIGHT,
    INK_TEXT,
    PAPER,
    PAPER_DARK,
    PAPER_LIGHT,
    draw_frame,
    draw_pixel_icon,
)

CardHandler = Callable[[object], None]
CardPredicate = Callable[[object], bool]


@dataclass(frozen=True)
class CardStyle:
    shadow: tuple[float, float, float, float] = (0.10, 0.08, 0.04, 0.26)
    border: tuple[float, float, float, float] = GOLD_DARK
    border_hover: tuple[float, float, float, float] = GOLD
    face: tuple[float, float, float, float] = PAPER
    face_hover: tuple[float, float, float, float] = PAPER_LIGHT
    panel: tuple[float, float, float, float] = PAPER_DARK
    accent: tuple[float, float, float, float] = FOREST
    accent_hover: tuple[float, float, float, float] = FOREST_LIGHT
    disabled_tint: tuple[float, float, float, float] = (0.22, 0.20, 0.14, 0.42)
    trim: tuple[float, float, float, float] = GOLD_LIGHT
    mana: tuple[float, float, float, float] = BLUE


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

    def draw_at(
        self,
        text,
        rect: tuple[float, float, float, float],
        *,
        enabled: bool = True,
        raised: bool = False,
    ) -> None:  # pragma: no cover - visual
        """Draw a non-mutating card preview inside an explicit rectangle."""

        x, y, w, h = self._coerce_rect(rect)
        style = self.style
        border = style.border_hover if raised and enabled else style.border
        face = style.face_hover if raised and enabled else style.face
        accent = style.accent_hover if raised and enabled else style.accent
        scale = min(w / 150.0, h / 207.0)
        edge = 10.0 * scale
        header_y = y + edge
        header_h = 28.0 * scale
        art_y = header_y + header_h + 6.0 * scale
        art_h = 66.0 * scale
        footer_h = 22.0 * scale
        footer_y = y + h - edge - footer_h
        detail_y = art_y + art_h + 6.0 * scale
        detail_h = max(1.0, footer_y - detail_y - 7.0 * scale)
        emblem, emblem_color = self._emblem()

        glDisable(GL_TEXTURE_2D)
        self._draw_rect(x + 4.0 * scale, y + 6.0 * scale, w, h, style.shadow)
        draw_frame(
            (x, y, w, h), fill=face, border=border, light=style.trim,
            shadow=False, inset=4.0 * scale,
        )
        self._draw_rect(x + edge, header_y, w - 2.0 * edge, header_h, accent)
        self._draw_rect(x + edge, art_y, w - 2.0 * edge, art_h, style.panel)
        # A small pixel emblem gives every action a readable visual identity.
        icon_size = 44.0 * scale
        icon_x = x + (w - icon_size) * 0.5
        icon_y = art_y + (art_h - icon_size) * 0.5
        draw_pixel_icon(emblem, (icon_x, icon_y, icon_size, icon_size), color=emblem_color)
        for ornament_x in (x + edge + 9.0 * scale, x + w - edge - 13.0 * scale):
            self._draw_rect(ornament_x, art_y + art_h * 0.5 - 2.0 * scale,
                            4.0 * scale, 4.0 * scale, style.trim)
        self._draw_rect(x + edge, art_y + art_h, w - 2.0 * edge, scale, style.border)
        if self.footer:
            self._draw_rect(x + edge, footer_y, w - 2.0 * edge, footer_h, style.mana)
        if not enabled:
            self._draw_rect(x, y, w, h, style.disabled_tint)

        glEnable(GL_TEXTURE_2D)
        text_alpha = 255 if enabled else 175
        text.draw_text(
            self.title,
            x + w * 0.5,
            header_y + header_h * 0.5,
            color=(*CREAM_TEXT, text_alpha),
            align="center",
            max_width=max(1.0, w - 2.0 * edge - 8.0 * scale),
            max_height=20.0 * scale,
        )
        text.draw_text_multiline(
            self.detail,
            x + w * 0.5,
            detail_y + detail_h * 0.5,
            color=(*INK_TEXT, text_alpha),
            align="center",
            line_spacing=1.08,
            max_width=max(1.0, w - 2.0 * edge - 6.0 * scale),
            max_height=detail_h,
            wrap=True,
        )
        if self.footer:
            text.draw_text(
                self.footer,
                x + w * 0.5,
                footer_y + footer_h * 0.5,
                color=(*CREAM_TEXT, text_alpha),
                align="center",
                max_width=max(1.0, w - 2.0 * edge - 8.0 * scale),
                max_height=16.0 * scale,
            )

    def _emblem(self) -> tuple[str, tuple[float, float, float, float]]:
        action = self.action.lower()
        if any(word in action for word in ("brace", "guard", "shell")):
            return "shield", FOREST
        if any(word in action for word in ("step", "scamper", "scrounge")):
            return "boot", GOLD_DARK
        if any(word in action for word in ("thought", "spell", "bomb")):
            return "spark", BLUE
        if any(word in action for word in ("strike", "swipe", "shiv", "attack")):
            return "sword", GOLD_DARK
        return "cards", FOREST

    def draw(self, text, *, enabled: bool = True) -> None:  # pragma: no cover - visual
        self.draw_at(
            text,
            self.rect,
            enabled=enabled,
            raised=self.hovered or self.dragging,
        )
