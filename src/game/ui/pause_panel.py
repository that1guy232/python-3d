"""Pause and settings menu drawing for the world HUD."""

from __future__ import annotations

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

from game.config import HEIGHT, WIDTH
from game.ui.interactions import WorldUIInteractions


class PauseMenuPanel:
    """Draw the active pause/settings menu."""

    def __init__(self, scene) -> None:
        self.scene = scene
        self.interactions = WorldUIInteractions(scene)

    @staticmethod
    def slider_track(
        rect,
        *,
        padding: float,
        ratio: float,
    ) -> tuple[float, float, float, float, float]:
        x, y, w, h = rect
        track_x = x + padding
        track_w = max(1, w - padding * 2)
        track_y = y + h - 12
        track_h = 5
        fill_w = track_w * max(0.0, min(1.0, float(ratio)))
        knob_x = track_x + fill_w
        return track_x, track_y, track_w, track_h, knob_x

    def active_menu(self):
        return self.interactions.active_pause_menu()

    def draw(self, text) -> None:  # pragma: no cover - visual
        import pygame

        text.begin()
        glDisable(GL_TEXTURE_2D)
        glColor4f(0.0, 0.0, 0.0, 0.6)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(WIDTH, 0)
        glVertex2f(WIDTH, HEIGHT)
        glVertex2f(0, HEIGHT)
        glEnd()

        menu = self.active_menu()
        if menu is None:
            glEnable(GL_TEXTURE_2D)
            text.end()
            return

        buttons = menu.compute_buttons()
        mx, my = pygame.mouse.get_pos()
        for button in buttons:
            x, y, w, h = button["rect"]
            hovered = x <= mx <= x + w and y <= my <= y + h
            is_slider = button.get("type") == "slider"

            glColor4f(0.12, 0.12, 0.12, 0.95 if hovered else 0.85)
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + w, y)
            glVertex2f(x + w, y + h)
            glVertex2f(x, y + h)
            glEnd()

            glColor4f(0.2, 0.2, 0.2, 0.9 if hovered else 0.75)
            glBegin(GL_QUADS)
            glVertex2f(x + 2, y + 2)
            glVertex2f(x + w - 2, y + 2)
            glVertex2f(x + w - 2, y + h - 2)
            glVertex2f(x + 2, y + h - 2)
            glEnd()

            if is_slider:
                padding = getattr(menu, "slider_horizontal_padding", 18)
                track_x, track_y, track_w, track_h, knob_x = self.slider_track(
                    button["rect"],
                    padding=padding,
                    ratio=button.get("ratio", 0.0),
                )
                fill_w = knob_x - track_x

                glColor4f(0.08, 0.08, 0.08, 0.9)
                glBegin(GL_QUADS)
                glVertex2f(track_x, track_y)
                glVertex2f(track_x + track_w, track_y)
                glVertex2f(track_x + track_w, track_y + track_h)
                glVertex2f(track_x, track_y + track_h)
                glEnd()

                glColor4f(0.35, 0.58, 0.86, 0.95)
                glBegin(GL_QUADS)
                glVertex2f(track_x, track_y)
                glVertex2f(track_x + fill_w, track_y)
                glVertex2f(track_x + fill_w, track_y + track_h)
                glVertex2f(track_x, track_y + track_h)
                glEnd()

                glColor4f(0.88, 0.92, 0.98, 1.0)
                glBegin(GL_QUADS)
                glVertex2f(knob_x - 5, track_y - 5)
                glVertex2f(knob_x + 5, track_y - 5)
                glVertex2f(knob_x + 5, track_y + track_h + 5)
                glVertex2f(knob_x - 5, track_y + track_h + 5)
                glEnd()

        glEnable(GL_TEXTURE_2D)
        for button in buttons:
            x, y, w, h = button["rect"]
            if button.get("type") == "slider":
                padding = getattr(menu, "slider_horizontal_padding", 18)
                text.draw_text(
                    button["label"],
                    x + padding,
                    y + 6,
                    color=(255, 255, 255, 255),
                    align="topleft",
                )
                text.draw_text(
                    button.get("value_text", ""),
                    x + w - padding,
                    y + 6,
                    color=(220, 230, 245, 255),
                    align="topright",
                )
            else:
                text.draw_text(
                    button["label"],
                    x + w / 2,
                    y + h / 2,
                    color=(255, 255, 255, 255),
                    align="center",
                )

        title = getattr(menu, "title", None)
        if title and buttons:
            text.draw_text(
                title,
                WIDTH // 2,
                buttons[0]["rect"][1] - 40,
                color=(230, 230, 230, 255),
                align="center",
            )

        text.end()
