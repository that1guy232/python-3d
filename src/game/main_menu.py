"""Main-menu scene shown before entering the world."""

from __future__ import annotations

import math

import pygame
from OpenGL.GL import (
    glBegin,
    glClear,
    glClearColor,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glVertex2f,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FOG,
    GL_QUADS,
    GL_TEXTURE_2D,
)

from engine.config import HEIGHT, WIDTH
from engine.core.loading_scene import LoadingScene
from engine.core.scene import Scene
from engine.ui.menu import ButtonMenu, MenuOption
from game.world.worldscene import WorldScene


def _draw_rect(x: float, y: float, w: float, h: float, color: tuple[float, ...]) -> None:
    glColor4f(*color)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


class MainMenu(ButtonMenu):
    button_height = 58
    button_spacing = 16
    button_width_max = 340

    def __init__(self, scene) -> None:
        super().__init__(scene)
        self.buttons = [
            MenuOption("start_game", "Start Game", self.start_game),
        ]

    def options(self) -> list[MenuOption]:
        return self.buttons

    def start_game(self, scene) -> None:
        scene.start_game()


class MainMenuScene(Scene):
    """Simple first scene with a button that starts world loading."""

    mouse_visible = True
    mouse_grabbed = False

    def __init__(self) -> None:
        super().__init__()
        self.menu = MainMenu(self)
        self.next_scene = None
        self.elapsed = 0.0

    def start_game(self) -> None:
        if self.next_scene is not None:
            return
        world = WorldScene(defer_setup=True)
        self.next_scene = LoadingScene(
            world,
            title="Loading World",
            initial_status="Preparing world",
        )

    def update(self, dt: float) -> None:
        self.elapsed += dt

    def handle_event(self, event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) == 1:
            self.menu.handle_click(getattr(event, "pos", pygame.mouse.get_pos()))
            return

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.start_game()
            elif event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

    def render(
        self, *, show_hud: bool = True, text=None, fps: float | None = None
    ) -> None:  # pragma: no cover - visual
        glDisable(GL_FOG)
        glDisable(GL_DEPTH_TEST)
        glClearColor(0.035, 0.045, 0.042, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if text is None:
            return

        pulse = 0.5 + 0.5 * math.sin(self.elapsed * 2.4)
        buttons = self.menu.compute_buttons()
        mx, my = pygame.mouse.get_pos()

        text.begin()
        glDisable(GL_TEXTURE_2D)

        _draw_rect(0, 0, WIDTH, HEIGHT, (0.035, 0.045, 0.042, 1.0))
        _draw_rect(0, HEIGHT * 0.62, WIDTH, HEIGHT * 0.38, (0.022, 0.027, 0.026, 1.0))
        _draw_rect(0, HEIGHT * 0.62 - 2, WIDTH, 4, (0.38, 0.58, 0.49, 0.38))

        mark_w = min(560.0, WIDTH * 0.46)
        mark_x = (WIDTH - mark_w) * 0.5
        mark_y = HEIGHT * 0.25
        _draw_rect(mark_x, mark_y, mark_w, 2, (0.55, 0.72, 0.64, 0.35 + pulse * 0.2))
        _draw_rect(
            mark_x + mark_w * 0.16,
            mark_y + 92,
            mark_w * 0.68,
            2,
            (0.55, 0.72, 0.64, 0.18),
        )

        for button in buttons:
            x, y, w, h = button["rect"]
            hovered = x <= mx <= x + w and y <= my <= y + h
            border_alpha = 0.95 if hovered else 0.55
            fill = (0.17, 0.24, 0.21, 0.98) if hovered else (0.11, 0.14, 0.13, 0.94)
            _draw_rect(x - 3, y - 3, w + 6, h + 6, (0.55, 0.72, 0.64, border_alpha))
            _draw_rect(x, y, w, h, fill)
            _draw_rect(x + 2, y + 2, w - 4, 12, (0.42, 0.55, 0.49, 0.22))

        glEnable(GL_TEXTURE_2D)
        text.draw_text(
            "MAIN MENU",
            WIDTH * 0.5,
            HEIGHT * 0.31,
            color=(238, 244, 239, 255),
            align="center",
        )

        for button in buttons:
            x, y, w, h = button["rect"]
            hovered = x <= mx <= x + w and y <= my <= y + h
            text.draw_text(
                button["label"],
                x + w * 0.5,
                y + h * 0.5,
                color=(255, 255, 255, 255) if hovered else (226, 236, 230, 255),
                align="center",
            )

        text.end()
