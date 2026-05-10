"""World pause-menu options and actions."""

from __future__ import annotations

import pygame

from engine.ui.menu import ButtonMenu, MenuOption


class PauseMenu(ButtonMenu):
    def __init__(self, scene) -> None:
        super().__init__(scene)
        self.buttons = [
            MenuOption("resume", "Resume", self.resume),
            MenuOption("settings", "Settings", self.open_settings),
            MenuOption("exit", "Exit", self.exit_game),
        ]

    def options(self) -> list[MenuOption]:
        return self.buttons

    def resume(self, scene) -> None:
        scene.paused = False
        scene.showing_settings_menu = False
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def open_settings(self, scene) -> None:
        setting_menu = getattr(scene, "setting_menu", None)
        if setting_menu is not None:
            setting_menu.page = 0
        scene.showing_settings_menu = True

    def exit_game(self, scene) -> None:
        pygame.event.post(pygame.event.Event(pygame.QUIT))
