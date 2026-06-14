"""Input forwarding for world UI surfaces."""

from __future__ import annotations

from game.config import HEIGHT, WIDTH


class WorldUIInteractions:
    """Route UI input without coupling it to the OpenGL renderer."""

    def __init__(self, scene) -> None:
        self.scene = scene

    @staticmethod
    def inventory_panel_rect(
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> tuple[float, float, float, float]:
        outer_w = min(float(width) - 72.0, 930.0)
        outer_h = min(float(height) - 72.0, 560.0)
        outer_x = (float(width) - outer_w) * 0.5
        outer_y = (float(height) - outer_h) * 0.5
        return outer_x, outer_y, outer_w, outer_h

    @classmethod
    def inventory_close_rect(
        cls,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> tuple[float, float, float, float]:
        outer_x, outer_y, outer_w, _outer_h = cls.inventory_panel_rect(width, height)
        size = 34.0
        return outer_x + outer_w - size - 20.0, outer_y + 18.0, size, size

    def active_pause_menu(self):
        if getattr(self.scene, "showing_settings_menu", False):
            return getattr(self.scene, "setting_menu", None)
        return getattr(self.scene, "pause_menu", None)

    def compute_pause_buttons(self, width: int = WIDTH, height: int = HEIGHT):
        menu = self.active_pause_menu()
        if menu is None:
            return []
        return menu.compute_buttons(width=width, height=height)

    def compute_battle_buttons(self, width: int = WIDTH, height: int = HEIGHT):
        return []

    def handle_inventory_click(self, pos) -> bool:
        mx, my = pos
        x, y, w, h = self.inventory_close_rect()
        if not (x <= mx <= x + w and y <= my <= y + h):
            return False

        self.scene.inventory_open = False
        self.scene.paused = False
        self.scene.showing_settings_menu = False
        self._set_mouse_captured()
        return True

    def handle_battle_click(self, pos) -> bool:
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        if battle_overlay is not None:
            return bool(battle_overlay.handle_mouse_down(pos))
        return False

    def handle_battle_motion(self, pos) -> bool:
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        if battle_overlay is not None:
            return bool(battle_overlay.handle_mouse_motion(pos))
        return False

    def handle_battle_release(self, pos) -> bool:
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        if battle_overlay is not None:
            return bool(battle_overlay.handle_mouse_up(pos))
        return False

    def handle_pause_click(self, pos) -> None:
        menu = self.active_pause_menu()
        if menu is not None:
            menu.handle_click(pos)

    def handle_pause_motion(self, pos) -> None:
        menu = self.active_pause_menu()
        if menu is not None:
            menu.handle_motion(pos)

    def handle_pause_release(self, pos) -> None:
        menu = self.active_pause_menu()
        if menu is not None:
            menu.handle_release(pos)

    @staticmethod
    def _set_mouse_captured() -> None:
        import pygame

        try:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        except pygame.error:
            pass
