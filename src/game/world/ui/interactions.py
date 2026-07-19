"""Input forwarding for world UI surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from game.config import HEIGHT, WIDTH
from game.world.inventory import (
    INVENTORY_SLOT_COUNT,
    inventory_slots,
    move_inventory_item,
)


@dataclass(frozen=True, slots=True)
class InventoryLayout:
    outer_rect: tuple[float, float, float, float]
    close_rect: tuple[float, float, float, float]
    grid_rect: tuple[float, float, float, float]
    stats_rect: tuple[float, float, float, float]
    slot_rects: tuple[tuple[float, float, float, float], ...]


class WorldUIInteractions:
    """Route UI input without coupling it to the OpenGL renderer."""

    def __init__(self, scene) -> None:
        self.scene = scene

    @staticmethod
    def inventory_panel_rect(
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> tuple[float, float, float, float]:
        return WorldUIInteractions.inventory_layout(width, height).outer_rect

    @staticmethod
    def inventory_layout(
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> InventoryLayout:
        outer_w = min(float(width) - 72.0, 930.0)
        outer_h = min(float(height) - 72.0, 560.0)
        outer_x = (float(width) - outer_w) * 0.5
        outer_y = (float(height) - outer_h) * 0.5
        padding = 24.0
        gap = 22.0
        stats_w = min(280.0, max(230.0, outer_w * 0.31))
        grid_x = outer_x + padding
        grid_y = outer_y + 82.0
        grid_w = outer_w - padding * 2.0 - stats_w - gap
        stats_h = outer_h - 108.0
        stats_x = grid_x + grid_w + gap
        slot_gap = 10.0
        cols = 6
        rows = INVENTORY_SLOT_COUNT // cols
        slot_size = min(
            72.0,
            max(
                44.0,
                min(
                    (grid_w - slot_gap * (cols - 1)) / cols,
                    (stats_h - slot_gap * (rows - 1)) / rows,
                ),
            ),
        )
        grid_h = rows * slot_size + (rows - 1) * slot_gap
        close_size = 34.0
        close_rect = (
            outer_x + outer_w - close_size - 20.0,
            outer_y + 18.0,
            close_size,
            close_size,
        )
        slot_rects = tuple(
            (
                grid_x + (index % cols) * (slot_size + slot_gap),
                grid_y + (index // cols) * (slot_size + slot_gap),
                slot_size,
                slot_size,
            )
            for index in range(INVENTORY_SLOT_COUNT)
        )
        return InventoryLayout(
            outer_rect=(outer_x, outer_y, outer_w, outer_h),
            close_rect=close_rect,
            grid_rect=(grid_x, grid_y, grid_w, grid_h),
            stats_rect=(stats_x, outer_y + 62.0, stats_w, outer_h - 86.0),
            slot_rects=slot_rects,
        )

    @classmethod
    def inventory_close_rect(
        cls,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> tuple[float, float, float, float]:
        return cls.inventory_layout(width, height).close_rect

    @classmethod
    def inventory_slot_at(
        cls,
        pos,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> int | None:
        mx, my = pos
        for index, (x, y, w, h) in enumerate(
            cls.inventory_layout(width, height).slot_rects
        ):
            if x <= mx <= x + w and y <= my <= y + h:
                return index
        return None

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
        if x <= mx <= x + w and y <= my <= y + h:
            self.scene.inventory_selected_slot = None
            self.scene.inventory_open = False
            self.scene.paused = False
            self.scene.showing_settings_menu = False
            self._set_mouse_captured()
            return True

        clicked = self.inventory_slot_at(pos)
        if clicked is None:
            return False

        selected = getattr(self.scene, "inventory_selected_slot", None)
        if selected is None:
            if inventory_slots(self.scene)[clicked] is None:
                return False
            self.scene.inventory_selected_slot = clicked
            return True

        if selected == clicked:
            self.scene.inventory_selected_slot = None
            return True

        move_inventory_item(self.scene, selected, clicked)
        self.scene.inventory_selected_slot = None
        return True

    def handle_inventory_release(self, pos) -> bool:
        selected = getattr(self.scene, "inventory_selected_slot", None)
        released = self.inventory_slot_at(pos)
        if selected is None or released is None or selected == released:
            return False
        moved = move_inventory_item(self.scene, selected, released)
        self.scene.inventory_selected_slot = None
        return moved

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
