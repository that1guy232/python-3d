"""Input forwarding for world UI surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from game.config import HEIGHT, WIDTH
from game.inventory import (
    BACKPACK_SLOT_COUNT,
    EQUIPMENT_SLOT_COUNT,
    inventory_slots,
    move_inventory_item,
)


@dataclass(frozen=True, slots=True)
class InventoryLayout:
    outer_rect: tuple[float, float, float, float]
    close_rect: tuple[float, float, float, float]
    deck_rect: tuple[float, float, float, float]
    grid_rect: tuple[float, float, float, float]
    stats_rect: tuple[float, float, float, float]
    slot_rects: tuple[tuple[float, float, float, float], ...]
    equipment_slot_rects: tuple[tuple[float, float, float, float], ...] = ()


class WorldUIInteractions:
    """Route UI input without coupling it to the OpenGL renderer."""

    def __init__(self, scene) -> None:
        self.scene = scene
        self._inventory_deck_pressed = False

    @property
    def inventory_deck_pressed(self) -> bool:
        return self._inventory_deck_pressed

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
        # One scaled layout owns both drawing and hit boxes. Keep the complete
        # panel on smaller displays without squeezing the stats into the slots.
        scale = min(1.0, (float(width) - 40.0) / 960.0,
                    (float(height) - 40.0) / 620.0)
        scale = max(0.1, scale)
        outer_w, outer_h = 960.0 * scale, 620.0 * scale
        outer_x = (float(width) - outer_w) * 0.5
        outer_y = (float(height) - outer_h) * 0.5

        def rect(x, y, w, h):
            return (outer_x + x * scale, outer_y + y * scale,
                    w * scale, h * scale)

        equipment_slot_rects = tuple(
            rect(167.0 + index * 84.0, 104.0, 70.0, 70.0)
            for index in range(EQUIPMENT_SLOT_COUNT)
        )
        backpack_slot_rects = tuple(
            rect(69.0 + (index % 6) * 88.0,
                 230.0 + (index // 6) * 88.0, 78.0, 78.0)
            for index in range(BACKPACK_SLOT_COUNT)
        )
        return InventoryLayout(
            outer_rect=(outer_x, outer_y, outer_w, outer_h),
            close_rect=rect(898.0, 20.0, 34.0, 34.0),
            deck_rect=rect(734.0, 20.0, 150.0, 34.0),
            grid_rect=rect(69.0, 230.0, 518.0, 342.0),
            stats_rect=rect(652.0, 76.0, 280.0, 504.0),
            slot_rects=backpack_slot_rects + equipment_slot_rects,
            equipment_slot_rects=equipment_slot_rects,
        )

    @classmethod
    def inventory_close_rect(
        cls,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> tuple[float, float, float, float]:
        return cls.inventory_layout(width, height).close_rect

    @classmethod
    def inventory_deck_rect(
        cls,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> tuple[float, float, float, float]:
        return cls.inventory_layout(width, height).deck_rect

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
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        sync_overlay = getattr(battle_overlay, "sync_state", None)
        if callable(sync_overlay):
            sync_overlay()
        handle_modal_down = getattr(
            battle_overlay,
            "handle_deck_view_mouse_down",
            None,
        )
        if callable(handle_modal_down) and handle_modal_down(pos):
            self._inventory_deck_pressed = False
            self.scene.inventory_drag_source = None
            return True

        mx, my = pos
        deck_x, deck_y, deck_w, deck_h = self.inventory_deck_rect()
        if deck_x <= mx <= deck_x + deck_w and deck_y <= my <= deck_y + deck_h:
            self._inventory_deck_pressed = True
            self.scene.inventory_drag_source = None
            return True

        self._inventory_deck_pressed = False
        x, y, w, h = self.inventory_close_rect()
        if x <= mx <= x + w and y <= my <= y + h:
            self.scene.inventory_selected_slot = None
            self.scene.inventory_drag_source = None
            self.scene.inventory_open = False
            self.scene.paused = False
            self.scene.showing_settings_menu = False
            close_deck_view = getattr(battle_overlay, "close_deck_view", None)
            if callable(close_deck_view):
                close_deck_view()
            self._set_mouse_captured()
            return True

        clicked = self.inventory_slot_at(pos)
        if clicked is None:
            self.scene.inventory_selected_slot = None
            self.scene.inventory_drag_source = None
            return False

        if inventory_slots(self.scene)[clicked] is None:
            self.scene.inventory_selected_slot = None
            self.scene.inventory_drag_source = None
            return True

        self.scene.inventory_selected_slot = clicked
        self.scene.inventory_drag_source = clicked
        return True

    def handle_inventory_release(self, pos) -> bool:
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        sync_overlay = getattr(battle_overlay, "sync_state", None)
        if callable(sync_overlay):
            sync_overlay()
        handle_modal_up = getattr(
            battle_overlay,
            "handle_deck_view_mouse_up",
            None,
        )
        if callable(handle_modal_up) and handle_modal_up(pos):
            self._inventory_deck_pressed = False
            self.scene.inventory_drag_source = None
            return True

        if self._inventory_deck_pressed:
            self._inventory_deck_pressed = False
            self.scene.inventory_drag_source = None
            deck_x, deck_y, deck_w, deck_h = self.inventory_deck_rect()
            mx, my = pos
            if (
                deck_x <= mx <= deck_x + deck_w
                and deck_y <= my <= deck_y + deck_h
            ):
                open_loadout_view = getattr(
                    battle_overlay,
                    "open_loadout_view",
                    None,
                )
                if callable(open_loadout_view):
                    return bool(open_loadout_view())
                return True
            return True

        source = getattr(self.scene, "inventory_drag_source", None)
        self.scene.inventory_drag_source = None
        released = self.inventory_slot_at(pos)
        if source is None or released is None or source == released:
            return False
        moved = move_inventory_item(self.scene, source, released)
        self.scene.inventory_selected_slot = released if moved else source
        return moved

    def handle_inventory_escape(self) -> bool:
        """Close the deck modal before allowing Escape to close inventory."""

        self._inventory_deck_pressed = False
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        close_deck_view = getattr(battle_overlay, "close_deck_view", None)
        return bool(close_deck_view()) if callable(close_deck_view) else False

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

    def handle_battle_escape(self) -> bool:
        battle_overlay = getattr(self.scene, "battle_overlay", None)
        close_deck_view = getattr(battle_overlay, "close_deck_view", None)
        return bool(close_deck_view()) if callable(close_deck_view) else False

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
