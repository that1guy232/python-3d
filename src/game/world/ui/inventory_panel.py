"""Inventory panel drawing for the world HUD."""

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
from game.world.ui.interactions import WorldUIInteractions


class InventoryPanel:
    """Draw the inventory overlay without making the world renderer own it."""

    def __init__(self, scene) -> None:
        self.scene = scene

    @staticmethod
    def _draw_overlay_rect(
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
    def _item_label(item) -> str:
        if item is None:
            return ""
        if isinstance(item, dict):
            for key in ("name", "label", "title", "id"):
                value = item.get(key)
                if value:
                    return str(value)
            return ""
        name = getattr(item, "name", None) or getattr(item, "label", None)
        if name:
            return str(name)
        return str(item)

    @staticmethod
    def _fit_text_width(text, label: str, max_width: float) -> str:
        if not label:
            return ""
        try:
            if text.font.size(label)[0] <= max_width:
                return label
            for length in range(len(label) - 1, 0, -1):
                candidate = label[:length] + "."
                if text.font.size(candidate)[0] <= max_width:
                    return candidate
        except Exception:
            return label[:8]
        return ""

    def _player_stat_rows(self) -> list[tuple[str, str]]:
        stats = getattr(self.scene, "player_stats", None)
        if stats is None:
            return []

        hp = max(0, int(getattr(stats, "hp", 5)))
        max_hp = max(1, int(getattr(stats, "max_hp", max(1, hp))))
        mana = max(0, int(getattr(stats, "mana", 5)))
        max_mana = max(1, int(getattr(stats, "max_mana", max(1, mana))))
        strength = max(0, int(getattr(stats, "strength", 1)))
        dexterity = max(0, int(getattr(stats, "dexterity", 1)))
        elemental = max(0, int(getattr(stats, "elemental_damage", 0)))
        card_draw = max(0, int(getattr(stats, "card_draw", 1)))

        crit_percent = getattr(stats, "crit_percent", None)
        if callable(crit_percent):
            crit = int(crit_percent())
        else:
            crit = int(round(max(0.0, float(getattr(stats, "crit_chance", 0.0)))))

        return [
            ("HP", f"{hp}/{max_hp}"),
            ("Mana", f"{mana}/{max_mana}"),
            ("Strength", str(strength)),
            ("Dexterity", str(dexterity)),
            ("Crit Chance", f"{crit}%"),
            ("Elemental Damage", str(elemental)),
            ("Card Draw", str(card_draw)),
        ]

    def draw(
        self, text, fps_label: str, *, profile=None
    ) -> None:  # pragma: no cover - visual
        import pygame

        text.begin()
        try:
            if profile is None:
                text.draw_text(
                    fps_label,
                    12,
                    10,
                    key="fps",
                    align="topleft",
                    color=[255, 0, 0, 0],
                )
            else:
                with profile("hud_text.fps"):
                    text.draw_text(
                        fps_label,
                        12,
                        10,
                        key="fps",
                        align="topleft",
                        color=[255, 0, 0, 0],
                    )

            outer_x, outer_y, outer_w, outer_h = (
                WorldUIInteractions.inventory_panel_rect()
            )
            padding = 24.0
            gap = 22.0
            stats_w = min(280.0, max(230.0, outer_w * 0.31))
            grid_x = outer_x + padding
            grid_y = outer_y + 82.0
            grid_w = outer_w - padding * 2.0 - stats_w - gap
            stats_x = grid_x + grid_w + gap
            stats_y = grid_y
            stats_h = outer_h - 108.0

            rows = self._player_stat_rows()
            items = list(getattr(self.scene, "inventory_items", ()) or ())
            cols = 6
            visible_rows = 4
            slot_gap = 10.0
            slot_size = min(
                72.0,
                max(
                    44.0,
                    min(
                        (grid_w - slot_gap * (cols - 1)) / cols,
                        (stats_h - slot_gap * (visible_rows - 1)) / visible_rows,
                    ),
                ),
            )
            grid_h = visible_rows * slot_size + (visible_rows - 1) * slot_gap
            slot_count = cols * visible_rows
            close_x, close_y, close_w, close_h = (
                WorldUIInteractions.inventory_close_rect()
            )
            mx, my = pygame.mouse.get_pos()
            close_hovered = (
                close_x <= mx <= close_x + close_w
                and close_y <= my <= close_y + close_h
            )

            glDisable(GL_TEXTURE_2D)
            self._draw_overlay_rect(0, 0, WIDTH, HEIGHT, (0.0, 0.0, 0.0, 0.55))
            self._draw_overlay_rect(
                outer_x,
                outer_y,
                outer_w,
                outer_h,
                (0.035, 0.04, 0.045, 0.96),
            )
            self._draw_overlay_rect(
                outer_x + 4.0,
                outer_y + 4.0,
                outer_w - 8.0,
                outer_h - 8.0,
                (0.085, 0.075, 0.065, 0.9),
            )
            self._draw_overlay_rect(
                grid_x - 12.0,
                grid_y - 14.0,
                grid_w + 24.0,
                grid_h + 28.0,
                (0.025, 0.026, 0.03, 0.72),
            )
            self._draw_overlay_rect(
                stats_x,
                outer_y + 62.0,
                stats_w,
                outer_h - 86.0,
                (0.025, 0.026, 0.03, 0.72),
            )
            self._draw_overlay_rect(
                close_x,
                close_y,
                close_w,
                close_h,
                (
                    (0.30, 0.12, 0.10, 0.96)
                    if close_hovered
                    else (0.12, 0.08, 0.075, 0.92)
                ),
            )
            self._draw_overlay_rect(
                close_x + 3.0,
                close_y + 3.0,
                close_w - 6.0,
                close_h - 6.0,
                (0.48, 0.18, 0.14, 0.74) if close_hovered else (0.22, 0.13, 0.11, 0.64),
            )

            for index in range(slot_count):
                col = index % cols
                row = index // cols
                x = grid_x + col * (slot_size + slot_gap)
                y = grid_y + row * (slot_size + slot_gap)
                filled = index < len(items)
                self._draw_overlay_rect(
                    x,
                    y,
                    slot_size,
                    slot_size,
                    (
                        (0.12, 0.105, 0.09, 0.96)
                        if filled
                        else (0.06, 0.058, 0.055, 0.92)
                    ),
                )
                self._draw_overlay_rect(
                    x + 3.0,
                    y + 3.0,
                    slot_size - 6.0,
                    slot_size - 6.0,
                    ((0.23, 0.18, 0.12, 0.54) if filled else (0.11, 0.105, 0.1, 0.5)),
                )

            glEnable(GL_TEXTURE_2D)
            text.draw_text(
                "Inventory",
                outer_x + padding,
                outer_y + 24.0,
                color=(255, 245, 230, 255),
                align="topleft",
            )
            text.draw_text(
                "Stats",
                stats_x + 16.0,
                outer_y + 24.0,
                color=(255, 245, 230, 255),
                align="topleft",
            )
            text.draw_text(
                "X",
                close_x + close_w * 0.5,
                close_y + close_h * 0.5,
                color=(255, 245, 230, 255),
                align="center",
            )

            for index, item in enumerate(items[:slot_count]):
                label = self._item_label(item)
                if not label:
                    continue
                col = index % cols
                row = index // cols
                x = grid_x + col * (slot_size + slot_gap)
                y = grid_y + row * (slot_size + slot_gap)
                label = self._fit_text_width(text, label, slot_size - 10.0)
                if not label:
                    continue
                text.draw_text(
                    label,
                    x + slot_size * 0.5,
                    y + slot_size * 0.5,
                    color=(245, 235, 215, 255),
                    align="center",
                )

            stat_line_h = 38.0
            for index, (label, value) in enumerate(rows):
                y = stats_y + index * stat_line_h
                text.draw_text(
                    label,
                    stats_x + 18.0,
                    y,
                    color=(210, 214, 220, 255),
                    align="topleft",
                )
                text.draw_text(
                    value,
                    stats_x + stats_w - 18.0,
                    y,
                    color=(255, 245, 230, 255),
                    align="topright",
                )
        finally:
            text.end()
