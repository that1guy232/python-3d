"""Inventory panel drawing for the world HUD."""

from __future__ import annotations

from OpenGL.GL import (
    glBegin,
    glBindTexture,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glTexCoord2f,
    glVertex2f,
    GL_QUADS,
    GL_TEXTURE_2D,
)

from engine.textures.texture_utils import get_texture_aspect
from game.config import HEIGHT, WIDTH
from game.inventory import (
    BACKPACK_SLOT_COUNT,
    EQUIPMENT_TYPES,
    item_type,
    slot_accepts_item,
)
from game.ui.interactions import WorldUIInteractions
from game.ui import theme


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
    def _draw_textured_rect(
        texture,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        alpha: float = 1.0,
    ) -> None:
        if not texture:
            return
        glBindTexture(GL_TEXTURE_2D, int(texture))
        glColor4f(1.0, 1.0, 1.0, alpha)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x, y)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x + w, y)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x + w, y + h)
        glTexCoord2f(0.0, 0.0)
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
    def _item_icon(item) -> str:
        if item is None:
            return ""
        if isinstance(item, dict):
            return str(item.get("icon", item.get("icon_key", "")) or "")
        return str(getattr(item, "icon", "") or "")

    @staticmethod
    def _icon_dimensions(texture, max_size: float) -> tuple[float, float]:
        aspect = max(0.01, float(get_texture_aspect(texture)))
        if aspect >= 1.0:
            return max_size, max_size / aspect
        return max_size * aspect, max_size

    @staticmethod
    def _tooltip_rect(
        text,
        label: str,
        mouse_pos: tuple[float, float],
        *,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> tuple[float, float, float, float]:
        try:
            label_w, label_h = text.font.size(label)
        except Exception:
            label_w, label_h = len(label) * 9, 18
        tooltip_w = max(72.0, float(label_w) + 24.0)
        tooltip_h = max(34.0, float(label_h) + 16.0)
        mx, my = mouse_pos
        x = float(mx) + 14.0
        y = float(my) + 18.0
        if x + tooltip_w > float(width) - 8.0:
            x = float(mx) - tooltip_w - 14.0
        if y + tooltip_h > float(height) - 8.0:
            y = float(my) - tooltip_h - 14.0
        return max(8.0, x), max(8.0, y), tooltip_w, tooltip_h

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

    @staticmethod
    def _item_description(item) -> str:
        if isinstance(item, dict):
            value = item.get("description", item.get("details", ""))
        else:
            value = getattr(item, "description", "")
        return str(value or "")

    @staticmethod
    def _item_attributes(item) -> tuple[tuple[str, str], ...]:
        if isinstance(item, dict):
            values = item.get("attributes", item.get("stats", ()))
        else:
            values = getattr(item, "attributes", ())
        if isinstance(values, dict):
            values = values.items()
        try:
            return tuple((str(label), str(value)) for label, value in values or ())
        except (TypeError, ValueError):
            return ()

    @staticmethod
    def _wrap_text(text, value: str, max_width: float, max_lines: int) -> list[str]:
        words = str(value or "").split()
        if not words or max_lines <= 0:
            return []
        lines: list[str] = []
        current = words.pop(0)
        while words:
            candidate = f"{current} {words[0]}"
            try:
                fits = text.font.size(candidate)[0] <= max_width
            except Exception:
                fits = len(candidate) <= max(8, int(max_width // 9))
            if fits:
                current = candidate
                words.pop(0)
                continue
            lines.append(current)
            if len(lines) == max_lines:
                break
            current = words.pop(0)
        else:
            lines.append(current)
        if words and lines:
            lines[-1] = lines[-1].rstrip(".") + "..."
        return lines[:max_lines]

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
        card_draw = max(0, int(getattr(stats, "card_draw", 3)))

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
            mx, my = pygame.mouse.get_pos()
            battle_overlay = getattr(self.scene, "battle_overlay", None)
            if getattr(battle_overlay, "loadout_view_open", False):
                battle_overlay.draw_deck_view(text, (mx, my))
                return

            layout = WorldUIInteractions.inventory_layout()
            outer_x, outer_y, outer_w, _outer_h = layout.outer_rect
            scale = outer_w / 960.0

            def rect(x, y, w, h):
                return (outer_x + x * scale, outer_y + y * scale,
                        w * scale, h * scale)

            def label(value, x, y, *, color=theme.INK_TEXT, align="topleft",
                      width=None, height=18.0):
                text.draw_text(
                    value, outer_x + x * scale, outer_y + y * scale,
                    color=(*color, 255), align=align,
                    max_width=None if width is None else width * scale,
                    max_height=height * scale,
                )

            items = list(getattr(self.scene, "inventory_items", ()) or ())
            selected_slot = getattr(self.scene, "inventory_selected_slot", None)
            drag_source = getattr(self.scene, "inventory_drag_source", None)
            hovered_slot = WorldUIInteractions.inventory_slot_at((mx, my))
            dragged_item = (
                items[drag_source] if isinstance(drag_source, int)
                and 0 <= drag_source < len(items) else None
            )
            selected_item = (
                items[selected_slot] if isinstance(selected_slot, int)
                and 0 <= selected_slot < len(items) else None
            )
            battle_cards = getattr(self.scene, "battle_cards", None)
            loadout_cards = getattr(battle_cards, "loadout_cards", ())
            if callable(loadout_cards):
                loadout_cards = loadout_cards()
            deck_count = len(loadout_cards or ())
            occupied = sum(item is not None for item in items[:BACKPACK_SLOT_COUNT])
            rows = self._player_stat_rows()

            def contains(bounds):
                x, y, w, h = bounds
                return x <= mx <= x + w and y <= my <= y + h

            glDisable(GL_TEXTURE_2D)
            theme.draw_rect((0, 0, WIDTH, HEIGHT), (0.04, 0.08, 0.05, 0.30))
            theme.draw_frame(layout.outer_rect, inset=4.0 * scale)
            theme.draw_rect(rect(12, 12, 936, 52), theme.FOREST)
            theme.draw_rect(rect(12, 64, 936, 3), theme.GOLD)
            theme.draw_pixel_icon("bag", rect(28, 25, 26, 26), color=theme.PAPER_LIGHT)
            theme.draw_inset(rect(28, 76, 600, 504), fill=theme.PAPER)
            theme.draw_inset(layout.stats_rect, fill=theme.PAPER_LIGHT)
            theme.draw_rect(rect(48, 202, 560, 2), theme.PAPER_DARK)
            theme.draw_rect(rect(670, 314, 244, 2), theme.PAPER_DARK)
            theme.draw_button(layout.close_rect, hovered=contains(layout.close_rect))
            theme.draw_pixel_icon("close", rect(908, 30, 14, 14), color=theme.PAPER_LIGHT)
            theme.draw_button(
                layout.deck_rect, hovered=contains(layout.deck_rect),
                pressed=bool(getattr(getattr(self.scene, "ui_interactions", None),
                                     "inventory_deck_pressed", False)),
            )
            theme.draw_pixel_icon("cards", rect(744, 27, 20, 20), color=theme.PAPER_LIGHT)

            for index, bounds in enumerate(layout.slot_rects):
                x, y, w, h = bounds
                filled = index < len(items) and items[index] is not None
                selected = index == selected_slot
                hovered = index == hovered_slot
                drop_target = dragged_item is not None and hovered and drag_source != index
                drop_valid = bool(
                    drop_target and slot_accepts_item(index, dragged_item)
                    and slot_accepts_item(drag_source, items[index] if filled else None)
                )
                border = (theme.FOREST_LIGHT if drop_valid else theme.RED) if drop_target else (
                    theme.GOLD if selected or hovered else theme.PAPER_DARK
                )
                theme.draw_rect(bounds, border)
                inset = (4.0 if selected or drop_target else 2.0) * scale
                theme.draw_inset((x + inset, y + inset, w - inset * 2, h - inset * 2),
                                 fill=theme.PAPER_LIGHT if filled or hovered else theme.PAPER)
                # A tiny corner mark distinguishes an occupied slot at a glance.
                if filled:
                    theme.draw_rect((x + 6 * scale, y + 6 * scale, 4 * scale, 4 * scale),
                                    theme.FOREST)

            # Health and mana echo the warm/cool colors of the compass needle.
            for index, (_name, value) in enumerate(rows[:2]):
                x = 670 + index * 128
                theme.draw_inset(rect(x, 112, 116, 47), fill=theme.PAPER)
                current, maximum = (int(part) for part in value.split("/"))
                theme.draw_rect(rect(x + 7, 147, 102, 5), theme.PAPER_DARK)
                theme.draw_rect(rect(x + 7, 147, 102 * min(1, current / max(1, maximum)), 5),
                                theme.RED if index == 0 else theme.BLUE)

            resources = getattr(self.scene, "render_resources", self.scene)
            equipment_textures = getattr(resources, "equipment_slot_textures", {}) or {}
            item_textures = getattr(resources, "item_textures", {}) or {}
            glEnable(GL_TEXTURE_2D)
            for offset, item_kind in enumerate(EQUIPMENT_TYPES):
                index = BACKPACK_SLOT_COUNT + offset
                # Empty-slot silhouettes stay legible on the parchment backing.
                if index < len(items) and items[index] is not None:
                    continue
                x, y, w, h = layout.equipment_slot_rects[offset]
                inset = 12.0 * scale
                self._draw_textured_rect(equipment_textures.get(item_kind.value),
                                         x + inset, y + inset, w - inset * 2, h - inset * 2,
                                         alpha=0.48)

            for index, item in enumerate(items[:len(layout.slot_rects)]):
                if item is None or index == drag_source:
                    continue
                x, y, w, h = layout.slot_rects[index]
                texture = item_textures.get(self._item_icon(item))
                if texture:
                    icon_w, icon_h = self._icon_dimensions(texture, min(w, h) - 22 * scale)
                    self._draw_textured_rect(texture, x + (w - icon_w) / 2,
                                             y + (h - icon_h) / 2, icon_w, icon_h)
                else:
                    text.draw_text(self._item_label(item), x + w / 2, y + h / 2,
                                   color=(*theme.INK_TEXT, 255), align="center",
                                   max_width=w - 12 * scale, max_height=16 * scale)

            label("Inventory", 66, 27, color=theme.CREAM_TEXT, align="topleft", height=22)
            label("View Deck (%d)" % deck_count, 817, 37,
                  color=theme.CREAM_TEXT, align="center", width=108, height=16)
            label("EQUIPMENT", 48, 85, color=theme.MUTED_TEXT, height=13)
            label("BACKPACK", 48, 211, color=theme.MUTED_TEXT, height=13)
            label(f"{occupied} / {BACKPACK_SLOT_COUNT}", 608, 211,
                  color=theme.MUTED_TEXT, align="topright", height=13)
            equipment_names = ("Body", "Boots", "Weapon", "Helmet")
            for offset, name in enumerate(equipment_names):
                label(name, 202 + offset * 84, 186, color=theme.MUTED_TEXT,
                      align="center", height=14)

            label("CHARACTER", 670, 87, color=theme.MUTED_TEXT, height=13)
            for index, (name, value) in enumerate(rows[:2]):
                x = 678 + index * 128
                label(name, x, 122, color=theme.MUTED_TEXT, height=15)
                label(value, x + 99, 122, align="topright", height=15)
            for index, (name, value) in enumerate(rows[2:]):
                y = 175 + index * 27
                label(name, 672, y, color=theme.MUTED_TEXT, width=195, height=16)
                label(value, 912, y, align="topright", width=42, height=16)
            label("ITEM DETAILS", 670, 329, color=theme.MUTED_TEXT, height=13)

            if selected_item is None:
                glDisable(GL_TEXTURE_2D)
                theme.draw_pixel_icon("bag", rect(773, 385, 40, 40), color=theme.PAPER_DARK)
                glEnable(GL_TEXTURE_2D)
                label("Select an item", 792, 452, align="center", height=17)
                label("to view its details", 792, 477, color=theme.MUTED_TEXT,
                      align="center", height=15)
            else:
                label(self._item_label(selected_item), 670, 357, width=244, height=18)
                label(item_type(selected_item).value.title(), 670, 384,
                      color=theme.MUTED_TEXT, height=14)
                description = self._item_description(selected_item)
                # Wrap and fit the whole block, so a long item record cannot
                # collide with the attributes below it on smaller displays.
                if description:
                    text.draw_text_multiline(
                        description, outer_x + 670 * scale, outer_y + 410 * scale,
                        color=(*theme.INK_TEXT, 255), max_width=244 * scale,
                        max_height=57 * scale, line_spacing=1.15, wrap=True,
                    )
                for index, (name, value) in enumerate(self._item_attributes(selected_item)[:2]):
                    y = 488 + index * 39
                    label(name, 670, y, color=theme.MUTED_TEXT, width=244, height=12)
                    label(value, 670, y + 16, width=244, height=15)

            label("Drag items to move or equip", 48, 596, color=theme.MUTED_TEXT,
                  align="topleft", width=480, height=13)
            label("I / Tab / Esc   Close", 932, 596, color=theme.MUTED_TEXT,
                  align="topright", height=13)

            if dragged_item is not None:
                drag_texture = item_textures.get(self._item_icon(dragged_item))
                if drag_texture:
                    icon_w, icon_h = self._icon_dimensions(drag_texture, 48 * scale)
                    self._draw_textured_rect(drag_texture, mx - icon_w / 2,
                                             my - icon_h / 2, icon_w, icon_h)
                else:
                    text.draw_text(self._item_label(dragged_item), mx + 14, my + 14,
                                   color=(*theme.CREAM_TEXT, 255), max_width=180,
                                   max_height=18)

            hovered_item = (
                items[hovered_slot] if isinstance(hovered_slot, int)
                and 0 <= hovered_slot < len(items) else None
            )
            if hovered_item is not None and drag_source is None:
                tooltip_label = self._fit_text_width(
                    text, self._item_label(hovered_item), WIDTH - 48,
                )
                tooltip_rect = self._tooltip_rect(text, tooltip_label, (mx, my))
                glDisable(GL_TEXTURE_2D)
                theme.draw_frame(tooltip_rect, fill=theme.PAPER_LIGHT, inset=2)
                glEnable(GL_TEXTURE_2D)
                x, y, w, h = tooltip_rect
                text.draw_text(tooltip_label, x + w / 2, y + h / 2,
                               color=(*theme.INK_TEXT, 255), align="center")
        finally:
            text.end()
