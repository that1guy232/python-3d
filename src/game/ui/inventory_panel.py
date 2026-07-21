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

            layout = WorldUIInteractions.inventory_layout()
            outer_x, outer_y, outer_w, outer_h = layout.outer_rect
            padding = 24.0
            grid_x, grid_y, grid_w, grid_h = layout.grid_rect
            stats_x, _stats_panel_y, stats_w, _stats_panel_h = layout.stats_rect
            stats_y = outer_y + 82.0
            equipment_slot_rects = layout.equipment_slot_rects
            equipment_y = equipment_slot_rects[0][1]

            rows = self._player_stat_rows()
            items = list(getattr(self.scene, "inventory_items", ()) or ())
            slot_rects = layout.slot_rects
            slot_count = len(slot_rects)
            close_x, close_y, close_w, close_h = layout.close_rect
            selected_slot = getattr(self.scene, "inventory_selected_slot", None)
            drag_source = getattr(self.scene, "inventory_drag_source", None)
            mx, my = pygame.mouse.get_pos()
            hovered_slot = WorldUIInteractions.inventory_slot_at((mx, my))
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
                equipment_y - 14.0,
                grid_w + 24.0,
                grid_y + grid_h - equipment_y + 28.0,
                (0.025, 0.026, 0.03, 0.72),
            )
            self._draw_overlay_rect(
                stats_x,
                outer_y + 62.0,
                stats_w,
                outer_h - 86.0,
                (0.025, 0.026, 0.03, 0.72),
            )
            details_y = outer_y + 350.0
            self._draw_overlay_rect(
                stats_x + 14.0,
                details_y - 13.0,
                stats_w - 28.0,
                2.0,
                (0.28, 0.24, 0.19, 0.8),
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

            for index, (x, y, slot_w, slot_h) in enumerate(slot_rects):
                filled = index < len(items) and items[index] is not None
                selected = index == selected_slot
                drop_target = (
                    drag_source is not None
                    and hovered_slot == index
                    and drag_source != index
                    and 0 <= drag_source < len(items)
                    and items[drag_source] is not None
                )
                drop_valid = bool(
                    drop_target
                    and slot_accepts_item(index, items[drag_source])
                    and slot_accepts_item(
                        drag_source,
                        items[index] if index < len(items) else None,
                    )
                )
                self._draw_overlay_rect(
                    x,
                    y,
                    slot_w,
                    slot_h,
                    (
                        ((0.13, 0.42, 0.20, 1.0) if drop_valid else (0.48, 0.13, 0.10, 1.0))
                        if drop_target
                        else (
                            (0.55, 0.39, 0.12, 1.0)
                            if selected
                            else (
                                (0.12, 0.105, 0.09, 0.96)
                                if filled
                                else (0.06, 0.058, 0.055, 0.92)
                            )
                        )
                    ),
                )
                self._draw_overlay_rect(
                    x + 3.0,
                    y + 3.0,
                    slot_w - 6.0,
                    slot_h - 6.0,
                    ((0.23, 0.18, 0.12, 0.54) if filled else (0.11, 0.105, 0.1, 0.5)),
                )

            glEnable(GL_TEXTURE_2D)
            equipment_textures = getattr(
                getattr(self.scene, "render_resources", self.scene),
                "equipment_slot_textures",
                {},
            ) or {}
            item_textures = getattr(
                getattr(self.scene, "render_resources", self.scene),
                "item_textures",
                {},
            ) or {}
            for offset, item_kind in enumerate(EQUIPMENT_TYPES):
                index = BACKPACK_SLOT_COUNT + offset
                x, y, slot_w, slot_h = equipment_slot_rects[offset]
                filled = index < len(items) and items[index] is not None
                inset = max(6.0, slot_w * 0.13)
                self._draw_textured_rect(
                    equipment_textures.get(item_kind.value),
                    x + inset,
                    y + inset,
                    slot_w - inset * 2.0,
                    slot_h - inset * 2.0,
                    alpha=0.28 if filled else 0.62,
                )

            for index, item in enumerate(items[:slot_count]):
                texture = item_textures.get(self._item_icon(item))
                if not texture or index == drag_source:
                    continue
                x, y, slot_w, slot_h = slot_rects[index]
                max_icon_size = min(42.0, slot_w - 12.0, slot_h - 12.0)
                icon_w, icon_h = self._icon_dimensions(texture, max_icon_size)
                self._draw_textured_rect(
                    texture,
                    x + (slot_w - icon_w) * 0.5,
                    y + (slot_h - icon_h) * 0.5,
                    icon_w,
                    icon_h,
                )

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
                "Item Details",
                stats_x + 16.0,
                details_y,
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

            for offset, item_kind in enumerate(EQUIPMENT_TYPES):
                x, y, slot_w, slot_h = equipment_slot_rects[offset]
                text.draw_text(
                    item_kind.value.title(),
                    x + slot_w * 0.5,
                    y + slot_h + 13.0,
                    color=(185, 180, 170, 255),
                    align="center",
                )

            for index, item in enumerate(items[:slot_count]):
                label = self._item_label(item)
                if (
                    not label
                    or index == drag_source
                    or item_textures.get(self._item_icon(item))
                ):
                    continue
                x, y, slot_w, slot_h = slot_rects[index]
                label = self._fit_text_width(text, label, slot_w - 10.0)
                if not label:
                    continue
                text.draw_text(
                    label,
                    x + slot_w * 0.5,
                    y + slot_h * 0.5,
                    color=(245, 235, 215, 255),
                    align="center",
                )

            selected_item = (
                items[selected_slot]
                if isinstance(selected_slot, int)
                and 0 <= selected_slot < len(items)
                else None
            )
            detail_x = stats_x + 18.0
            detail_w = stats_w - 36.0
            if selected_item is None:
                for line_index, line in enumerate(
                    self._wrap_text(
                        text,
                        "Click an item to view its details.",
                        detail_w,
                        2,
                    )
                ):
                    text.draw_text(
                        line,
                        detail_x,
                        details_y + 38.0 + line_index * 22.0,
                        color=(175, 176, 180, 255),
                        align="topleft",
                    )
            else:
                name = self._fit_text_width(
                    text,
                    self._item_label(selected_item),
                    detail_w,
                )
                text.draw_text(
                    name,
                    detail_x,
                    details_y + 34.0,
                    color=(245, 221, 164, 255),
                    align="topleft",
                )
                text.draw_text(
                    f"Type: {item_type(selected_item).value.title()}",
                    detail_x,
                    details_y + 60.0,
                    color=(205, 207, 212, 255),
                    align="topleft",
                )
                detail_line = details_y + 87.0
                description = self._item_description(selected_item)
                if description:
                    for line in self._wrap_text(text, description, detail_w, 2):
                        text.draw_text(
                            line,
                            detail_x,
                            detail_line,
                            color=(180, 181, 185, 255),
                            align="topleft",
                        )
                        detail_line += 22.0
                for label, value in self._item_attributes(selected_item)[:2]:
                    if detail_line > outer_y + outer_h - 28.0:
                        break
                    text.draw_text(
                        label,
                        detail_x,
                        detail_line,
                        color=(195, 197, 202, 255),
                        align="topleft",
                    )
                    text.draw_text(
                        value,
                        stats_x + stats_w - 18.0,
                        detail_line,
                        color=(245, 221, 164, 255),
                        align="topright",
                    )
                    detail_line += 22.0

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

            if (
                isinstance(drag_source, int)
                and 0 <= drag_source < len(items)
                and items[drag_source] is not None
            ):
                dragged_item = items[drag_source]
                drag_texture = item_textures.get(self._item_icon(dragged_item))
                if drag_texture:
                    icon_w, icon_h = self._icon_dimensions(drag_texture, 42.0)
                    self._draw_textured_rect(
                        drag_texture,
                        mx - icon_w * 0.5,
                        my - icon_h * 0.5,
                        icon_w,
                        icon_h,
                    )
                else:
                    drag_label = self._fit_text_width(
                        text,
                        self._item_label(dragged_item),
                        180.0,
                    )
                    text.draw_text(
                        drag_label,
                        mx + 14.0,
                        my + 14.0,
                        color=(255, 238, 195, 255),
                        align="topleft",
                    )

            hovered_item = (
                items[hovered_slot]
                if isinstance(hovered_slot, int)
                and 0 <= hovered_slot < len(items)
                else None
            )
            if hovered_item is not None and drag_source is None:
                tooltip_label = self._item_label(hovered_item)
                tooltip_x, tooltip_y, tooltip_w, tooltip_h = self._tooltip_rect(
                    text,
                    tooltip_label,
                    (mx, my),
                )
                glDisable(GL_TEXTURE_2D)
                self._draw_overlay_rect(
                    tooltip_x,
                    tooltip_y,
                    tooltip_w,
                    tooltip_h,
                    (0.03, 0.028, 0.025, 0.98),
                )
                self._draw_overlay_rect(
                    tooltip_x + 2.0,
                    tooltip_y + 2.0,
                    tooltip_w - 4.0,
                    tooltip_h - 4.0,
                    (0.19, 0.15, 0.10, 0.98),
                )
                glEnable(GL_TEXTURE_2D)
                text.draw_text(
                    tooltip_label,
                    tooltip_x + tooltip_w * 0.5,
                    tooltip_y + tooltip_h * 0.5,
                    color=(255, 238, 202, 255),
                    align="center",
                )
        finally:
            text.end()
