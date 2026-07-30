"""Battle-mode screen-space resource overlay."""

from __future__ import annotations

import math
import time

from OpenGL.GL import (
    glBegin,
    glColor4f,
    glDisable,
    glEnable,
    glEnd,
    glVertex2f,
    GL_QUADS,
    GL_TEXTURE_2D,
    GL_TRIANGLE_FAN,
)

from game.config import HEIGHT, WIDTH


class BattleResourceOverlay:
    """Draw player battle resources in the 2D overlay pass."""

    enter_duration_s = 0.48

    def __init__(self, scene) -> None:
        self.scene = scene
        self._active = False
        self._enter_s = 0.0
        self._target_id = None
        self._end_turn_pressed = False
        self._deck_button_pressed = False
        self._deck_button_press_rect = None
        self._deck_view_open = False
        self._deck_view_kind: str | None = None
        self._deck_view_block_release = False

    @property
    def deck_view_open(self) -> bool:
        return self._deck_view_open

    @property
    def deck_view_kind(self) -> str | None:
        return self._deck_view_kind if self._deck_view_open else None

    @property
    def loadout_view_open(self) -> bool:
        return self.deck_view_kind == "loadout"

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _lerp(start: float, end: float, amount: float) -> float:
        return float(start) + (float(end) - float(start)) * amount

    def sync_state(self) -> None:
        active = bool(getattr(self.scene, "battle_mode", False))
        target = getattr(self.scene, "active_battle_creature", None)
        target_id = id(target) if target is not None else None
        if active:
            if not self._active or self._target_id != target_id:
                self._enter_s = time.perf_counter()
                self._deck_view_open = False
                self._deck_view_kind = None
                self._deck_button_pressed = False
                self._deck_button_press_rect = None
                self._deck_view_block_release = False
                self._reset_cards()
            self._active = True
            self._target_id = target_id
            return

        preserve_inventory_view = bool(
            getattr(self.scene, "inventory_open", False)
            and (self.loadout_view_open or self._deck_view_block_release)
        )
        self._active = False
        self._enter_s = 0.0
        self._target_id = None
        self._end_turn_pressed = False
        self._deck_button_pressed = False
        self._deck_button_press_rect = None
        if not preserve_inventory_view:
            self._deck_view_open = False
            self._deck_view_kind = None
            self._deck_view_block_release = False
        self._reset_cards()

    def _reset_cards(self) -> None:
        battle_cards = getattr(self.scene, "battle_cards", None)
        reset = getattr(battle_cards, "reset", None)
        if callable(reset):
            reset()
            return
        for card in self._cards():
            card.reset_to_home()

    def _cards(self) -> list:
        battle_cards = getattr(self.scene, "battle_cards", None)
        cards = getattr(battle_cards, "cards", None)
        if callable(cards):
            cards = cards()
        return list(cards or ())

    def _deck_cards(self) -> list:
        battle_cards = getattr(self.scene, "battle_cards", None)
        cards = getattr(battle_cards, "deck_cards", ())
        if callable(cards):
            cards = cards()
        return list(cards or ())

    def _loadout_cards(self) -> list:
        battle_cards = getattr(self.scene, "battle_cards", None)
        cards = getattr(battle_cards, "loadout_cards", ())
        if callable(cards):
            cards = cards()
        return list(cards or ())

    def close_deck_view(self, *, preserve_release: bool = False) -> bool:
        """Close the active deck viewer and report whether it was open."""

        was_open = self._deck_view_open
        self._deck_view_open = False
        self._deck_view_kind = None
        self._deck_button_pressed = False
        self._deck_button_press_rect = None
        if not preserve_release:
            self._deck_view_block_release = False
        return was_open

    def _open_deck_view(self, kind: str = "draw_pile") -> None:
        self._deck_view_open = True
        self._deck_view_kind = kind
        self._deck_button_pressed = False
        self._deck_button_press_rect = None
        self._end_turn_pressed = False
        self._deck_view_block_release = False
        self._reset_cards()

    def open_loadout_view(self) -> bool:
        """Open the full equipped-loadout viewer outside battle."""

        if getattr(self.scene, "battle_mode", False):
            return False
        self._open_deck_view("loadout")
        return True

    def _entry_progress(self) -> float:
        if not self._active:
            return 0.0
        elapsed = max(0.0, time.perf_counter() - self._enter_s)
        progress = self._clamp01(elapsed / self.enter_duration_s)
        return progress * progress * (3.0 - 2.0 * progress)

    @staticmethod
    def _draw_circle(
        x: float,
        y: float,
        radius: float,
        color: tuple[float, float, float, float],
        *,
        segments: int = 72,
    ) -> None:
        glColor4f(*color)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        for index in range(segments + 1):
            angle = (math.tau * index) / segments
            glVertex2f(x + math.cos(angle) * radius, y + math.sin(angle) * radius)
        glEnd()

    @staticmethod
    def _draw_quad(
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

    def _resource_layout(self) -> dict[str, float]:
        radius = min(76.0, max(48.0, min(float(WIDTH) * 0.045, float(HEIGHT) * 0.08)))
        edge_margin = max(32.0, radius * 0.72)
        left_x = edge_margin + radius
        right_x = float(WIDTH) - left_x
        final_y = float(HEIGHT) - radius - max(34.0, float(HEIGHT) * 0.05)
        start_y = float(HEIGHT) + radius + 20.0
        y = self._lerp(start_y, final_y, self._entry_progress())
        return {"radius": radius, "left_x": left_x, "right_x": right_x, "y": y}

    @staticmethod
    def _card_size() -> tuple[float, float]:
        card_w = min(118.0, max(92.0, float(WIDTH) * 0.082))
        return card_w, card_w * 1.38

    def _sync_card_layout(self, layout: dict[str, float], cards: list) -> None:
        card_w, card_h = self._card_size()
        card_gap = max(18.0, card_w * 0.18)
        total_w = len(cards) * card_w + max(0, len(cards) - 1) * card_gap
        first_x = float(WIDTH) * 0.5 - total_w * 0.5 + card_w * 0.5
        for index, card in enumerate(cards):
            card.set_home_center(
                first_x + index * (card_w + card_gap),
                layout["y"],
                size=(card_w, card_h),
            )

    @classmethod
    def _deck_button_rect(
        cls,
        layout: dict[str, float],
    ) -> tuple[float, float, float, float]:
        _card_w, card_h = cls._card_size()
        center_x = float(WIDTH) * 0.5 - 74.0
        center_y = layout["y"] - card_h * 0.5 - 18.0
        return center_x - 62.0, center_y - 17.0, 124.0, 34.0

    @staticmethod
    def _deck_view_rect() -> tuple[float, float, float, float]:
        panel_w = min(max(320.0, float(WIDTH) - 96.0), 1120.0)
        panel_h = min(max(300.0, float(HEIGHT) - 96.0), 520.0)
        return (
            (float(WIDTH) - panel_w) * 0.5,
            (float(HEIGHT) - panel_h) * 0.5,
            panel_w,
            panel_h,
        )

    @classmethod
    def _deck_view_close_rect(cls) -> tuple[float, float, float, float]:
        panel_x, panel_y, panel_w, _panel_h = cls._deck_view_rect()
        return panel_x + panel_w - 52.0, panel_y + 16.0, 36.0, 36.0

    @classmethod
    def _deck_view_card_rects(
        cls,
        card_count: int,
    ) -> tuple[tuple[float, float, float, float], ...]:
        count = max(0, int(card_count))
        if count == 0:
            return ()

        panel_x, panel_y, panel_w, panel_h = cls._deck_view_rect()
        content_x = panel_x + 32.0
        content_y = panel_y + 92.0
        content_w = panel_w - 64.0
        content_h = panel_h - 140.0
        gap = 18.0
        max_columns = max(1, int((content_w + gap) // (92.0 + gap)))
        columns = min(count, max_columns, 6)
        rows = max(1, math.ceil(count / columns))
        card_w = min(
            118.0,
            (content_w - gap * (columns - 1)) / columns,
            (content_h - gap * (rows - 1)) / (rows * 1.38),
        )
        card_w = max(36.0, card_w)
        card_h = card_w * 1.38
        total_h = rows * card_h + (rows - 1) * gap
        first_y = content_y + max(0.0, (content_h - total_h) * 0.5)
        rects: list[tuple[float, float, float, float]] = []
        for row in range(rows):
            row_start = row * columns
            row_count = min(columns, count - row_start)
            row_w = row_count * card_w + max(0, row_count - 1) * gap
            first_x = content_x + (content_w - row_w) * 0.5
            for column in range(row_count):
                rects.append(
                    (
                        first_x + column * (card_w + gap),
                        first_y + row * (card_h + gap),
                        card_w,
                        card_h,
                    )
                )
        return tuple(rects)

    @staticmethod
    def _play_rect() -> tuple[float, float, float, float]:
        zone_w = min(300.0, max(210.0, float(WIDTH) * 0.26))
        zone_h = min(210.0, max(150.0, float(HEIGHT) * 0.24))
        return (
            float(WIDTH) * 0.5 - zone_w * 0.5,
            float(HEIGHT) * 0.5 - zone_h * 0.5,
            zone_w,
            zone_h,
        )

    @staticmethod
    def _end_turn_rect() -> tuple[float, float, float, float]:
        return float(WIDTH) - 166.0, 20.0, 142.0, 46.0

    @staticmethod
    def _contains(rect: tuple[float, float, float, float], pos) -> bool:
        x, y, w, h = rect
        px, py = pos
        return x <= px <= x + w and y <= py <= y + h

    def _draw_play_zone(self) -> None:  # pragma: no cover - visual
        x, y, w, h = self._play_rect()
        glDisable(GL_TEXTURE_2D)
        self._draw_quad(x, y, w, h, (0.02, 0.025, 0.03, 0.34))
        self._draw_quad(x + 4.0, y + 4.0, w - 8.0, h - 8.0, (0.58, 0.17, 0.10, 0.18))
        line_w = min(80.0, w * 0.34)
        line_h = 5.0
        cx = x + w * 0.5
        cy = y + h * 0.5
        self._draw_quad(
            cx - line_w * 0.5,
            cy - line_h * 0.5,
            line_w,
            line_h,
            (0.9, 0.78, 0.56, 0.45),
        )
        self._draw_quad(
            cx - line_h * 0.5,
            cy - line_w * 0.5,
            line_h,
            line_w,
            (0.9, 0.78, 0.56, 0.45),
        )
        glEnable(GL_TEXTURE_2D)

    def _draw_deck_view(self, text, mouse_pos) -> None:  # pragma: no cover - visual
        viewing_loadout = self._deck_view_kind == "loadout"
        cards = self._loadout_cards() if viewing_loadout else self._deck_cards()
        cards.sort(key=lambda card: (str(card.title), str(card.action)))
        panel_x, panel_y, panel_w, panel_h = self._deck_view_rect()
        close_x, close_y, close_w, close_h = self._deck_view_close_rect()
        close_hovered = self._contains(
            (close_x, close_y, close_w, close_h),
            mouse_pos,
        )

        glDisable(GL_TEXTURE_2D)
        self._draw_quad(
            0.0,
            0.0,
            float(WIDTH),
            float(HEIGHT),
            (0.0, 0.0, 0.0, 0.68),
        )
        self._draw_quad(
            panel_x + 8.0,
            panel_y + 10.0,
            panel_w,
            panel_h,
            (0.0, 0.0, 0.0, 0.36),
        )
        self._draw_quad(
            panel_x,
            panel_y,
            panel_w,
            panel_h,
            (0.045, 0.038, 0.034, 0.98),
        )
        self._draw_quad(
            panel_x + 4.0,
            panel_y + 4.0,
            panel_w - 8.0,
            panel_h - 8.0,
            (0.12, 0.09, 0.07, 0.98),
        )
        self._draw_quad(
            close_x,
            close_y,
            close_w,
            close_h,
            (0.64, 0.28, 0.08, 0.98)
            if close_hovered
            else (0.34, 0.16, 0.06, 0.96),
        )
        glEnable(GL_TEXTURE_2D)

        count = len(cards)
        text.draw_text(
            "Your Deck" if viewing_loadout else "Draw Pile",
            panel_x + panel_w * 0.5,
            panel_y + 27.0,
            color=(255, 242, 220, 255),
            align="center",
            max_width=panel_w - 140.0,
            max_height=28.0,
        )
        text.draw_text(
            (
                f"{count} card{'s' if count != 1 else ''} in current loadout"
                if viewing_loadout
                else f"{count} card{'s' if count != 1 else ''} remaining"
            ),
            panel_x + panel_w * 0.5,
            panel_y + 58.0,
            color=(218, 205, 188, 255),
            align="center",
            max_width=panel_w - 100.0,
            max_height=20.0,
        )
        text.draw_text(
            "X",
            close_x + close_w * 0.5,
            close_y + close_h * 0.5,
            color=(255, 244, 224, 255),
            align="center",
            max_width=18.0,
            max_height=18.0,
        )

        if cards:
            for card, rect in zip(cards, self._deck_view_card_rects(count)):
                card.draw_at(text, rect, enabled=True, raised=False)
        else:
            text.draw_text(
                (
                    "No cards are in the current loadout."
                    if viewing_loadout
                    else "No cards remain in the draw pile."
                ),
                panel_x + panel_w * 0.5,
                panel_y + panel_h * 0.5,
                color=(225, 214, 198, 255),
                align="center",
                max_width=panel_w - 80.0,
                max_height=28.0,
            )

        text.draw_text(
            "Click outside or press Escape to close",
            panel_x + panel_w * 0.5,
            panel_y + panel_h - 24.0,
            color=(188, 178, 165, 255),
            align="center",
            max_width=panel_w - 80.0,
            max_height=18.0,
        )

    def draw_deck_view(self, text, mouse_pos) -> bool:  # pragma: no cover - visual
        """Draw the shared modal last when either deck-view mode is active."""

        if not self._deck_view_open:
            return False
        self._draw_deck_view(text, mouse_pos)
        return True

    def draw(self, text) -> None:  # pragma: no cover - visual
        if not getattr(self.scene, "battle_mode", False):
            return

        self.sync_state()
        stats = getattr(self.scene, "player_stats", None)
        if stats is None:
            return

        hp = max(0, int(getattr(stats, "hp", 5)))
        max_hp = max(1, int(getattr(stats, "max_hp", max(1, hp))))
        mana = max(0, int(getattr(stats, "mana", 5)))
        max_mana = max(1, int(getattr(stats, "max_mana", max(1, mana))))

        layout = self._resource_layout()
        cards = self._cards()
        self._sync_card_layout(layout, cards)
        radius = layout["radius"]
        left_x = layout["left_x"]
        right_x = layout["right_x"]
        y = layout["y"]

        circles = (
            ("HP", f"{hp}/{max_hp}", left_x, y, (0.86, 0.08, 0.06, 0.96)),
            ("Mana", f"{mana}/{max_mana}", right_x, y, (0.10, 0.36, 0.98, 0.96)),
        )

        try:
            import pygame

            mouse_pos = pygame.mouse.get_pos()
        except Exception:
            mouse_pos = getattr(self.scene, "_last_mouse_pos", (0, 0))

        end_turn_rect = self._end_turn_rect()
        end_turn_hovered = (
            not self._deck_view_open
            and self._contains(end_turn_rect, mouse_pos)
        )
        end_x, end_y, end_w, end_h = end_turn_rect
        deck_button_rect = self._deck_button_rect(layout)
        deck_x, deck_y, deck_w, deck_h = deck_button_rect
        deck_button_hovered = (
            not self._deck_view_open
            and self._contains(deck_button_rect, mouse_pos)
        )
        battle_cards = getattr(self.scene, "battle_cards", None)
        deck_count = max(0, int(getattr(battle_cards, "deck_count", 0)))
        discard_count = max(0, int(getattr(battle_cards, "discard_count", 0)))

        glDisable(GL_TEXTURE_2D)
        for _label, _value, x, circle_y, color in circles:
            self._draw_circle(
                x + 5.0,
                circle_y + 7.0,
                radius,
                (0.0, 0.0, 0.0, 0.26),
            )
            self._draw_circle(
                x,
                circle_y,
                radius + 7.0,
                (0.025, 0.03, 0.04, 0.82),
            )
            self._draw_circle(x, circle_y, radius, color)
            self._draw_circle(
                x,
                circle_y - radius * 0.16,
                radius * 0.72,
                (1.0, 1.0, 1.0, 0.08),
            )
            self._draw_circle(
                x,
                circle_y,
                radius * 0.58,
                (0.025, 0.028, 0.036, 0.32),
            )

        dragging_card = any(card.dragging for card in cards)
        if dragging_card:
            self._draw_play_zone()
            glDisable(GL_TEXTURE_2D)

        self._draw_quad(
            end_x + 4.0,
            end_y + 5.0,
            end_w,
            end_h,
            (0.0, 0.0, 0.0, 0.28),
        )
        self._draw_quad(end_x, end_y, end_w, end_h, (0.05, 0.04, 0.03, 0.96))
        button_face = (
            (0.66, 0.31, 0.08, 0.98)
            if end_turn_hovered or self._end_turn_pressed
            else (0.42, 0.20, 0.06, 0.94)
        )
        self._draw_quad(
            end_x + 4.0,
            end_y + 4.0,
            end_w - 8.0,
            end_h - 8.0,
            button_face,
        )
        self._draw_quad(
            deck_x + 3.0,
            deck_y + 4.0,
            deck_w,
            deck_h,
            (0.0, 0.0, 0.0, 0.28),
        )
        self._draw_quad(
            deck_x,
            deck_y,
            deck_w,
            deck_h,
            (0.045, 0.04, 0.04, 0.96),
        )
        deck_face = (
            (0.34, 0.24, 0.14, 0.98)
            if deck_button_hovered or self._deck_button_pressed
            else (0.12, 0.10, 0.09, 0.94)
        )
        self._draw_quad(
            deck_x + 3.0,
            deck_y + 3.0,
            deck_w - 6.0,
            deck_h - 6.0,
            deck_face,
        )

        glEnable(GL_TEXTURE_2D)
        for label, value, x, circle_y, _color in circles:
            text.draw_text(
                label,
                x,
                circle_y - 11.0,
                color=(255, 245, 235, 255),
                align="center",
            )
            text.draw_text(
                value,
                x,
                circle_y + 16.0,
                color=(255, 255, 255, 255),
                align="center",
            )

        text.draw_text(
            "End Turn",
            end_x + end_w * 0.5,
            end_y + end_h * 0.5,
            color=(255, 244, 218, 255),
            align="center",
        )
        text.draw_text(
            f"Deck {deck_count}",
            deck_x + deck_w * 0.5,
            deck_y + deck_h * 0.5,
            color=(245, 230, 208, 255),
            align="center",
            max_width=deck_w - 18.0,
            max_height=18.0,
        )
        _card_w, card_h = self._card_size()
        pile_y = y - card_h * 0.5 - 18.0
        text.draw_text(
            f"Discard {discard_count}",
            float(WIDTH) * 0.5 + 74.0,
            pile_y,
            color=(220, 226, 238, 255),
            align="center",
        )

        for card in cards:
            enabled = card.enabled_for(self.scene)
            if self._deck_view_open:
                card.hovered = False
            else:
                card.update_hover(mouse_pos, self.scene)
            card.draw(text, enabled=enabled)

        if self._deck_view_open:
            self.draw_deck_view(text, mouse_pos)

    def _prepare_for_input(self) -> list:
        self.sync_state()
        cards = self._cards()
        self._sync_card_layout(self._resource_layout(), cards)
        return cards

    def handle_deck_view_mouse_down(self, pos) -> bool:
        """Consume a modal mouse-down, closing on its X or backdrop."""

        if not self._deck_view_open:
            self._deck_view_block_release = False
            return False
        self._deck_view_block_release = True
        if self._contains(self._deck_view_close_rect(), pos) or not self._contains(
            self._deck_view_rect(),
            pos,
        ):
            self.close_deck_view(preserve_release=True)
        return True

    def handle_deck_view_mouse_up(self, _pos) -> bool:
        """Consume modal releases, including the release that follows a close."""

        if self._deck_view_block_release:
            self._deck_view_block_release = False
            return True
        return self._deck_view_open

    def handle_mouse_down(self, pos) -> bool:
        if not getattr(self.scene, "battle_mode", False):
            self._deck_button_pressed = False
            self._deck_button_press_rect = None
            self._deck_view_block_release = False
            return False
        if self.handle_deck_view_mouse_down(pos):
            return True
        layout = self._resource_layout()
        deck_button_rect = self._deck_button_rect(layout)
        if self._contains(deck_button_rect, pos):
            self._deck_button_pressed = True
            self._deck_button_press_rect = deck_button_rect
            self._end_turn_pressed = False
            return True
        if self._contains(self._end_turn_rect(), pos):
            self._end_turn_pressed = True
            self._deck_button_pressed = False
            self._deck_button_press_rect = None
            return True
        cards = self._prepare_for_input()
        for card in reversed(cards):
            if card.handle_mouse_down(pos, self.scene):
                return True
        return False

    def handle_mouse_motion(self, pos) -> bool:
        if not getattr(self.scene, "battle_mode", False):
            return False
        if self._deck_view_open:
            return True
        cards = self._prepare_for_input()
        handled = (
            self._end_turn_pressed
            or self._deck_button_pressed
            or self._contains(self._end_turn_rect(), pos)
            or self._contains(self._deck_button_rect(self._resource_layout()), pos)
        )
        for card in cards:
            handled = card.handle_mouse_motion(pos, self.scene) or handled
        return handled

    def handle_mouse_up(self, pos) -> bool:
        if not getattr(self.scene, "battle_mode", False):
            self._end_turn_pressed = False
            self._deck_button_pressed = False
            self._deck_button_press_rect = None
            self._deck_view_block_release = False
            self._deck_view_open = False
            self._deck_view_kind = None
            return False
        if self.handle_deck_view_mouse_up(pos):
            return True
        if self._deck_button_pressed:
            self._deck_button_pressed = False
            current_rect = self._deck_button_rect(self._resource_layout())
            pressed_rect = self._deck_button_press_rect
            self._deck_button_press_rect = None
            if self._contains(current_rect, pos) or (
                pressed_rect is not None and self._contains(pressed_rect, pos)
            ):
                self._open_deck_view()
            return True
        if self._end_turn_pressed:
            self._end_turn_pressed = False
            if self._contains(self._end_turn_rect(), pos):
                end_player_turn = getattr(self.scene, "end_player_turn", None)
                if callable(end_player_turn):
                    end_player_turn()
            return True
        cards = self._prepare_for_input()
        handled = False
        play_rect = self._play_rect()
        for card in cards:
            handled = (
                card.handle_mouse_up(pos, self.scene, play_rect=play_rect) or handled
            )
        return handled
