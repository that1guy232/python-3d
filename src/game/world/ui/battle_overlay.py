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

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _lerp(start: float, end: float, amount: float) -> float:
        return float(start) + (float(end) - float(start)) * amount

    def sync_state(self) -> None:
        active = bool(getattr(self.scene, "battle_mode", False))
        target = getattr(self.scene, "active_battle_goblin", None)
        target_id = id(target) if target is not None else None
        if active:
            if not self._active or self._target_id != target_id:
                self._enter_s = time.perf_counter()
                self._reset_cards()
            self._active = True
            self._target_id = target_id
            return

        self._active = False
        self._enter_s = 0.0
        self._target_id = None
        self._end_turn_pressed = False
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

    def _sync_card_layout(self, layout: dict[str, float], cards: list) -> None:
        card_w = min(118.0, max(92.0, float(WIDTH) * 0.082))
        card_h = card_w * 1.38
        card_gap = max(18.0, card_w * 0.18)
        total_w = len(cards) * card_w + max(0, len(cards) - 1) * card_gap
        first_x = float(WIDTH) * 0.5 - total_w * 0.5 + card_w * 0.5
        for index, card in enumerate(cards):
            card.set_home_center(
                first_x + index * (card_w + card_gap),
                layout["y"],
                size=(card_w, card_h),
            )

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
        end_turn_hovered = self._contains(end_turn_rect, mouse_pos)
        end_x, end_y, end_w, end_h = end_turn_rect
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
        card_w = min(118.0, max(92.0, float(WIDTH) * 0.082))
        pile_y = y - card_w * 1.38 * 0.5 - 18.0
        text.draw_text(
            f"Deck {deck_count}",
            float(WIDTH) * 0.5 - 74.0,
            pile_y,
            color=(220, 226, 238, 255),
            align="center",
        )
        text.draw_text(
            f"Discard {discard_count}",
            float(WIDTH) * 0.5 + 74.0,
            pile_y,
            color=(220, 226, 238, 255),
            align="center",
        )

        for card in cards:
            enabled = card.enabled_for(self.scene)
            card.update_hover(mouse_pos, self.scene)
            card.draw(text, enabled=enabled)

    def _prepare_for_input(self) -> list:
        self.sync_state()
        cards = self._cards()
        self._sync_card_layout(self._resource_layout(), cards)
        return cards

    def handle_mouse_down(self, pos) -> bool:
        if not getattr(self.scene, "battle_mode", False):
            return False
        if self._contains(self._end_turn_rect(), pos):
            self._end_turn_pressed = True
            return True
        cards = self._prepare_for_input()
        for card in reversed(cards):
            if card.handle_mouse_down(pos, self.scene):
                return True
        return False

    def handle_mouse_motion(self, pos) -> bool:
        if not getattr(self.scene, "battle_mode", False):
            return False
        cards = self._prepare_for_input()
        handled = self._end_turn_pressed or self._contains(self._end_turn_rect(), pos)
        for card in cards:
            handled = card.handle_mouse_motion(pos, self.scene) or handled
        return handled

    def handle_mouse_up(self, pos) -> bool:
        if not getattr(self.scene, "battle_mode", False):
            self._end_turn_pressed = False
            return False
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
