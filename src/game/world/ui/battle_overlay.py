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
            self._active = True
            self._target_id = target_id
            return

        self._active = False
        self._enter_s = 0.0
        self._target_id = None

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

        radius = min(76.0, max(48.0, min(float(WIDTH) * 0.045, float(HEIGHT) * 0.08)))
        edge_margin = max(32.0, radius * 0.72)
        left_x = edge_margin + radius
        right_x = float(WIDTH) - left_x
        final_y = float(HEIGHT) - radius - max(34.0, float(HEIGHT) * 0.05)
        start_y = float(HEIGHT) + radius + 20.0
        y = self._lerp(start_y, final_y, self._entry_progress())

        circles = (
            ("HP", f"{hp}/{max_hp}", left_x, y, (0.86, 0.08, 0.06, 0.96)),
            ("Mana", f"{mana}/{max_mana}", right_x, y, (0.10, 0.36, 0.98, 0.96)),
        )

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
