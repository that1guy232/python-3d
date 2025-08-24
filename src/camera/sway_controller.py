"""Sway controller: encapsulates mouse-look weapon sway state and smoothing.

This replaces the inline _sway/_sway_target logic previously in WorldScene and
exposes a small API used by the HUD and WorldScene input handling.
"""

from __future__ import annotations

import math
from pygame.math import Vector2
from typing import Tuple


class SwayController:
    def __init__(
        self,
        max_x: float = 1.25,
        max_y: float = 0.75,
        mouse_scale: float = 0.01,
        responsiveness: float = 12.0,
        return_rate: float = 8.0,
        right_mult: float = 1.1,
        up_mult: float = 1.1,
        forward_mult: float = 0.05,
    ) -> None:
        # Public sway vector in (right, up) space
        self.sway = Vector2(0.0, 0.0)
        # Internal target that accumulates raw mouse deltas then decays
        self._target = Vector2(0.0, 0.0)

        self.max = Vector2(max_x, max_y)
        self.mouse_scale = float(mouse_scale)
        self.responsiveness = float(responsiveness)
        self.return_rate = float(return_rate)

        # Multipliers used by HUD to convert sway.x/y into world offsets
        self.right_mult = float(right_mult)
        self.up_mult = float(up_mult)
        self.forward_mult = float(forward_mult)

    def on_mouse_delta(self, dx: float, dy: float) -> None:
        """Called with raw mouse delta (dx, dy) to nudge sway target."""
        self._target.x += dx * self.mouse_scale
        self._target.y += dy * self.mouse_scale
        # Clamp target to configured maximums
        self._target.x = max(-self.max.x, min(self.max.x, self._target.x))
        self._target.y = max(-self.max.y, min(self.max.y, self._target.y))

    def update(self, dt: float) -> None:
        """Smoothly update the sway toward the target and decay the target."""
        if dt <= 0.0:
            return
        a = 1.0 - math.exp(-self.responsiveness * dt)
        # Ease current sway toward target
        self.sway += (self._target - self.sway) * a
        # Let target decay toward 0 so it recenters over time
        decay = math.exp(-self.return_rate * dt)
        self._target *= decay

    def get_sway(self) -> Vector2:
        return self.sway.copy()


__all__ = ["SwayController"]
