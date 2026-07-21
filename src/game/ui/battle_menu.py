"""Battle-mode overlay options and actions."""

from __future__ import annotations

from engine.ui.menu import ButtonMenu, MenuItem


class BattleMenu(ButtonMenu):
    title = "Battle mode"
    button_width_max = 300

    def options(self) -> list[MenuItem]:
        return []
