"""Battle-mode overlay options and actions."""

from __future__ import annotations

from engine.ui.menu import ButtonMenu, MenuOption


class BattleMenu(ButtonMenu):
    title = "Battle mode"
    button_width_max = 300

    def __init__(self, scene) -> None:
        super().__init__(scene)
        self.buttons = [
            MenuOption("end_fight", "End fight", self.end_fight),
        ]

    def options(self) -> list[MenuOption]:
        return self.buttons

    def end_fight(self, scene) -> None:
        end_battle = getattr(scene, "end_battle", None)
        if callable(end_battle):
            end_battle()
