"""Battle-mode overlay options and actions."""

from __future__ import annotations

from engine.ui.menu import ButtonMenu, MenuOption


class BattleMenu(ButtonMenu):
    title = "Battle mode"
    button_width_max = 300

    def __init__(self, scene) -> None:
        super().__init__(scene)
        self.damage_button = MenuOption(
            "damage_goblin",
            self.attack_label,
            self.damage_goblin,
        )

    def attack_label(self, scene) -> str:
        preview = getattr(scene, "player_attack_damage_preview", None)
        amount = preview() if callable(preview) else 1
        return f"Attack ({max(0, int(amount))})"

    def options(self) -> list[MenuOption]:
        goblin = getattr(self.scene, "active_battle_goblin", None)
        if goblin is None or not getattr(goblin, "enabled", True):
            return []

        max_hp = max(1, int(getattr(goblin, "max_hp", 5)))
        hp = int(getattr(goblin, "hp", max_hp))
        return [self.damage_button] if hp > 0 else []

    def damage_goblin(self, scene) -> None:
        damage_battle_goblin = getattr(scene, "damage_battle_goblin", None)
        if callable(damage_battle_goblin):
            damage_battle_goblin()
