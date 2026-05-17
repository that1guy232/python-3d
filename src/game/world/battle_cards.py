"""Current battle card loadout."""

from __future__ import annotations

from game.world.ui.card import Card


class BattleCardLoadout:
    """Own active battle cards separate from overlay layout/rendering."""

    def __init__(self, scene) -> None:
        self.scene = scene
        self._cards: list[Card] = [self._build_test_strike()]

    @property
    def cards(self) -> list[Card]:
        return self._cards

    def reset(self) -> None:
        for card in self.cards:
            card.reset_to_home()

    def _build_test_strike(self) -> Card:
        return Card(
            "test_strike",
            "Strike",
            "1 Damage",
            self._play_strike,
        )

    def _play_strike(self, scene) -> None:
        damage_battle_goblin = getattr(scene, "damage_battle_goblin", None)
        if callable(damage_battle_goblin):
            damage_battle_goblin(1)
