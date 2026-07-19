"""Player combat stats used by battle-mode systems."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random, random


@dataclass
class PlayerStats:
    max_mana: int = 5
    mana: int = 5
    max_hp: int = 5
    hp: int = 5
    strength: int = 1
    dexterity: int = 1
    crit_chance: float = 0.0
    elemental_damage: int = 0
    card_draw: int = 3

    def __post_init__(self) -> None:
        self.max_mana = max(1, int(self.max_mana))
        self.mana = max(0, min(self.max_mana, int(self.mana)))
        self.max_hp = max(1, int(self.max_hp))
        self.hp = max(0, min(self.max_hp, int(self.hp)))
        self.strength = max(0, int(self.strength))
        self.dexterity = max(0, int(self.dexterity))
        self.crit_chance = max(0.0, float(self.crit_chance))
        self.elemental_damage = max(0, int(self.elemental_damage))
        self.card_draw = max(0, int(self.card_draw))

    def base_attack_damage(self) -> int:
        return max(0, self.strength + self.elemental_damage)

    def roll_attack_damage(self, rng: Random | None = None) -> tuple[int, bool]:
        damage = self.base_attack_damage()
        if damage <= 0 or self.crit_chance <= 0.0:
            return damage, False

        roll = rng.random() if rng is not None else random()
        if roll < min(1.0, self.crit_chance / 100.0):
            return damage * 2, True
        return damage, False

    def crit_percent(self) -> int:
        return int(round(self.crit_chance))
