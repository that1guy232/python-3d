"""Generic world creatures that can participate in turn-based combat."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from pygame.math import Vector3

from engine.entity import Entity


CombatIntent = dict[str, str | int]


@runtime_checkable
class CombatCreature(Protocol):
    """Capabilities consumed by encounter detection, combat, and battle UI."""

    position: Vector3
    enabled: bool
    combat_enabled: bool
    display_name: str
    hp: int
    max_hp: int
    attack_damage: int
    battle_trigger_distance: float | None

    def take_damage(self, amount: int = 1) -> int: ...

    def is_defeated(self) -> bool: ...

    def plan_combat_intent(self, scene: Any) -> CombatIntent: ...

    def combat_intent_text(self, intent: CombatIntent) -> str: ...

    def combat_rewards(self, scene: Any) -> Iterable[Any]: ...


class Creature(Entity):
    """Reusable combat state for Goblins, Skeletons, Blackguards, and others."""

    DEFAULT_MAX_HP = 5
    DEFAULT_ATTACK_DAMAGE = 1

    def __init__(
        self,
        position: Vector3,
        *,
        display_name: str,
        max_hp: int = DEFAULT_MAX_HP,
        attack_damage: int = DEFAULT_ATTACK_DAMAGE,
        battle_trigger_distance: float | None = None,
    ) -> None:
        super().__init__(position=position)
        self.combat_enabled = True
        self.display_name = str(display_name or type(self).__name__)
        self.max_hp = max(1, int(max_hp))
        self.hp = self.max_hp
        self.attack_damage = max(0, int(attack_damage))
        self.battle_trigger_distance = (
            None
            if battle_trigger_distance is None
            else max(0.0, float(battle_trigger_distance))
        )

    def take_damage(self, amount: int = 1) -> int:
        self.hp = max(0, int(self.hp) - max(0, int(amount)))
        return self.hp

    def is_defeated(self) -> bool:
        return int(self.hp) <= 0

    def plan_combat_intent(self, scene: Any) -> CombatIntent:
        return {"action": "attack", "damage": self.attack_damage}

    def combat_intent_text(self, intent: CombatIntent) -> str:
        if intent.get("action") == "attack":
            damage = max(0, int(intent.get("damage", 0)))
            return f"{self.display_name} plans to attack for {damage} damage"
        return f"{self.display_name} is preparing an action"

    def combat_rewards(self, scene: Any) -> Iterable[Any]:
        return ()


def is_combat_creature(value: object) -> bool:
    """Return whether an entity opted into the generic combat contract."""

    return bool(
        value is not None
        and getattr(value, "combat_enabled", False)
        and getattr(value, "position", None) is not None
        and hasattr(value, "hp")
        and hasattr(value, "max_hp")
    )


def creature_display_name(creature: object | None) -> str:
    if creature is None:
        return "Creature"
    value = str(getattr(creature, "display_name", "") or "").strip()
    return value or type(creature).__name__


__all__ = [
    "CombatCreature",
    "CombatIntent",
    "Creature",
    "creature_display_name",
    "is_combat_creature",
]
