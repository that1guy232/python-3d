"""Battle-mode orchestration for world scenes."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from game.world.creature import (
    CombatCreature,
    CombatIntent,
    creature_display_name,
    is_combat_creature,
)
from game.world.inventory import receive_inventory_item
from game.world.player_stats import PlayerStats

if TYPE_CHECKING:
    from game.world.worldscene import WorldScene


class BattleController:
    """Own battle flow while keeping scene attributes backward compatible."""

    DEFAULT_ATTACK_DAMAGE = 1
    GOBLIN_ATTACK_DAMAGE = DEFAULT_ATTACK_DAMAGE
    COMBAT_NOTICE_SECONDS = 1.75

    def __init__(self, scene: WorldScene) -> None:
        self.scene = scene
        scene.battle_mode = False
        scene.active_battle_creature = None
        scene.battle_cards = None
        scene.player_stats = PlayerStats()
        scene.last_player_attack: dict[str, int | bool] | None = None
        self.enemy_intent: CombatIntent | None = None
        self.last_enemy_attack: dict[str, int] | None = None
        self._combat_notice_text = ""
        self._combat_notice_expires_at = 0.0

    def start(self, creature: CombatCreature) -> bool:
        if (
            not is_combat_creature(creature)
            or not getattr(creature, "enabled", True)
        ):
            return False

        max_hp = max(1, int(getattr(creature, "max_hp", 5)))
        setattr(creature, "max_hp", max_hp)
        if not hasattr(creature, "hp"):
            setattr(creature, "hp", max_hp)

        scene = self.scene
        self.last_enemy_attack = None
        self._combat_notice_text = ""
        self._combat_notice_expires_at = 0.0
        scene.battle_mode = True
        scene.active_battle_creature = creature
        self._plan_enemy_turn()
        battle_cards = getattr(scene, "battle_cards", None)
        start_battle = getattr(battle_cards, "start_battle", None)
        if callable(start_battle):
            start_battle()
        scene.paused = False
        scene.inventory_open = False
        scene.inventory_selected_slot = None
        scene.inventory_drag_source = None
        scene.showing_settings_menu = False

        controller = getattr(scene, "_camera_controller", None)
        look_at = getattr(controller, "look_at", None)
        if callable(look_at):
            position = getattr(creature, "position", None)
            if position is not None:
                look_at(position)

        self._set_mouse_for_battle(active=True)
        return True

    def player_attack_damage_preview(self) -> int:
        stats = getattr(self.scene, "player_stats", None)
        if stats is None:
            return 1
        preview = getattr(stats, "base_attack_damage", None)
        if callable(preview):
            return max(0, int(preview()))
        strength = max(0, int(getattr(stats, "strength", 1)))
        elemental = max(0, int(getattr(stats, "elemental_damage", 0)))
        return strength + elemental

    def roll_player_attack_damage(self) -> tuple[int, bool]:
        scene = self.scene
        stats = getattr(scene, "player_stats", None)
        if stats is None:
            scene.last_player_attack = {"damage": 1, "critical": False}
            return 1, False

        roll_damage = getattr(stats, "roll_attack_damage", None)
        if callable(roll_damage):
            rng = getattr(scene, "rng", None)
            try:
                damage, critical = roll_damage(rng)
            except TypeError:
                damage, critical = roll_damage()
        else:
            damage = self.player_attack_damage_preview()
            critical = False

        damage = max(0, int(damage))
        critical = bool(critical)
        scene.last_player_attack = {"damage": damage, "critical": critical}
        return damage, critical

    def damage_active_creature(self, amount: int | None = None) -> int:
        scene = self.scene
        creature = getattr(scene, "active_battle_creature", None)
        if creature is None or not getattr(creature, "enabled", True):
            return 0

        if amount is None:
            amount, _critical = self.roll_player_attack_damage()

        take_damage = getattr(creature, "take_damage", None)
        if callable(take_damage):
            hp = int(take_damage(amount))
        else:
            max_hp = max(1, int(getattr(creature, "max_hp", 5)))
            hp = max(0, int(getattr(creature, "hp", max_hp)) - max(0, int(amount)))
            setattr(creature, "max_hp", max_hp)
            setattr(creature, "hp", hp)

        if hp <= 0:
            self.remove_active_creature()
        return hp

    def end_player_turn(self) -> bool:
        """Discard the hand, resolve enemy intent, then start a new turn."""

        scene = self.scene
        if not getattr(scene, "battle_mode", False):
            return False

        battle_cards = getattr(scene, "battle_cards", None)
        finish_player_turn = getattr(battle_cards, "finish_player_turn", None)
        if callable(finish_player_turn):
            finish_player_turn()

        self.resolve_enemy_intent()
        if not getattr(scene, "battle_mode", False):
            return True

        start_player_turn = getattr(battle_cards, "start_player_turn", None)
        if callable(start_player_turn):
            start_player_turn()
        else:
            stats = getattr(scene, "player_stats", None)
            if stats is not None:
                max_mana = max(1, int(getattr(stats, "max_mana", 5)))
                setattr(stats, "mana", max_mana)
        return True

    def _plan_enemy_turn(self) -> None:
        """Ask the active creature to announce its next combat action."""

        creature = getattr(self.scene, "active_battle_creature", None)
        if creature is None or not getattr(creature, "enabled", True):
            self.enemy_intent = None
            return
        plan_intent = getattr(creature, "plan_combat_intent", None)
        intent = plan_intent(self.scene) if callable(plan_intent) else None
        if not isinstance(intent, dict):
            intent = {
                "action": "attack",
                "damage": max(
                    0,
                    int(getattr(creature, "attack_damage", self.DEFAULT_ATTACK_DAMAGE)),
                ),
            }
        self.enemy_intent = dict(intent)

    def enemy_intent_text(self) -> str | None:
        """Return player-facing text for the active creature's announced action."""

        intent = self.enemy_intent
        if not intent:
            return None
        creature = getattr(self.scene, "active_battle_creature", None)
        format_intent = getattr(creature, "combat_intent_text", None)
        if callable(format_intent):
            text = format_intent(intent)
            if text:
                return str(text)
        name = creature_display_name(creature)
        if intent.get("action") == "attack":
            damage = max(0, int(intent.get("damage", 0)))
            return f"{name} plans to attack for {damage} damage"
        return f"{name} is preparing an action"

    def resolve_enemy_intent(self) -> int:
        """Perform the action announced before the player's turn."""

        intent = self.enemy_intent
        self.enemy_intent = None
        stats = getattr(self.scene, "player_stats", None)
        if not intent:
            return max(0, int(getattr(stats, "hp", 0)))

        if intent.get("action") == "attack":
            hp = self.enemy_attack_player(int(intent.get("damage", 0)))
        else:
            hp = max(0, int(getattr(stats, "hp", 0)))

        if getattr(self.scene, "battle_mode", False):
            self._plan_enemy_turn()
        return hp

    def enemy_attack_player(self, amount: int | None = None) -> int:
        """Apply the active creature's response and return the player's new HP."""

        stats = getattr(self.scene, "player_stats", None)
        if stats is None:
            return 0

        creature = getattr(self.scene, "active_battle_creature", None)
        if amount is None:
            amount = getattr(creature, "attack_damage", self.DEFAULT_ATTACK_DAMAGE)
        damage = max(0, int(amount))
        max_hp = max(1, int(getattr(stats, "max_hp", 5)))
        hp = max(0, min(max_hp, int(getattr(stats, "hp", max_hp))))
        hp = max(0, hp - damage)
        setattr(stats, "max_hp", max_hp)
        setattr(stats, "hp", hp)

        self.last_enemy_attack = {"damage": damage, "player_hp": hp}
        name = creature_display_name(creature)
        self._combat_notice_text = f"{name} attacks for {damage} damage!"
        self._combat_notice_expires_at = (
            time.monotonic() + self.COMBAT_NOTICE_SECONDS
        )
        return hp

    def active_combat_notice(self, *, now: float | None = None) -> str | None:
        """Return the current combat notice while its display window is active."""

        current = time.monotonic() if now is None else float(now)
        if not self._combat_notice_text or current >= self._combat_notice_expires_at:
            self._combat_notice_text = ""
            self._combat_notice_expires_at = 0.0
            return None
        return self._combat_notice_text

    def remove_active_creature(self) -> bool:
        scene = self.scene
        creature = getattr(scene, "active_battle_creature", None)
        if creature is None:
            self.end()
            return False

        max_hp = max(1, int(getattr(creature, "max_hp", 5)))
        hp = int(getattr(creature, "hp", max_hp))
        if hp > 0:
            return False

        scene.remove_entity(creature)
        reward_items = getattr(creature, "combat_rewards", None)
        rewards = reward_items(scene) if callable(reward_items) else ()
        for reward in rewards or ():
            receive_inventory_item(scene, reward)
        self.end()
        return True

    # Compatibility surface for code written against the original
    # Goblin-only controller. All battle state and behavior above is generic.
    @property
    def goblin_intent(self) -> CombatIntent | None:
        return self.enemy_intent

    @goblin_intent.setter
    def goblin_intent(self, value: CombatIntent | None) -> None:
        self.enemy_intent = value

    @property
    def last_goblin_attack(self) -> dict[str, int] | None:
        return self.last_enemy_attack

    @last_goblin_attack.setter
    def last_goblin_attack(self, value: dict[str, int] | None) -> None:
        self.last_enemy_attack = value

    def damage_active_goblin(self, amount: int | None = None) -> int:
        return self.damage_active_creature(amount)

    def goblin_intent_text(self) -> str | None:
        return self.enemy_intent_text()

    def resolve_goblin_intent(self) -> int:
        return self.resolve_enemy_intent()

    def goblin_attack_player(self, amount: int = GOBLIN_ATTACK_DAMAGE) -> int:
        return self.enemy_attack_player(amount)

    def remove_active_goblin(self) -> bool:
        return self.remove_active_creature()

    def end(self) -> None:
        scene = self.scene
        stats = getattr(scene, "player_stats", None)
        if stats is not None:
            max_hp = max(1, int(getattr(stats, "max_hp", 5)))
            setattr(stats, "max_hp", max_hp)
            setattr(stats, "hp", max_hp)
            max_mana = max(1, int(getattr(stats, "max_mana", 5)))
            setattr(stats, "max_mana", max_mana)
            setattr(stats, "mana", max_mana)

        battle_cards = getattr(scene, "battle_cards", None)
        end_battle = getattr(battle_cards, "end_battle", None)
        if callable(end_battle):
            end_battle()

        scene.active_battle_creature = None
        scene.battle_mode = False
        self.enemy_intent = None
        self.last_enemy_attack = None
        self._combat_notice_text = ""
        self._combat_notice_expires_at = 0.0

        controller = getattr(scene, "_camera_controller", None)
        sync_target = getattr(controller, "sync_rotation_target_to_camera", None)
        if callable(sync_target):
            sync_target()

        self._set_mouse_for_battle(active=False)

    def _set_mouse_for_battle(self, *, active: bool) -> None:
        scene = self.scene
        scene.mouse_visible = active
        scene.mouse_grabbed = not active

        import pygame

        try:
            pygame.mouse.set_visible(active)
            pygame.event.set_grab(not active)
        except pygame.error:
            pass
