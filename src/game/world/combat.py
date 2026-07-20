"""Battle-mode orchestration for world scenes."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from engine.entity import Entity
from game.world.inventory import (
    GOBLIN_FISTS_ICON,
    GOBLIN_FISTS_NAME,
    GOBLIN_FISTS_STRIKE_CARD_BONUS,
    InventoryItem,
    ItemType,
    receive_inventory_item,
)
from game.world.player_stats import PlayerStats

if TYPE_CHECKING:
    from game.world.worldscene import WorldScene


class BattleController:
    """Own battle flow while keeping scene attributes backward compatible."""

    GOBLIN_ATTACK_DAMAGE = 1
    COMBAT_NOTICE_SECONDS = 1.75

    def __init__(self, scene: WorldScene) -> None:
        self.scene = scene
        scene.battle_mode = False
        scene.active_battle_goblin = None
        scene.battle_cards = None
        scene.player_stats = PlayerStats()
        scene.last_player_attack: dict[str, int | bool] | None = None
        self.goblin_intent: dict[str, str | int] | None = None
        self.last_goblin_attack: dict[str, int] | None = None
        self._combat_notice_text = ""
        self._combat_notice_expires_at = 0.0

    def start(self, goblin: Entity) -> bool:
        if goblin is None or not getattr(goblin, "enabled", True):
            return False

        max_hp = max(1, int(getattr(goblin, "max_hp", 5)))
        setattr(goblin, "max_hp", max_hp)
        if not hasattr(goblin, "hp"):
            setattr(goblin, "hp", max_hp)

        scene = self.scene
        self.last_goblin_attack = None
        self._combat_notice_text = ""
        self._combat_notice_expires_at = 0.0
        scene.battle_mode = True
        scene.active_battle_goblin = goblin
        self._plan_goblin_turn()
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
            position = getattr(goblin, "position", None)
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

    def damage_active_goblin(self, amount: int | None = None) -> int:
        scene = self.scene
        goblin = getattr(scene, "active_battle_goblin", None)
        if goblin is None or not getattr(goblin, "enabled", True):
            return 0

        if amount is None:
            amount, _critical = self.roll_player_attack_damage()

        take_damage = getattr(goblin, "take_damage", None)
        if callable(take_damage):
            hp = int(take_damage(amount))
        else:
            max_hp = max(1, int(getattr(goblin, "max_hp", 5)))
            hp = max(0, int(getattr(goblin, "hp", max_hp)) - max(0, int(amount)))
            setattr(goblin, "max_hp", max_hp)
            setattr(goblin, "hp", hp)

        if hp <= 0:
            self.remove_active_goblin()
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

        self.resolve_goblin_intent()
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

    def _plan_goblin_turn(self) -> None:
        """Announce the active goblin's action before the player acts."""

        goblin = getattr(self.scene, "active_battle_goblin", None)
        if goblin is None or not getattr(goblin, "enabled", True):
            self.goblin_intent = None
            return
        self.goblin_intent = {
            "action": "attack",
            "damage": self.GOBLIN_ATTACK_DAMAGE,
        }

    def goblin_intent_text(self) -> str | None:
        """Return player-facing text for the goblin's announced action."""

        intent = self.goblin_intent
        if not intent:
            return None
        if intent.get("action") == "attack":
            damage = max(0, int(intent.get("damage", 0)))
            return f"Goblin plans to attack for {damage} damage"
        return "Goblin is preparing an action"

    def resolve_goblin_intent(self) -> int:
        """Perform the action announced before the player's turn."""

        intent = self.goblin_intent
        self.goblin_intent = None
        stats = getattr(self.scene, "player_stats", None)
        if not intent:
            return max(0, int(getattr(stats, "hp", 0)))

        if intent.get("action") == "attack":
            hp = self.goblin_attack_player(int(intent.get("damage", 0)))
        else:
            hp = max(0, int(getattr(stats, "hp", 0)))

        if getattr(self.scene, "battle_mode", False):
            self._plan_goblin_turn()
        return hp

    def goblin_attack_player(self, amount: int = GOBLIN_ATTACK_DAMAGE) -> int:
        """Apply the active goblin's response and return the player's new HP."""

        stats = getattr(self.scene, "player_stats", None)
        if stats is None:
            return 0

        damage = max(0, int(amount))
        max_hp = max(1, int(getattr(stats, "max_hp", 5)))
        hp = max(0, min(max_hp, int(getattr(stats, "hp", max_hp))))
        hp = max(0, hp - damage)
        setattr(stats, "max_hp", max_hp)
        setattr(stats, "hp", hp)

        self.last_goblin_attack = {"damage": damage, "player_hp": hp}
        self._combat_notice_text = f"Goblin attacks for {damage} damage!"
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

    def remove_active_goblin(self) -> bool:
        scene = self.scene
        goblin = getattr(scene, "active_battle_goblin", None)
        if goblin is None:
            self.end()
            return False

        max_hp = max(1, int(getattr(goblin, "max_hp", 5)))
        hp = int(getattr(goblin, "hp", max_hp))
        if hp > 0:
            return False

        scene.remove_entity(goblin)
        receive_inventory_item(
            scene,
            InventoryItem(
                GOBLIN_FISTS_NAME,
                ItemType.WEAPON,
                (
                    "The goblin's own fighting style. "
                    "Equip it to add two Strike cards."
                ),
                {"Strike Cards": f"+{GOBLIN_FISTS_STRIKE_CARD_BONUS}"},
                GOBLIN_FISTS_ICON,
            ),
        )
        self.end()
        return True

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

        scene.active_battle_goblin = None
        scene.battle_mode = False
        self.goblin_intent = None
        self.last_goblin_attack = None
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
