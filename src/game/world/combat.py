"""Battle-mode orchestration for world scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from engine.entity import Entity
from game.world.inventory import (
    InventoryItem,
    TEST_GOBLIN_DROP_NAME,
    receive_inventory_item,
)
from game.world.player_stats import PlayerStats

if TYPE_CHECKING:
    from game.world.worldscene import WorldScene


class BattleController:
    """Own battle flow while keeping scene attributes backward compatible."""

    def __init__(self, scene: WorldScene) -> None:
        self.scene = scene
        scene.battle_mode = False
        scene.active_battle_goblin = None
        scene.battle_cards = None
        scene.player_stats = PlayerStats()
        scene.last_player_attack: dict[str, int | bool] | None = None

    def start(self, goblin: Entity) -> bool:
        if goblin is None or not getattr(goblin, "enabled", True):
            return False

        max_hp = max(1, int(getattr(goblin, "max_hp", 5)))
        setattr(goblin, "max_hp", max_hp)
        if not hasattr(goblin, "hp"):
            setattr(goblin, "hp", max_hp)

        scene = self.scene
        scene.battle_mode = True
        scene.active_battle_goblin = goblin
        scene.paused = False
        scene.inventory_open = False
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
        receive_inventory_item(scene, InventoryItem(TEST_GOBLIN_DROP_NAME))
        self.end()
        return True

    def end(self) -> None:
        scene = self.scene
        scene.active_battle_goblin = None
        scene.battle_mode = False

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
