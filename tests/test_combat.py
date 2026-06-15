from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.combat import BattleController
from game.world.player_stats import PlayerStats


class FakeCameraController:
    def __init__(self) -> None:
        self.looked_at = None
        self.synced = False

    def look_at(self, position) -> bool:
        self.looked_at = position
        return True

    def sync_rotation_target_to_camera(self) -> None:
        self.synced = True


class FakeScene:
    def __init__(self) -> None:
        self.paused = True
        self.inventory_open = True
        self.showing_settings_menu = True
        self.mouse_visible = False
        self.mouse_grabbed = True
        self._camera_controller = FakeCameraController()
        self.removed = []
        self.rng = None

    def remove_entity(self, entity) -> None:
        self.removed.append(entity)
        entity.enabled = False


class CombatTests(unittest.TestCase):
    def test_start_battle_initializes_state_and_targets_enemy(self) -> None:
        scene = FakeScene()
        combat = BattleController(scene)
        goblin = SimpleNamespace(enabled=True, max_hp=3, position=object())

        self.assertTrue(combat.start(goblin))

        self.assertTrue(scene.battle_mode)
        self.assertIs(scene.active_battle_goblin, goblin)
        self.assertEqual(goblin.hp, 3)
        self.assertFalse(scene.paused)
        self.assertFalse(scene.inventory_open)
        self.assertFalse(scene.showing_settings_menu)
        self.assertTrue(scene.mouse_visible)
        self.assertFalse(scene.mouse_grabbed)
        self.assertIs(scene._camera_controller.looked_at, goblin.position)

    def test_start_battle_rejects_disabled_enemy(self) -> None:
        scene = FakeScene()
        combat = BattleController(scene)

        self.assertFalse(combat.start(SimpleNamespace(enabled=False)))
        self.assertFalse(scene.battle_mode)
        self.assertIsNone(scene.active_battle_goblin)

    def test_roll_player_attack_damage_records_critical_result(self) -> None:
        scene = FakeScene()
        combat = BattleController(scene)
        scene.player_stats = PlayerStats(
            strength=2, elemental_damage=3, crit_chance=100
        )

        damage, critical = combat.roll_player_attack_damage()

        self.assertEqual((damage, critical), (10, True))
        self.assertEqual(scene.last_player_attack, {"damage": 10, "critical": True})

    def test_damage_active_goblin_removes_enemy_when_hp_reaches_zero(self) -> None:
        scene = FakeScene()
        combat = BattleController(scene)
        goblin = SimpleNamespace(enabled=True, max_hp=2, hp=2)
        combat.start(goblin)

        hp = combat.damage_active_goblin(2)

        self.assertEqual(hp, 0)
        self.assertEqual(scene.removed, [goblin])
        self.assertFalse(scene.battle_mode)
        self.assertIsNone(scene.active_battle_goblin)
        self.assertTrue(scene._camera_controller.synced)
        self.assertFalse(scene.mouse_visible)
        self.assertTrue(scene.mouse_grabbed)


if __name__ == "__main__":
    unittest.main()
