from __future__ import annotations

from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.player_stats import PlayerStats


class PlayerStatsTests(unittest.TestCase):
    def test_post_init_clamps_resource_and_combat_values(self) -> None:
        stats = PlayerStats(
            max_mana=0,
            mana=9,
            max_hp=-1,
            hp=-4,
            strength=-2,
            dexterity=-3,
            crit_chance=-10.0,
            elemental_damage=-5,
            card_draw=-1,
        )

        self.assertEqual(stats.max_mana, 1)
        self.assertEqual(stats.mana, 1)
        self.assertEqual(stats.max_hp, 1)
        self.assertEqual(stats.hp, 0)
        self.assertEqual(stats.strength, 0)
        self.assertEqual(stats.dexterity, 0)
        self.assertEqual(stats.crit_chance, 0.0)
        self.assertEqual(stats.elemental_damage, 0)
        self.assertEqual(stats.card_draw, 0)

    def test_base_attack_damage_includes_strength_and_elemental_damage(self) -> None:
        self.assertEqual(
            PlayerStats(strength=3, elemental_damage=4).base_attack_damage(),
            7,
        )

    def test_roll_attack_damage_reports_critical_hit(self) -> None:
        damage, critical = PlayerStats(
            strength=2,
            elemental_damage=3,
            crit_chance=100.0,
        ).roll_attack_damage(random.Random(1))

        self.assertEqual(damage, 10)
        self.assertTrue(critical)

    def test_roll_attack_damage_reports_non_critical_hit(self) -> None:
        damage, critical = PlayerStats(
            strength=2,
            elemental_damage=3,
            crit_chance=0.0,
        ).roll_attack_damage(random.Random(1))

        self.assertEqual(damage, 5)
        self.assertFalse(critical)


if __name__ == "__main__":
    unittest.main()
