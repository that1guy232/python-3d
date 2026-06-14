from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.ui.battle_panel import BattlePanel


class BattlePanelTests(unittest.TestCase):
    def test_goblin_hp_clamps_to_valid_range(self) -> None:
        self.assertEqual(BattlePanel.goblin_hp(SimpleNamespace(max_hp=5, hp=12)), (5, 5))
        self.assertEqual(BattlePanel.goblin_hp(SimpleNamespace(max_hp=0, hp=-3)), (0, 1))

    def test_hp_plate_rect_stays_inside_screen(self) -> None:
        self.assertEqual(
            BattlePanel.hp_plate_rect((1.0, 1.0), 40.0, width=200, height=120),
            (10.0, 10.0, 112.0, 42.0),
        )
        self.assertEqual(
            BattlePanel.hp_plate_rect((500.0, 500.0), 200.0, width=300, height=160),
            (62.0, 108.0, 228.0, 42.0),
        )

    def test_player_stat_lines_format_combat_rows(self) -> None:
        scene = SimpleNamespace(
            player_stats=SimpleNamespace(
                hp=3,
                max_hp=5,
                mana=2,
                max_mana=7,
                strength=4,
                dexterity=6,
                crit_chance=12.4,
                elemental_damage=8,
                card_draw=2,
            )
        )

        self.assertEqual(
            BattlePanel(scene).player_stat_lines(),
            ["STR 4  DEX 6", "Crit 12%  Elem 8", "Draw 2"],
        )


if __name__ == "__main__":
    unittest.main()
