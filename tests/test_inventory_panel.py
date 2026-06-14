from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.ui.inventory_panel import InventoryPanel


class InventoryPanelTests(unittest.TestCase):
    def test_item_label_uses_common_item_fields(self) -> None:
        self.assertEqual(InventoryPanel._item_label({"name": "Iron Key"}), "Iron Key")
        self.assertEqual(InventoryPanel._item_label({"label": "Potion"}), "Potion")
        self.assertEqual(InventoryPanel._item_label(SimpleNamespace(name="Map")), "Map")
        self.assertEqual(InventoryPanel._item_label(None), "")

    def test_player_stat_rows_format_scene_stats(self) -> None:
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

        rows = InventoryPanel(scene)._player_stat_rows()

        self.assertEqual(
            rows,
            [
                ("HP", "3/5"),
                ("Mana", "2/7"),
                ("Strength", "4"),
                ("Dexterity", "6"),
                ("Crit Chance", "12%"),
                ("Elemental Damage", "8"),
                ("Card Draw", "2"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
