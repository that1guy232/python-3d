from __future__ import annotations

from pathlib import Path
import sys
import unittest

from pygame.math import Vector3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.world_content import (
    BuildingSpec,
    WorldContent,
    as_building_spec,
    building,
)


class WorldContentTests(unittest.TestCase):
    def test_building_helper_normalizes_position_side_and_feature_specs(self) -> None:
        spec = building(
            (10, 20),
            doorway_side="North",
            windows=[{"side": "east"}],
            torches=[{"side": "west"}],
        )

        self.assertEqual(spec.position, Vector3(10, 0, 20))
        self.assertEqual(spec.doorway_side, "north")
        self.assertEqual(spec.windows, ({"side": "east"},))
        self.assertEqual(spec.torches, ({"side": "west"},))

    def test_world_content_converts_dict_declarations_to_building_specs(self) -> None:
        content = WorldContent.with_buildings(
            [
                {
                    "position": (1, 2, 3),
                    "width": 4,
                    "depth": 5,
                    "doorway_side": "WEST",
                }
            ]
        )

        self.assertIsInstance(content.buildings[0], BuildingSpec)
        self.assertEqual(content.buildings[0].position, Vector3(1, 2, 3))
        self.assertEqual(content.buildings[0].width, 4.0)
        self.assertEqual(content.buildings[0].doorway_side, "west")

    def test_to_building_specs_returns_mutable_runtime_copies(self) -> None:
        content = WorldContent.with_buildings(
            [building((1, 2), windows=[{"side": "north"}])]
        )

        runtime_specs = content.to_building_specs()
        runtime_specs[0]["position"].x = 99
        runtime_specs[0]["windows"][0]["side"] = "south"

        self.assertEqual(content.buildings[0].position, Vector3(1, 0, 2))
        self.assertEqual(content.buildings[0].windows, ({"side": "north"},))

    def test_as_building_spec_rejects_unknown_declarations(self) -> None:
        with self.assertRaises(TypeError):
            as_building_spec(object())


if __name__ == "__main__":
    unittest.main()
