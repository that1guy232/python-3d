from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

from pygame.math import Vector3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.world_road_planner import BuildingRoadPlanner


class WorldRoadPlannerTests(unittest.TestCase):
    def test_route_score_prefers_shorter_route_with_fewer_turns(self) -> None:
        direct = [(0.0, 0.0), (0.0, 20.0)]
        dogleg = [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]

        self.assertLess(
            BuildingRoadPlanner._route_score(direct, road_width=8.0),
            BuildingRoadPlanner._route_score(dogleg, road_width=8.0),
        )

    def test_prune_route_removes_duplicate_and_collinear_points(self) -> None:
        route = [
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 5.0),
            (0.0, 10.0),
            (4.0, 10.0),
        ]

        self.assertEqual(
            BuildingRoadPlanner._prune_route(route),
            [(0.0, 0.0), (0.0, 10.0), (4.0, 10.0)],
        )

    def test_find_building_access_route_selects_clear_route_to_network(self) -> None:
        scene = SimpleNamespace(ground_bounds=(0.0, 100.0, 0.0, 100.0))
        building = SimpleNamespace(
            position=Vector3(50.0, 0.0, 50.0),
            bounds=(40.0, 60.0, 40.0, 60.0),
        )

        route = BuildingRoadPlanner(scene)._find_building_access_route(
            building=building,
            spec={"doorway_side": "south"},
            road_center_z=10.0,
            road_width=10.0,
            buildings=[building],
            road_network=[((0.0, 10.0), (100.0, 10.0))],
        )

        self.assertEqual(route[0], (50.0, 39.5))
        self.assertEqual(route[-1], (50.0, 10.0))
        self.assertGreaterEqual(len(route), 2)


if __name__ == "__main__":
    unittest.main()
