from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.profile_packet_lighting import (  # noqa: E402
    ProfileScenario,
    _environment,
    _portal_counts,
    default_scenarios,
)


class PacketProfileScenarioTests(unittest.TestCase):
    def test_default_profile_includes_reference_and_scaling_axes(self) -> None:
        scenarios = {scenario.name: scenario for scenario in default_scenarios()}

        self.assertEqual(
            scenarios["generated_reference"],
            ProfileScenario("generated_reference", 17, 3, 13),
        )
        self.assertEqual(scenarios["local_256"].local_lights, 256)
        self.assertEqual(scenarios["environment_16x4"].portals, 64)
        self.assertEqual(scenarios["overlap_8x4"].region_layout, "overlapping")

    def test_portals_are_distributed_without_losing_remainder(self) -> None:
        self.assertEqual(_portal_counts(3, 13), (5, 4, 4))
        self.assertEqual(_portal_counts(0, 13), ())
        self.assertEqual(sum(_portal_counts(8, 32)), 32)

    def test_partitioned_environment_has_one_region_per_x_slice(self) -> None:
        scenario = ProfileScenario("partitioned", 0, 4, 10)
        environment = _environment(scenario)

        self.assertEqual(len(environment.regions), 4)
        self.assertEqual(sum(len(region.portals) for region in environment.regions), 10)
        self.assertEqual(environment.regions[0].min_x, -1.0)
        self.assertEqual(environment.regions[-1].max_x, 1.0)
        for previous, current in zip(
            environment.regions,
            environment.regions[1:],
        ):
            self.assertEqual(previous.max_x, current.min_x)

    def test_record_cost_estimates_expose_nested_overlap(self) -> None:
        partitioned = ProfileScenario("partitioned", 0, 8, 32)
        overlapping = ProfileScenario("overlapping", 0, 8, 32, "overlapping")

        self.assertEqual(partitioned.estimated_record_visits_per_fragment, 40)
        self.assertEqual(overlapping.estimated_record_visits_per_fragment, 264)
        self.assertEqual(overlapping.record_bytes, 1280)


if __name__ == "__main__":
    unittest.main()
