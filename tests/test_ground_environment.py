from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game.world.environment import EnvironmentVolume  # noqa: E402
from game.world.objects.ground import TexturedGroundGridBuilder  # noqa: E402


class GroundEnvironmentTests(unittest.TestCase):
    @staticmethod
    def builder(*, environment_volumes=None, covered_regions=None):
        builder = TexturedGroundGridBuilder(
            count=1,
            tile_size=10.0,
            gap=0.0,
            texture=0,
            environment_volumes=environment_volumes,
            covered_regions=covered_regions,
        )
        builder._sample_region_average_base_height = lambda **_kwargs: 12.5
        return builder

    def test_typed_volume_is_authoritative_for_terrain_flattening(self) -> None:
        volume = EnvironmentVolume(
            volume_id="building:0",
            min_x=-5.0,
            max_x=5.0,
            min_z=-7.0,
            max_z=7.0,
            indoor_factor=0.34,
        )
        legacy_render_region = {
            "min_x": 100.0,
            "max_x": 120.0,
            "min_z": 100.0,
            "max_z": 120.0,
            "factor": 0.34,
        }
        builder = self.builder(
            environment_volumes=[volume],
            covered_regions=[legacy_render_region],
        )

        pads = builder._build_terrain_flatten_pads((object(),))

        self.assertEqual(len(pads), 1)
        self.assertEqual(
            (pads[0].min_x, pads[0].max_x, pads[0].min_z, pads[0].max_z),
            (-5.0, 5.0, -7.0, 7.0),
        )
        self.assertEqual(pads[0].height, 12.5)

    def test_explicit_empty_environment_has_no_terrain_pads(self) -> None:
        builder = self.builder(
            environment_volumes=[],
            covered_regions=[
                {
                    "min_x": -5.0,
                    "max_x": 5.0,
                    "min_z": -5.0,
                    "max_z": 5.0,
                }
            ],
        )
        self.assertEqual(builder._build_terrain_flatten_pads((object(),)), [])

    def test_legacy_region_fallback_remains_available(self) -> None:
        builder = self.builder(
            covered_regions=[
                {
                    "min_x": -3.0,
                    "max_x": 3.0,
                    "min_z": -4.0,
                    "max_z": 4.0,
                }
            ]
        )
        pads = builder._build_terrain_flatten_pads((object(),))
        self.assertEqual(len(pads), 1)
        self.assertEqual((pads[0].min_x, pads[0].max_z), (-3.0, 4.0))


if __name__ == "__main__":
    unittest.main()

