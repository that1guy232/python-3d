from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.rendering.lighting import (  # noqa: E402
    INDOOR_LIGHT_FACTOR,
    apply_brightness_modifiers,
    covered_region_factor_at,
    indoor_light_contribution_weight,
    sunlight_factor_for_normal,
    triangle_normals,
)


class DirectionalLightingTests(unittest.TestCase):
    def test_sunlight_factor_uses_inverse_sun_direction(self) -> None:
        factor = sunlight_factor_for_normal(
            (0.0, 1.0, 0.0),
            sun_direction=(0.0, -1.0, 0.0),
            ambient=0.2,
            diffuse=0.6,
            max_factor=1.0,
        )
        self.assertAlmostEqual(factor, 0.8)

    def test_triangle_normals_can_be_forced_upward(self) -> None:
        vertices = np.array(
            [
                (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                (0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
                (1.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            ],
            dtype=np.float32,
        )
        normals = triangle_normals(vertices, prefer_upward=True)
        np.testing.assert_allclose(normals, ((0.0, 1.0, 0.0),) * 3)


class BrightnessAreaTests(unittest.TestCase):
    def test_target_brightness_is_multiplicative_relative_to_base(self) -> None:
        vertices = np.array(
            [
                (0.0, 0.0, 0.0, 0.5, 0.5, 0.5),
                (10.0, 0.0, 0.0, 0.5, 0.5, 0.5),
            ],
            dtype=np.float32,
        )
        apply_brightness_modifiers(
            vertices,
            modifiers=[
                {
                    "center": (0.0, 0.0, 0.0),
                    "radius": 10.0,
                    "value": 1.6,
                    "falloff": 1.0,
                }
            ],
            default_brightness=0.8,
        )
        np.testing.assert_allclose(vertices[0, 3:6], (0.8, 0.8, 0.8))
        np.testing.assert_allclose(vertices[1, 3:6], (0.4, 0.4, 0.4))

    def test_indoor_contribution_has_stable_endpoints(self) -> None:
        self.assertEqual(indoor_light_contribution_weight(INDOOR_LIGHT_FACTOR), 1.0)
        self.assertEqual(indoor_light_contribution_weight(1.0), 0.0)
        middle = indoor_light_contribution_weight(
            (INDOOR_LIGHT_FACTOR + 1.0) * 0.5
        )
        self.assertAlmostEqual(middle, 0.5)


class CoveredRegionTests(unittest.TestCase):
    def test_opening_only_raises_factor_near_its_wall(self) -> None:
        region = {
            "min_x": -10.0,
            "max_x": 10.0,
            "min_z": -10.0,
            "max_z": 10.0,
            "factor": INDOOR_LIGHT_FACTOR,
            "openings": [
                {
                    "side": "north",
                    "center_x": 0.0,
                    "center_z": 0.0,
                    "width": 4.0,
                    "depth": 6.0,
                    "side_fade": 2.0,
                    "edge_factor": 1.0,
                }
            ],
        }
        edge = covered_region_factor_at(0.0, 10.0, covered_regions=[region])
        deep_inside = covered_region_factor_at(0.0, 0.0, covered_regions=[region])
        outside = covered_region_factor_at(20.0, 20.0, covered_regions=[region])
        self.assertAlmostEqual(edge, 1.0)
        self.assertAlmostEqual(deep_inside, INDOOR_LIGHT_FACTOR)
        self.assertAlmostEqual(outside, 1.0)


if __name__ == "__main__":
    unittest.main()

