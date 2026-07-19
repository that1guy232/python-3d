from __future__ import annotations

from pathlib import Path
import sys
import unittest
import warnings


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.core.compat_shader import (  # noqa: E402
    LightingCapacityWarning,
    MAX_BRIGHTNESS_AREAS,
    MAX_ENVIRONMENT_OPENINGS,
    MAX_ENVIRONMENT_REGIONS,
    _brightness_area_uniforms,
    _environment_region_uniforms,
)


def brightness_areas(count: int) -> list[dict]:
    return [
        {
            "center": (float(index), 0.0, 0.0),
            "radius": 10.0,
            "value": 1.0,
            "falloff": 1.0,
        }
        for index in range(count)
    ]


def opening(index: int) -> dict:
    return {
        "side": "north",
        "center_x": float(index),
        "center_z": 10.0,
        "width": 4.0,
        "depth": 6.0,
        "side_fade": 2.0,
        "edge_factor": 1.0,
    }


def regions(count: int, *, opening_count: int = 0) -> list[dict]:
    values = [
        {
            "min_x": float(index * 20),
            "max_x": float(index * 20 + 10),
            "min_z": 0.0,
            "max_z": 10.0,
            "factor": 0.34,
            "openings": [],
        }
        for index in range(count)
    ]
    if values:
        values[0]["openings"] = [opening(index) for index in range(opening_count)]
    return values


class ShaderCapacityTests(unittest.TestCase):
    def assert_no_capacity_warning(self, callback) -> object:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = callback()
        capacity_warnings = [
            item
            for item in caught
            if issubclass(item.category, LightingCapacityWarning)
        ]
        self.assertEqual(capacity_warnings, [])
        return result

    def assert_one_capacity_warning(self, callback, text: str) -> object:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = callback()
        capacity_warnings = [
            item
            for item in caught
            if issubclass(item.category, LightingCapacityWarning)
        ]
        self.assertEqual(len(capacity_warnings), 1)
        self.assertIn(text, str(capacity_warnings[0].message))
        return result

    def test_zero_inputs_pack_without_warning(self) -> None:
        area_values = self.assert_no_capacity_warning(
            lambda: _brightness_area_uniforms([])
        )
        region_values = self.assert_no_capacity_warning(
            lambda: _environment_region_uniforms([])
        )
        self.assertEqual(area_values[0], ())
        self.assertEqual(region_values[0], ())

    def test_exact_brightness_area_limit_has_no_warning(self) -> None:
        packed = self.assert_no_capacity_warning(
            lambda: _brightness_area_uniforms(
                brightness_areas(MAX_BRIGHTNESS_AREAS)
            )
        )
        self.assertEqual(len(packed[0]), MAX_BRIGHTNESS_AREAS)

    def test_brightness_area_overflow_warns_and_truncates(self) -> None:
        packed = self.assert_one_capacity_warning(
            lambda: _brightness_area_uniforms(
                brightness_areas(MAX_BRIGHTNESS_AREAS + 1)
            ),
            "brightness areas",
        )
        self.assertEqual(len(packed[0]), MAX_BRIGHTNESS_AREAS)

    def test_exact_environment_region_limit_has_no_warning(self) -> None:
        packed = self.assert_no_capacity_warning(
            lambda: _environment_region_uniforms(
                regions(MAX_ENVIRONMENT_REGIONS)
            )
        )
        self.assertEqual(len(packed[0]), MAX_ENVIRONMENT_REGIONS)

    def test_environment_region_overflow_warns_and_truncates(self) -> None:
        packed = self.assert_one_capacity_warning(
            lambda: _environment_region_uniforms(
                regions(MAX_ENVIRONMENT_REGIONS + 1)
            ),
            "environment regions",
        )
        self.assertEqual(len(packed[0]), MAX_ENVIRONMENT_REGIONS)

    def test_exact_environment_opening_limit_has_no_warning(self) -> None:
        packed = self.assert_no_capacity_warning(
            lambda: _environment_region_uniforms(
                regions(1, opening_count=MAX_ENVIRONMENT_OPENINGS)
            )
        )
        self.assertEqual(len(packed[2]), MAX_ENVIRONMENT_OPENINGS)

    def test_environment_opening_overflow_warns_and_truncates(self) -> None:
        packed = self.assert_one_capacity_warning(
            lambda: _environment_region_uniforms(
                regions(1, opening_count=MAX_ENVIRONMENT_OPENINGS + 1)
            ),
            "environment openings",
        )
        self.assertEqual(len(packed[2]), MAX_ENVIRONMENT_OPENINGS)


if __name__ == "__main__":
    unittest.main()
