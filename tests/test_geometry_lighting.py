from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.rendering.geometry_lighting import (  # noqa: E402
    uses_dynamic_textured_lighting,
)


class GeometryLightingModeTests(unittest.TestCase):
    def test_explicit_backend_choice_does_not_probe_legacy_shader(self) -> None:
        with patch(
            "engine.core.compat_shader.texture_color_exposure_shader_available",
            side_effect=AssertionError("legacy probe should not run"),
        ):
            self.assertTrue(uses_dynamic_textured_lighting(True))
            self.assertFalse(uses_dynamic_textured_lighting(False))

    def test_omitted_choice_preserves_standalone_legacy_probe(self) -> None:
        with patch(
            "engine.core.compat_shader.texture_color_exposure_shader_available",
            return_value=True,
        ) as available:
            self.assertTrue(uses_dynamic_textured_lighting(None))

        available.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
