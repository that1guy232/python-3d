from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from game.settings import DEFAULT_SETTINGS, load_settings, save_settings
from game.world.ui.setting_menu import SettingMenu


class FakeScene:
    def __init__(self) -> None:
        self.ui_state = SimpleNamespace()
        self.camera = SimpleNamespace(
            brightness_default=0.8,
            manual_height_offset=0.0,
            height_adjust_speed=50.0,
        )
        self._camera_controller = SimpleNamespace(rot_smooth_hz=4.0)
        self._headbob = SimpleNamespace(
            enabled=True,
            frequency=0.5,
            amplitude_y=4.0,
            amplitude_x=3.0,
            damping=2.0,
            idle_enabled=True,
            _idle_threshold=1.0,
            _idle_amplitude=0.35,
            _idle_breath_amplitude=1.0,
        )
        self._sway_controller = SimpleNamespace(
            enabled=True,
            mouse_scale=0.01,
            max=SimpleNamespace(x=1.25, y=0.75),
        )

    def set_brightness(self, value: float) -> None:
        self.camera.brightness_default = float(value)


class SettingsTests(unittest.TestCase):
    def test_round_trip_is_atomic_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "settings.json"
            self.assertTrue(
                save_settings(
                    {
                        "hud_visible": False,
                        "fov": 999,
                        "walk_speed": "not a number",
                        "unknown": 123,
                    },
                    target,
                )
            )

            loaded = load_settings(target)
            self.assertFalse(loaded["hud_visible"])
            self.assertEqual(loaded["fov"], 110.0)
            self.assertEqual(loaded["walk_speed"], DEFAULT_SETTINGS["walk_speed"])
            self.assertNotIn("unknown", loaded)
            self.assertFalse(target.with_name("settings.json.tmp").exists())

            document = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(document["version"], 1)

    def test_corrupt_file_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "settings.json"
            target.write_text("{broken", encoding="utf-8")
            self.assertEqual(load_settings(target), DEFAULT_SETTINGS)

    def test_menu_values_survive_a_fresh_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "settings.json"
            expected = dict(DEFAULT_SETTINGS)
            for name, value in tuple(expected.items()):
                if isinstance(value, bool):
                    expected[name] = not value
            expected.update(
                {
                    "fov": 103.0,
                    "brightness": 1.1,
                    "vibrance": 1.7,
                    "mouse_sensitivity": 0.0024,
                    "look_smooth": 9.0,
                    "eye_height": 25.0,
                    "headbob_frequency": 1.2,
                    "idle_amount": 0.8,
                    "sway_scale": 0.025,
                    "sway_limit_x": 2.0,
                    "sway_limit_y": 1.2,
                }
            )

            with patch.dict(os.environ, {"PY3D_SETTINGS_PATH": str(target)}):
                self.assertTrue(save_settings(expected))
                scene = FakeScene()
                menu = SettingMenu(scene)
                menu.apply_saved_settings(scene)
                self.assertEqual(menu.values(scene), expected)


if __name__ == "__main__":
    unittest.main()
