from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import game.world.lighting_controller as lighting_module
from game.world.lighting_controller import StaticLightingController


class StaticLightingControllerTests(unittest.TestCase):
    def test_sync_aliases_copies_shared_lighting_fields_to_scene(self) -> None:
        lighting = SimpleNamespace(
            sun_position=(1, 2, 3),
            sun_direction=(0, -1, 0),
            brightness_modifiers=[{"value": 0.8}],
            covered_regions=[{"min_x": 0}],
        )
        scene = SimpleNamespace(lighting=lighting)

        self.assertIs(StaticLightingController(scene).sync_aliases(), lighting)
        self.assertEqual(scene.sun_pos, (1, 2, 3))
        self.assertEqual(scene.sun_direction, (0, -1, 0))
        self.assertIs(scene.brightness_modifiers, lighting.brightness_modifiers)
        self.assertIs(scene.covered_regions, lighting.covered_regions)

    def test_sync_uniforms_uses_cached_result_for_same_scene_state(self) -> None:
        calls = []

        def fake_set_texture_lighting_state(**kwargs):
            calls.append(kwargs)
            return True

        original = lighting_module.set_texture_lighting_state
        lighting_module.set_texture_lighting_state = fake_set_texture_lighting_state
        try:
            lighting = SimpleNamespace(
                ambient=0.72,
                diffuse=0.48,
                max_factor=1.15,
                sun_position=(0, 1, 0),
                sun_direction=(0, -1, 0),
                brightness_modifiers=[],
                covered_regions=[],
                set_base_brightness=lambda value: setattr(lighting, "base", value),
            )
            scene = SimpleNamespace(
                camera=SimpleNamespace(brightness_default=0.9, brightness_areas=[]),
                lighting=lighting,
                covered_regions=[],
                sun_direction=(1, 0, 0),
                doors=[],
                _texture_lighting_sync_key=None,
                _texture_lighting_sync_result=False,
            )
            controller = StaticLightingController(scene)

            self.assertTrue(controller.sync_uniforms(compile_shader=False))
            self.assertTrue(controller.sync_uniforms(compile_shader=False))
        finally:
            lighting_module.set_texture_lighting_state = original

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["base_brightness"], 0.9)
        self.assertFalse(calls[0]["compile_shader"])

    def test_untextured_exposure_skips_textured_meshes_and_deduplicates_roads(
        self,
    ) -> None:
        calls = []

        def exposed(name, *, texture=None):
            return SimpleNamespace(
                texture=texture,
                set_exposure=lambda value: calls.append((name, value)),
            )

        road = exposed("road")
        scene = SimpleNamespace(
            ground_mesh=exposed("ground"),
            fence_meshes=[exposed("fence", texture=object())],
            road_batches=[exposed("road_batch")],
            wall_tile_batches=[exposed("wall")],
            road=road,
            roads=[road],
            others=[road, exposed("other")],
        )

        StaticLightingController(scene).apply_untextured_static_exposure_cpu(0.65)

        self.assertEqual(
            calls,
            [
                ("ground", 0.65),
                ("road_batch", 0.65),
                ("wall", 0.65),
                ("road", 0.65),
                ("other", 0.65),
            ],
        )


if __name__ == "__main__":
    unittest.main()
