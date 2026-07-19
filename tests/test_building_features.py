from __future__ import annotations

from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pygame.math import Vector3  # noqa: E402

from engine.rendering.packet_shader_source import (  # noqa: E402
    PACKET_LIGHTING_FRAGMENT_SOURCE,
    PACKET_LIGHTING_VERTEX_SOURCE,
)
from engine.rendering.sprite import WorldSprite, _sprite_data_cache  # noqa: E402
from game.world.building_pipeline import _build_building_torches  # noqa: E402
from game.world.interior_layout import (  # noqa: E402
    create_building_interior_layout,
    exterior_partition_blocked_ranges,
)
from game.world.objects import Door, Window  # noqa: E402
from game.world.objects.torch import Torch  # noqa: E402
from game.world.world_content import create_world_content  # noqa: E402


def _overlaps(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


class TorchPlacementTests(unittest.TestCase):
    def test_default_mount_is_centered_and_explicit_mount_is_preserved(self) -> None:
        spec = {"height": 66.0}

        self.assertEqual(Torch.mount_height_for_building_spec(spec), 33.0)
        self.assertEqual(
            Torch.mount_height_for_building_spec(spec, {"mount_height": 17.0}),
            17.0,
        )

    def test_visible_torches_are_reauthored_after_final_base_height(self) -> None:
        spec = {
            "position": Vector3(10.0, 0.0, 20.0),
            "base_y": 45.0,
            "width": 180.0,
            "depth": 140.0,
            "height": 66.0,
            "doorway_side": "south",
            "torches": [{"side": "north", "offset": 0.0}],
        }
        scene = SimpleNamespace(
            building_specs=[spec],
            lighting_backend="packet",
            torch_point_lights=[],
            torch_tex=None,
            camera=object(),
            build_state=SimpleNamespace(torches=[]),
            render_resources=SimpleNamespace(torch_tex=None, sprite_items=[]),
        )

        def reauthor(target, specs) -> None:
            target.torch_point_lights = Torch.point_lights_for_building_specs(specs)

        with (
            patch(
                "game.world.building_pipeline.apply_building_lighting",
                side_effect=reauthor,
            ) as apply_lighting,
            patch.object(Torch, "texture_or_load", return_value=(1,)),
            patch.object(Torch, "build_for_point_lights", return_value=[]) as build,
        ):
            _build_building_torches(scene)

        apply_lighting.assert_called_once_with(scene, scene.building_specs)
        lights = tuple(build.call_args.args[0])
        self.assertEqual(len(lights), 1)
        self.assertEqual(lights[0].position[1], 78.0)


class TorchEmissiveMaterialTests(unittest.TestCase):
    def test_torch_marks_the_batched_sprite_material_emissive(self) -> None:
        torch = Torch(
            position=Vector3(),
            size=(6.0, 15.0),
            texture=1,
            camera=object(),
            frames=(1,),
        )
        cache = _sprite_data_cache(SimpleNamespace(), [torch], static_data=False)

        self.assertTrue(torch.emissive)
        self.assertEqual(float(cache["emissive"][0]), 1.0)

    def test_ordinary_sprite_remains_non_emissive(self) -> None:
        sprite = WorldSprite(Vector3(), (1.0, 1.0), 1, object())
        cache = _sprite_data_cache(SimpleNamespace(), [sprite], static_data=False)

        self.assertFalse(sprite.emissive)
        self.assertEqual(float(cache["emissive"][0]), 0.0)

    def test_packet_shader_bypasses_direct_light_for_emissive_material(self) -> None:
        self.assertIn("v_emissive = clamp(gl_MultiTexCoord0.z", PACKET_LIGHTING_VERTEX_SOURCE)
        self.assertIn("receiver_rgb * u_exposure", PACKET_LIGHTING_FRAGMENT_SOURCE)
        self.assertIn("v_emissive", PACKET_LIGHTING_FRAGMENT_SOURCE)


class GeneratedExteriorFeaturePlacementTests(unittest.TestCase):
    def test_generated_windows_and_torches_clear_partition_wall_junctions(self) -> None:
        checked_windows = 0
        checked_torches = 0
        checked_junctions = 0
        scene = SimpleNamespace(ground_bounds=(0.0, 2400.0, 0.0, 2400.0))

        for seed in range(30):
            content = create_world_content(
                scene,
                building_count=9,
                rng=random.Random(seed),
            )
            for authored in content.buildings:
                spec = authored.to_runtime_dict()
                spec["wall_thickness"] = 2.5
                layout = create_building_interior_layout(spec)
                blocked = exterior_partition_blocked_ranges(
                    layout,
                    wall_thickness=spec["wall_thickness"],
                )
                checked_junctions += sum(len(ranges) for ranges in blocked.values())

                for window in spec["windows"]:
                    side = str(window["side"])
                    offset = float(window["offset"])
                    width = float(window.get("width", Window.DEFAULT_WIDTH))
                    feature = (offset - width * 0.5, offset + width * 0.5)
                    self.assertFalse(
                        any(_overlaps(feature, junction) for junction in blocked[side]),
                        (seed, side, feature, blocked[side]),
                    )
                    checked_windows += 1

                for torch in spec["torches"]:
                    side = str(torch["side"])
                    offset = float(torch["offset"])
                    feature = (offset - 9.0, offset + 9.0)
                    self.assertFalse(
                        any(_overlaps(feature, junction) for junction in blocked[side]),
                        (seed, side, feature, blocked[side]),
                    )
                    checked_torches += 1

        self.assertGreater(checked_junctions, 100)
        self.assertGreater(checked_windows, 100)
        self.assertGreater(checked_torches, 30)

    def test_exterior_partition_ranges_cover_both_shell_ends(self) -> None:
        layout = {
            "rooms": [
                {
                    "x_min": -50.0,
                    "x_max": 50.0,
                    "z_min": -40.0,
                    "z_max": 40.0,
                }
            ],
            "hallways": [],
            "partitions": [
                {
                    "axis": "x",
                    "coord": 12.0,
                    "span_min": -40.0,
                    "span_max": 40.0,
                },
                {
                    "axis": "z",
                    "coord": -7.0,
                    "span_min": -50.0,
                    "span_max": 50.0,
                },
            ],
        }

        ranges = exterior_partition_blocked_ranges(layout, wall_thickness=4.0)

        self.assertEqual(ranges["north"], [(10.0, 14.0)])
        self.assertEqual(ranges["south"], [(10.0, 14.0)])
        self.assertEqual(ranges["east"], [(-9.0, -5.0)])
        self.assertEqual(ranges["west"], [(-9.0, -5.0)])


if __name__ == "__main__":
    unittest.main()
