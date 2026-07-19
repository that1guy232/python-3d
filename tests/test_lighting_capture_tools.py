from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pygame.math import Vector3  # noqa: E402

from scripts.capture_lighting_scene_ab import (  # noqa: E402
    _dynamic_geometry_contract,
    _fence_shadow_gap_probe,
    _fixture_viewpoints,
    _generated_viewpoints,
    _packet_construction_contract,
    _packet_local_light_ownership_contract,
    _packet_receiver_contract,
    _torch_emissive_probe,
    _window_transmission_probe,
)
from scripts.soak_lighting_scene_ab import _result_summary  # noqa: E402
from game.world.lighting_receivers import (  # noqa: E402
    PACKET_RUNTIME_LIGHTING_RECEIVER_IDS,
    ROLLBACK_ONLY_LIGHTING_RECEIVER_IDS,
)


class LightingCaptureViewpointTests(unittest.TestCase):
    @staticmethod
    def _scene(side: str):
        return SimpleNamespace(
            building_specs=[
                {
                    "position": Vector3(400.0, 0.0, 500.0),
                    "base_y": 7.0,
                    "width": 200.0,
                    "depth": 120.0,
                    "height": 66.0,
                    "doorway_side": side,
                }
            ],
            ground_height_at=lambda x, z: x * 0.001 + z * 0.002,
        )

    def test_generated_exterior_view_tracks_each_doorway_side(self) -> None:
        expected = {
            "north": Vector3(0.0, 0.0, 220.0),
            "east": Vector3(260.0, 0.0, 0.0),
            "south": Vector3(0.0, 0.0, -220.0),
            "west": Vector3(-260.0, 0.0, 0.0),
        }
        center = Vector3(400.0, 0.0, 500.0)
        for side, offset in expected.items():
            with self.subTest(side=side):
                viewpoints = _generated_viewpoints(self._scene(side))
                name, position, target = viewpoints[0]
                self.assertEqual(name, "building_exterior")
                self.assertAlmostEqual(position.x, center.x + offset.x)
                self.assertAlmostEqual(position.z, center.z + offset.z)
                self.assertEqual(target, Vector3(400.0, 38.0, 500.0))

    def test_generated_interior_looks_away_from_doorway(self) -> None:
        viewpoints = _generated_viewpoints(self._scene("west"))
        _name, position, target = viewpoints[1]
        self.assertEqual(position, Vector3(400.0, 34.0, 500.0))
        self.assertEqual(target, Vector3(520.0, 34.0, 500.0))

    def test_fixture_viewpoints_remain_fixed(self) -> None:
        scene = SimpleNamespace(
            ground_height_at=lambda _x, _z: 12.0,
            ground_bounds=(12.5, 1287.5, 12.5, 1287.5),
        )
        viewpoints = _fixture_viewpoints(scene)
        self.assertEqual(viewpoints[0][1], Vector3(650.0, 44.0, 640.0))
        self.assertEqual(viewpoints[1][1], Vector3(650.0, 39.0, 380.0))
        self.assertEqual(viewpoints[2][1], Vector3(920.0, 202.0, 720.0))
        self.assertEqual(viewpoints[3][0], "torch_interior")
        self.assertEqual(viewpoints[3][1], Vector3(685.0, 39.0, 400.0))
        self.assertEqual(viewpoints[4][0], "torch_closeup")
        self.assertEqual(viewpoints[4][1], Vector3(685.0, 48.0, 430.0))
        self.assertEqual(viewpoints[5][0], "window_interior")
        self.assertEqual(viewpoints[5][1], Vector3(605.0, 39.0, 400.0))
        self.assertEqual(viewpoints[6][0], "fence_shadow")


class WindowTransmissionProbeTests(unittest.TestCase):
    @staticmethod
    def _frame(*, shadow: int, sunlit: int) -> np.ndarray:
        frame = np.full((270, 480, 3), shadow, dtype=np.uint8)
        frame[205:250, 300:430, :] = sunlit
        return frame

    def test_probe_accepts_distinct_direct_sun_band(self) -> None:
        result = _window_transmission_probe(self._frame(shadow=20, sunlit=30))

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["shadow_mean_luma"], 20.0, places=4)
        self.assertAlmostEqual(result["sunlit_mean_luma"], 30.0, places=4)

    def test_probe_rejects_uniformly_ambient_floor(self) -> None:
        result = _window_transmission_probe(self._frame(shadow=20, sunlit=20))

        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["sunlit_minus_shadow"], 0.0, places=4)


class TorchEmissiveProbeTests(unittest.TestCase):
    def test_probe_accepts_a_bright_warm_flame_tip(self) -> None:
        frame = np.full((270, 480, 3), 20, dtype=np.uint8)
        frame[92:125, 228:245, :] = (220, 150, 25)

        result = _torch_emissive_probe(frame)

        self.assertTrue(result["passed"])
        self.assertGreater(result["flame_tip_p99_luma"], 110.0)

    def test_probe_rejects_an_ambient_dark_flame_tip(self) -> None:
        frame = np.full((270, 480, 3), 20, dtype=np.uint8)
        frame[92:125, 228:245, :] = (70, 55, 15)

        result = _torch_emissive_probe(frame)

        self.assertFalse(result["passed"])
        self.assertEqual(result["warm_flame_pixel_count"], 0)


class FenceShadowGapProbeTests(unittest.TestCase):
    @staticmethod
    def _frame(ground_color) -> np.ndarray:
        frame = np.full((450, 800, 3), (65, 130, 210), dtype=np.uint8)
        frame[180:300, 0:300, :] = ground_color
        return frame

    def test_probe_accepts_sunlit_ground_between_fence_shadows(self) -> None:
        result = _fence_shadow_gap_probe(self._frame((25, 65, 10)))

        self.assertTrue(result["passed"])
        self.assertGreaterEqual(result["ground_p25_luma"], 38.0)

    def test_probe_rejects_an_opaque_fence_wall_shadow(self) -> None:
        result = _fence_shadow_gap_probe(self._frame((10, 30, 5)))

        self.assertFalse(result["passed"])
        self.assertLess(result["ground_p25_luma"], 38.0)


class LightingSoakSummaryTests(unittest.TestCase):
    def test_summary_keeps_only_gate_evidence(self) -> None:
        report = {
            "passed": True,
            "world": {"mode": "generated"},
            "resource_counts": {"local_lights": 4},
            "dynamic_geometry_contract": {"passed": True},
            "packet_construction_contract": {"passed": True},
            "packet_local_light_ownership_contract": {"passed": True},
            "viewpoints": {
                "world_overview": {
                    "legacy_drift": {"passed": True},
                    "packet_parity": {
                        "passed": True,
                        "mean_absolute_error": 0.02,
                        "p99_absolute_error": 1.0,
                        "max_absolute_error": 4,
                        "changed_pixel_ratio": 0.0005,
                    },
                }
            },
        }
        output_dir = ROOT / "artifacts" / "lighting_scene_soak" / "generated_1"
        summary = _result_summary(1, output_dir, report)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["report"], "artifacts/lighting_scene_soak/generated_1/report.json")
        self.assertEqual(
            summary["viewpoints"]["world_overview"]["packet_p99_absolute_error"],
            1.0,
        )
        self.assertEqual(
            summary["viewpoints"]["world_overview"]["packet_max_absolute_error"],
            4,
        )
        self.assertTrue(
            summary["packet_local_light_ownership_contract"]["passed"]
        )


class PacketReceiverCaptureContractTests(unittest.TestCase):
    @staticmethod
    def _scene(receiver_ids):
        return SimpleNamespace(
            lighting_controller=SimpleNamespace(
                render_packets={receiver_id: object() for receiver_id in receiver_ids}
            )
        )

    def test_exact_runtime_receiver_set_passes(self) -> None:
        contract = _packet_receiver_contract(
            self._scene(PACKET_RUNTIME_LIGHTING_RECEIVER_IDS)
        )

        self.assertTrue(contract["passed"])
        self.assertEqual(contract["missing_runtime_receiver_ids"], [])
        self.assertEqual(contract["prepared_rollback_receiver_ids"], [])

    def test_missing_or_rollback_receiver_fails(self) -> None:
        runtime_ids = set(PACKET_RUNTIME_LIGHTING_RECEIVER_IDS)
        runtime_ids.remove(next(iter(runtime_ids)))
        runtime_ids.add(next(iter(ROLLBACK_ONLY_LIGHTING_RECEIVER_IDS)))
        contract = _packet_receiver_contract(self._scene(runtime_ids))

        self.assertFalse(contract["passed"])
        self.assertEqual(len(contract["missing_runtime_receiver_ids"]), 1)
        self.assertEqual(len(contract["prepared_rollback_receiver_ids"]), 1)


class PacketStaticGeometryCaptureContractTests(unittest.TestCase):
    @staticmethod
    def _scene(violations=(), *, ground_casts_sun_shadows=False):
        mesh = SimpleNamespace(
            vertex_width=11,
            texture=1,
            alpha_test=True,
            alpha_cutoff=0.5,
        )
        ground = SimpleNamespace(
            vertex_width=11,
            texture=1,
            casts_sun_shadows=ground_casts_sun_shadows,
        )
        return SimpleNamespace(
            render_resources=SimpleNamespace(
                ground_mesh=ground,
                road_batches=[SimpleNamespace(_meshes=[mesh])],
                wall_tile_batches=[mesh],
                fence_meshes=[mesh],
            ),
            lighting_controller=SimpleNamespace(
                packet_static_geometry_violations=lambda: tuple(violations)
            ),
        )

    def test_contract_requires_no_rollback_shaped_static_geometry(self) -> None:
        passing = _dynamic_geometry_contract(self._scene())
        failing = _dynamic_geometry_contract(
            self._scene(("ground:rollback-shaped",))
        )

        self.assertTrue(passing["passed"])
        self.assertEqual(passing["static_geometry_violations"], [])
        self.assertFalse(failing["passed"])
        self.assertEqual(
            failing["static_geometry_violations"],
            ["ground:rollback-shaped"],
        )

    def test_contract_rejects_self_shadowing_terrain(self) -> None:
        contract = _dynamic_geometry_contract(
            self._scene(ground_casts_sun_shadows=True)
        )

        self.assertFalse(contract["passed"])
        self.assertFalse(contract["ground_sun_caster_disabled"])

    def test_contract_rejects_opaque_fence_shadow_panels(self) -> None:
        scene = self._scene()
        scene.render_resources.fence_meshes[0].alpha_test = False

        contract = _dynamic_geometry_contract(scene)

        self.assertFalse(contract["passed"])
        self.assertFalse(contract["fence_alpha_cutout_shadows"])


class PacketConstructionCaptureContractTests(unittest.TestCase):
    def test_contract_rejects_legacy_module_or_scene_aliases(self) -> None:
        clean = _packet_construction_contract(
            SimpleNamespace(),
            loaded_modules={},
        )
        aliased = _packet_construction_contract(
            SimpleNamespace(covered_regions=[]),
            loaded_modules={},
        )
        module_loaded = _packet_construction_contract(
            SimpleNamespace(),
            loaded_modules={"engine.core.compat_shader": object()},
        )
        bridge_loaded = _packet_construction_contract(
            SimpleNamespace(),
            loaded_modules={"game.world.legacy_lighting_bridge": object()},
        )

        self.assertTrue(clean["passed"])
        self.assertFalse(aliased["passed"])
        self.assertEqual(aliased["legacy_lighting_aliases_present"], ["covered_regions"])
        self.assertFalse(module_loaded["passed"])
        self.assertFalse(bridge_loaded["passed"])
        self.assertTrue(bridge_loaded["legacy_bridge_module_loaded"])


class PacketLocalLightOwnershipCaptureContractTests(unittest.TestCase):
    @staticmethod
    def _scene(lighting, *, legacy_regions=()):
        return SimpleNamespace(
            lighting=lighting,
            lighting_controller=SimpleNamespace(
                legacy_covered_regions=list(legacy_regions)
            ),
            camera=SimpleNamespace(
                brightness_query_lights=[],
                _brightness_source_revision=0,
            ),
            build_state=SimpleNamespace(doors=[]),
            torch_light_modifiers=[],
            doorway_light_modifiers=[],
            window_light_modifiers=[],
            opening_light_modifiers=[],
        )

    def test_contract_rejects_stored_legacy_projection(self) -> None:
        typed_only = _packet_local_light_ownership_contract(
            self._scene(SimpleNamespace(local_lights=(), revision=0))
        )
        duplicated_scene = self._scene(
            SimpleNamespace(
                local_lights=[],
                brightness_modifiers=[],
                covered_regions=[],
                add_brightness_modifier=lambda _value: None,
                revision=0,
            ),
            legacy_regions=[{}],
        )
        duplicated_scene.camera.brightness_areas = []
        duplicated = _packet_local_light_ownership_contract(duplicated_scene)

        self.assertTrue(typed_only["passed"])
        self.assertFalse(duplicated["passed"])
        self.assertTrue(duplicated["legacy_projection_stored_on_lighting"])
        self.assertTrue(duplicated["legacy_projection_api_on_lighting"])
        self.assertFalse(duplicated["local_light_view_immutable"])
        self.assertEqual(
            duplicated["legacy_mutation_apis_on_lighting"],
            ["add_brightness_modifier"],
        )
        self.assertEqual(
            duplicated["legacy_projection_apis_on_camera"],
            ["brightness_areas"],
        )
        self.assertTrue(duplicated["legacy_regions_stored_on_lighting"])
        self.assertEqual(duplicated["controller_legacy_region_count"], 1)


if __name__ == "__main__":
    unittest.main()
