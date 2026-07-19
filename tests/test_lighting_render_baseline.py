from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pygame.math import Vector3


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.camera.camera import Camera  # noqa: E402
from engine.core.mesh import BatchedMesh  # noqa: E402
from engine.rendering.lighting import (  # noqa: E402
    SceneLighting,
    covered_region_factor_at,
)
from engine.rendering.lighting_state import LocalBrightnessLight  # noqa: E402
from game.world.environment import EnvironmentPortal, EnvironmentVolume  # noqa: E402
from game.world.lighting_controller import (  # noqa: E402
    LEGACY_LIGHTING_ALIAS_NAMES,
    PacketStaticGeometryError,
    StaticLightingController,
)
from game.world.lighting_receivers import (  # noqa: E402
    FENCE_LIGHTING_RECEIVER,
    GROUND_LIGHTING_RECEIVER,
    ROAD_LIGHTING_RECEIVER,
    TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
)
from game.world.world_lighting_plan import (  # noqa: E402
    building_environment_volumes,
)


class LightingReceiverPolicyBaselineTests(unittest.TestCase):
    def test_current_batched_mesh_receiver_policy_matrix(self) -> None:
        rows = {
            "ground": BatchedMesh(
                1,
                3,
                texture=1,
                vertex_width=11,
                shader_lighting=True,
                environment_lighting=True,
                shine_enabled=True,
                owns_vbo=False,
            ),
            "textured_wall": BatchedMesh(
                1,
                3,
                texture=1,
                vertex_width=11,
                shader_lighting=True,
                environment_lighting=False,
                shine_enabled=True,
                owns_vbo=False,
            ),
            "door_slab": BatchedMesh(
                1,
                3,
                texture=1,
                vertex_width=11,
                shader_lighting=False,
                environment_lighting=False,
                shine_enabled=True,
                owns_vbo=False,
            ),
            "decal": BatchedMesh(
                1,
                3,
                texture=1,
                vertex_width=8,
                shader_lighting=False,
                environment_lighting=False,
                shine_enabled=False,
                owns_vbo=False,
            ),
        }
        expected = {
            "ground": (False, True, False, True, True, True, True, True),
            "textured_wall": (
                False,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
            ),
            "door_slab": (False, True, False, False, False, False, True, True),
            "decal": (False, False, False, False, False, False, True, False),
        }

        actual = {
            name: BatchedMesh._prepared_draw_key(mesh, object())
            for name, mesh in rows.items()
        }

        self.assertEqual(actual, expected)

    def test_exposure_and_doorway_factor_sample_matrix(self) -> None:
        specs = [
            {
                "position": Vector3(0.0, 0.0, 0.0),
                "width": 20.0,
                "depth": 20.0,
                "doorway_side": "north",
                "doorway_width": 4.0,
            }
        ]
        volume = building_environment_volumes(specs)[0]
        region = volume.to_legacy_dict()
        assert volume.doorway is not None
        samples = {}
        for openness in (0.0, 0.5, 1.0):
            volume.doorway.set_openness(openness)
            region["doorway"]["edge_factor"] = volume.doorway.factor
            factor = covered_region_factor_at(
                0.0,
                volume.max_z,
                covered_regions=[region],
            )
            samples[openness] = tuple(
                round(factor * exposure, 3)
                for exposure in (0.5, 1.0, 1.5)
            )

        self.assertEqual(
            samples,
            {
                0.0: (0.17, 0.34, 0.51),
                0.5: (0.335, 0.67, 1.005),
                1.0: (0.5, 1.0, 1.5),
            },
        )


class LightingUpdateDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def lighting() -> SceneLighting:
        return SceneLighting.from_world_center(
            Vector3(),
            sky_color=(0.7, 0.8, 1.0, 1.0),
        )

    def test_uniform_sync_counts_cache_hits_and_revision_updates(self) -> None:
        lighting = self.lighting()
        scene = SimpleNamespace(
            lighting=lighting,
            camera=Camera(),
            covered_regions=[],
            _texture_lighting_sync_key=None,
            _texture_lighting_sync_result=False,
        )
        build_state = SimpleNamespace(doors=[])
        controller = StaticLightingController(
            scene,
            resources=SimpleNamespace(),
            build_state=build_state,
        )

        with patch(
            "game.world.legacy_lighting_bridge.set_texture_lighting_state",
            return_value=True,
        ) as set_state:
            controller.sync_uniforms(compile_shader=False)
            controller.sync_uniforms(compile_shader=False)
            lighting.set_base_brightness(1.25)
            scene.camera.set_brightness_default(1.25)
            controller.sync_uniforms(compile_shader=False)

        diagnostics = controller.diagnostics
        self.assertEqual(diagnostics.uniform_sync_attempts, 3)
        self.assertEqual(diagnostics.uniform_sync_cache_hits, 1)
        self.assertEqual(diagnostics.shader_state_updates, 2)
        self.assertEqual(diagnostics.shader_uniform_uploads, 2)
        self.assertEqual(diagnostics.legacy_alias_projections, 3)
        self.assertEqual(set_state.call_count, 2)
        self.assertIs(scene.sun_pos, lighting.sun_position)
        self.assertEqual(scene.sun_direction, lighting.sun_direction)
        projected = controller.legacy_brightness_modifiers()
        self.assertEqual(scene.brightness_modifiers, projected)
        self.assertIsNot(scene.brightness_modifiers, projected)
        self.assertFalse(hasattr(lighting, "brightness_modifiers"))
        self.assertIs(scene.covered_regions, controller.legacy_covered_regions)
        self.assertFalse(hasattr(lighting, "covered_regions"))

    def test_static_refresh_counts_rebuilt_resource_families(self) -> None:
        lighting = self.lighting()
        built_ground = SimpleNamespace(height_sampler=None)
        builder = SimpleNamespace(build=lambda: built_ground)
        resources = SimpleNamespace(
            ground_mesh=None,
            ground_height_sampler=None,
            road=None,
            others=[],
            wall_tile_batches=[],
        )
        build_state = SimpleNamespace(
            builder=builder,
            walls=[],
            roads=[],
            doors=[],
        )
        scene = SimpleNamespace(
            _initialized=True,
            _last_static_lighting_brightness=None,
            _texture_lighting_sync_key=None,
            _texture_lighting_sync_result=False,
            lighting=lighting,
            camera=Camera(),
            covered_regions=[],
            environment_volumes=[],
        )
        controller = StaticLightingController(
            scene,
            resources=resources,
            build_state=build_state,
        )

        with (
            patch(
                "game.world.legacy_lighting_bridge.build_wall_tile_batches",
                return_value=[],
            ),
            patch(
                "game.world.legacy_lighting_bridge.world_builder._build_road_batches"
            ),
            patch("game.world.legacy_lighting_bridge.world_builder._build_fences"),
            patch.object(controller, "sync_uniforms", return_value=False),
        ):
            controller.refresh_static()

        diagnostics = controller.diagnostics
        self.assertEqual(diagnostics.static_refreshes, 1)
        self.assertEqual(diagnostics.camera_projection_rebuilds, 1)
        self.assertEqual(diagnostics.ground_rebuilds, 1)
        self.assertEqual(diagnostics.road_refreshes, 0)
        self.assertEqual(diagnostics.wall_batch_rebuilds, 1)
        self.assertEqual(diagnostics.fence_rebuilds, 1)

    def test_packet_sync_prepares_packets_without_touching_legacy_shader(self) -> None:
        lighting = self.lighting()
        legacy_sun_position = object()
        legacy_sun_direction = object()
        legacy_brightness_modifiers = []
        legacy_covered_regions = []
        scene = SimpleNamespace(
            lighting_backend="packet",
            lighting=lighting,
            camera=Camera(),
            sun_pos=legacy_sun_position,
            sun_direction=legacy_sun_direction,
            brightness_modifiers=legacy_brightness_modifiers,
            covered_regions=legacy_covered_regions,
            environment_volumes=[],
            _texture_lighting_sync_key=None,
            _texture_lighting_sync_result=False,
        )
        controller = StaticLightingController(
            scene,
            resources=SimpleNamespace(),
            build_state=SimpleNamespace(doors=[]),
        )

        result = controller.sync_uniforms()

        self.assertTrue(result)
        self.assertFalse(controller.legacy_bridge_instantiated)
        self.assertIn("world.ground", controller.render_packets)
        self.assertIsNone(controller.render_environment_snapshot)
        self.assertIsNone(controller.render_packets["world.ground"].environment)
        self.assertEqual(controller.diagnostics.shader_state_updates, 0)
        self.assertEqual(controller.diagnostics.shader_uniform_uploads, 0)
        self.assertEqual(controller.diagnostics.legacy_alias_projections, 0)
        self.assertIs(scene.sun_pos, legacy_sun_position)
        self.assertIs(scene.sun_direction, legacy_sun_direction)
        self.assertIs(scene.brightness_modifiers, legacy_brightness_modifiers)
        self.assertIs(scene.covered_regions, legacy_covered_regions)

    def test_packet_static_refresh_skips_dynamic_textured_families(self) -> None:
        lighting = self.lighting()
        ground = SimpleNamespace(
            texture=1,
            vertex_width=11,
            lighting_receiver=GROUND_LIGHTING_RECEIVER,
            height_sampler=None,
            dispose=Mock(),
        )
        road = SimpleNamespace(
            texture=1,
            vertex_width=11,
            lighting_receiver=ROAD_LIGHTING_RECEIVER,
            refresh_lighting=Mock(),
        )
        builder = SimpleNamespace(build=Mock())
        resources = SimpleNamespace(
            ground_mesh=ground,
            ground_height_sampler=None,
            road=road,
            roads=[road],
            others=[],
            road_batches=[
                SimpleNamespace(
                    texture=1,
                    vertex_width=11,
                    lighting_receiver=ROAD_LIGHTING_RECEIVER,
                )
            ],
            wall_tile_batches=[
                SimpleNamespace(
                    texture=1,
                    vertex_width=11,
                    lighting_receiver=TEXTURED_STATIC_WALL_LIGHTING_RECEIVER,
                )
            ],
            fence_meshes=[
                SimpleNamespace(
                    texture=1,
                    vertex_width=11,
                    lighting_receiver=FENCE_LIGHTING_RECEIVER,
                )
            ],
        )
        build_state = SimpleNamespace(
            builder=builder,
            walls=[SimpleNamespace()],
            roads=[road],
            doors=[],
        )
        scene = SimpleNamespace(
            lighting_backend="packet",
            _initialized=True,
            _last_static_lighting_brightness=None,
            _texture_lighting_sync_key=None,
            _texture_lighting_sync_result=False,
            lighting=lighting,
            camera=Camera(),
            covered_regions=[],
            environment_volumes=[],
        )
        controller = StaticLightingController(
            scene,
            resources=resources,
            build_state=build_state,
        )

        with patch.object(controller, "sync_uniforms", return_value=True):
            controller.refresh_static()

        builder.build.assert_not_called()
        ground.dispose.assert_not_called()
        road.refresh_lighting.assert_not_called()
        self.assertFalse(controller.legacy_bridge_instantiated)
        diagnostics = controller.diagnostics
        self.assertEqual(diagnostics.static_refreshes, 1)
        self.assertEqual(diagnostics.ground_rebuilds, 0)
        self.assertEqual(diagnostics.road_refreshes, 0)
        self.assertEqual(diagnostics.wall_batch_rebuilds, 0)
        self.assertEqual(diagnostics.fence_rebuilds, 0)

    def test_packet_static_refresh_rejects_rollback_shaped_geometry(self) -> None:
        lighting = self.lighting()
        old_ground = SimpleNamespace(
            texture=None,
            vertex_width=8,
            dispose=Mock(),
        )
        built_ground = SimpleNamespace(
            texture=None,
            vertex_width=8,
            height_sampler=None,
        )
        builder = SimpleNamespace(build=Mock(return_value=built_ground))
        road = SimpleNamespace(
            texture=None,
            vertex_width=8,
            refresh_lighting=Mock(),
        )
        resources = SimpleNamespace(
            ground_mesh=old_ground,
            ground_height_sampler=None,
            road=road,
            others=[],
            road_batches=[SimpleNamespace(texture=None, vertex_width=8)],
            wall_tile_batches=[
                SimpleNamespace(texture=None, vertex_width=6, dispose=Mock())
            ],
            fence_meshes=[SimpleNamespace(texture=None, vertex_width=8)],
        )
        build_state = SimpleNamespace(
            builder=builder,
            walls=[SimpleNamespace()],
            roads=[road],
            doors=[],
        )
        scene = SimpleNamespace(
            lighting_backend="packet",
            _initialized=True,
            _last_static_lighting_brightness=None,
            _texture_lighting_sync_key=None,
            _texture_lighting_sync_result=False,
            lighting=lighting,
            camera=Camera(),
            covered_regions=[],
            environment_volumes=[],
        )
        controller = StaticLightingController(
            scene,
            resources=resources,
            build_state=build_state,
        )

        with patch.object(
            controller,
            "sync_uniforms",
            return_value=True,
        ) as sync:
            with self.assertRaisesRegex(
                PacketStaticGeometryError,
                "ground:rollback-shaped",
            ):
                controller.refresh_static()

        builder.build.assert_not_called()
        old_ground.dispose.assert_not_called()
        road.refresh_lighting.assert_not_called()
        sync.assert_not_called()
        self.assertFalse(controller.legacy_bridge_instantiated)
        diagnostics = controller.diagnostics
        self.assertEqual(diagnostics.ground_rebuilds, 0)
        self.assertEqual(diagnostics.road_refreshes, 0)
        self.assertEqual(diagnostics.wall_batch_rebuilds, 0)
        self.assertEqual(diagnostics.fence_rebuilds, 0)

    def test_packet_brightness_update_never_applies_cpu_mesh_exposure(self) -> None:
        lighting = self.lighting()
        lighting.add_local_light(
            LocalBrightnessLight(
                light_id="test:packet-owner",
                center=(0.0, 0.0, 0.0),
                radius=10.0,
                value=1.4,
            ),
            project_to_camera=False,
        )
        legacy_brightness_modifiers = []
        ground = SimpleNamespace(
            texture=1,
            vertex_width=11,
            lighting_receiver=GROUND_LIGHTING_RECEIVER,
            set_exposure=Mock(),
        )
        resources = SimpleNamespace(
            ground_mesh=ground,
            fence_meshes=[],
            road_batches=[],
            wall_tile_batches=[],
            road=None,
            others=[],
        )
        scene = SimpleNamespace(
            lighting_backend="packet",
            _initialized=True,
            _last_static_lighting_brightness=None,
            _texture_lighting_sync_key=None,
            _texture_lighting_sync_result=False,
            lighting=lighting,
            camera=Camera(),
            covered_regions=[],
            environment_volumes=[],
            brightness_modifiers=legacy_brightness_modifiers,
        )
        controller = StaticLightingController(
            scene,
            resources=resources,
            build_state=SimpleNamespace(roads=[], doors=[]),
        )

        result = controller.set_brightness(1.2)

        self.assertEqual(result, 1.2)
        self.assertEqual(lighting.base_brightness, 1.2)
        self.assertIs(scene.brightness_modifiers, legacy_brightness_modifiers)
        ground.set_exposure.assert_not_called()
        self.assertEqual(controller.diagnostics.ground_rebuilds, 0)

    def test_packet_static_validation_rejects_wrong_receiver_identity(self) -> None:
        ground = SimpleNamespace(
            texture=1,
            vertex_width=11,
            lighting_receiver=ROAD_LIGHTING_RECEIVER,
        )
        controller = StaticLightingController(
            SimpleNamespace(lighting_backend="packet"),
            resources=SimpleNamespace(
                ground_mesh=ground,
                road_batches=[],
                wall_tile_batches=[],
                fence_meshes=[],
                road=None,
                others=[],
            ),
            build_state=SimpleNamespace(roads=[]),
        )

        self.assertEqual(
            controller.packet_static_geometry_violations(),
            (
                "ground:receiver='world.road',expected='world.ground'",
            ),
        )
        with self.assertRaisesRegex(
            PacketStaticGeometryError,
            "expected='world.ground'",
        ):
            controller.validate_packet_static_geometry()

    def test_backend_activation_materializes_and_clears_legacy_projections(self) -> None:
        lighting = self.lighting()
        portal = EnvironmentPortal(
            portal_id="building:0:doorway",
            kind="doorway",
            side="south",
            center_x=0.0,
            center_z=-10.0,
            width=8.0,
            depth=12.0,
            side_fade=3.0,
            closed_factor=0.34,
            open_factor=1.0,
        )
        volume = EnvironmentVolume(
            volume_id="building:0",
            min_x=-10.0,
            max_x=10.0,
            min_z=-10.0,
            max_z=10.0,
            indoor_factor=0.34,
            portals=(portal,),
        )
        modifier = {"light_id": "building:0:doorway:wall-splash"}
        door = SimpleNamespace(
            _environment_portal=portal,
            _doorway_brightness_modifier=modifier,
            bind_doorway_light=Mock(),
        )
        scene = SimpleNamespace(
            lighting_backend="packet",
            lighting=lighting,
            environment_volumes=[volume],
            _texture_lighting_sync_key=None,
        )
        controller = StaticLightingController(
            scene,
            resources=SimpleNamespace(),
            build_state=SimpleNamespace(doors=[door]),
        )

        self.assertEqual(controller.activate_backend("legacy"), "legacy")
        self.assertTrue(all(hasattr(scene, name) for name in LEGACY_LIGHTING_ALIAS_NAMES))
        self.assertEqual(len(scene.covered_regions), 1)
        door.bind_doorway_light.assert_called_once_with(
            scene.covered_regions[0],
            brightness_modifier=modifier,
            portal=portal,
            synchronize=False,
        )
        door.bind_doorway_light.reset_mock()

        self.assertEqual(controller.activate_backend("packet"), "packet")
        self.assertTrue(
            all(not hasattr(scene, name) for name in LEGACY_LIGHTING_ALIAS_NAMES)
        )
        self.assertEqual(controller.legacy_covered_regions, [])
        self.assertFalse(hasattr(lighting, "covered_regions"))
        door.bind_doorway_light.assert_called_once_with(
            None,
            brightness_modifier=modifier,
            portal=portal,
            synchronize=False,
        )


class LightingOpenGLBaselineTests(unittest.TestCase):
    def test_shader_and_fallback_framebuffer_samples(self) -> None:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "capture_lighting_gl_baseline.py"),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output_lines = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip().startswith("{")
        ]
        self.assertTrue(output_lines, completed.stdout)
        captured = json.loads(output_lines[-1])
        if captured.get("status") != "ok":
            self.skipTest(str(captured.get("reason", "OpenGL unavailable")))

        baseline = json.loads(
            (ROOT / "tests" / "baselines" / "lighting_gl_baseline.json").read_text(
                encoding="utf-8"
            )
        )
        for path in ("shader_samples", "fallback_samples"):
            self.assertEqual(set(captured[path]), set(baseline[path]))
            for exposure, expected_states in baseline[path].items():
                self.assertEqual(
                    set(captured[path][exposure]),
                    set(expected_states),
                )
                for state, expected_rgb in expected_states.items():
                    actual_rgb = captured[path][exposure][state]
                    self.assertTrue(
                        all(
                            abs(int(actual) - int(expected)) <= 1
                            for actual, expected in zip(actual_rgb, expected_rgb)
                        ),
                        (
                            f"{path}/{exposure}/{state}: "
                            f"expected {expected_rgb}, got {actual_rgb}"
                        ),
                    )

        captured_receivers = captured["receiver_samples"]
        expected_receivers = baseline["receiver_samples"]
        self.assertEqual(
            captured_receivers["policies"],
            expected_receivers["policies"],
        )
        for path in ("shader_samples", "fallback_samples"):
            self.assertEqual(
                set(captured_receivers[path]),
                set(expected_receivers[path]),
            )
            for receiver, expected_rgb in expected_receivers[path].items():
                actual_rgb = captured_receivers[path][receiver]
                self.assertTrue(
                    all(
                        abs(int(actual) - int(expected)) <= 1
                        for actual, expected in zip(actual_rgb, expected_rgb)
                    ),
                    (
                        f"receiver_samples/{path}/{receiver}: "
                        f"expected {expected_rgb}, got {actual_rgb}"
                    ),
                )

        self.assertNotEqual(
            captured_receivers["shader_samples"]["ground_interior"],
            captured_receivers["fallback_samples"]["ground_interior"],
            "baseline must preserve the current tinted-interior divergence",
        )


if __name__ == "__main__":
    unittest.main()
