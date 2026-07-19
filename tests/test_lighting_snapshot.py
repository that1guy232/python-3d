from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pygame.math import Vector3


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.camera.camera import Camera, CameraBrightnessArea  # noqa: E402
from engine.core.compat_shader import (  # noqa: E402
    _brightness_area_uniforms,
    _light_direction_from,
    get_texture_lighting_state,
    set_texture_lighting_state,
)
from engine.rendering.lighting import SceneLighting  # noqa: E402
from engine.rendering.lighting_state import (  # noqa: E402
    LightingSnapshot,
    LocalBrightnessLight,
)
from game.world.lighting_controller import StaticLightingController  # noqa: E402
from game.world.objects.door import Door  # noqa: E402
from game.world.objects.ground import TexturedGroundGridBuilder  # noqa: E402


class LightingSnapshotTests(unittest.TestCase):
    @staticmethod
    def lighting() -> SceneLighting:
        return SceneLighting.from_world_center(
            Vector3(0.0, 0.0, 0.0),
            sky_color=(0.7, 0.8, 1.0, 1.0),
            base_brightness=0.8,
        )

    @staticmethod
    def modifier(*, radius: float = 12.0) -> dict:
        return {
            "light_id": "building:0:doorway:wall-splash",
            "center": Vector3(0.0, 0.0, 0.0),
            "radius": radius,
            "value": 1.36,
            "falloff": 1.65,
            "bounds": (-20.0, 20.0, -20.0, 20.0),
            "indoor_only": True,
            "floor_scale": 0.0,
        }

    @classmethod
    def light(cls, *, radius: float = 12.0) -> LocalBrightnessLight:
        return LocalBrightnessLight.from_normalized(
            cls.modifier(radius=radius),
            fallback_id="test:local-light",
        )

    def test_snapshot_is_immutable_and_revisioned(self) -> None:
        lighting = self.lighting()
        initial = lighting.snapshot()
        self.assertEqual(initial.revision, 0)

        lighting.set_base_brightness(1.0)
        installed = lighting.add_local_light(self.light())
        snapshot = lighting.snapshot()

        self.assertGreater(snapshot.revision, initial.revision)
        self.assertEqual(snapshot.base_brightness, 1.0)
        self.assertEqual(len(snapshot.local_lights), 1)
        self.assertEqual(snapshot.local_lights[0].light_id, installed.light_id)
        with self.assertRaises(FrozenInstanceError):
            snapshot.base_brightness = 0.5

    def test_camera_projection_rebuilds_from_authoritative_revision(self) -> None:
        lighting = self.lighting()
        camera = Camera(default_brightness=0.8)
        installed = lighting.add_local_light(
            self.light(radius=0.0),
            camera=camera,
        )
        original_revision = lighting.revision

        lighting.update_local_light(
            installed,
            camera=camera,
            radius=50.0,
            value=1.5,
        )

        self.assertGreater(lighting.revision, original_revision)
        self.assertEqual(lighting.local_lights[0].radius, 50.0)
        self.assertEqual(camera.brightness_query_lights[0].radius, 50.0)
        self.assertEqual(camera._brightness_source_revision, lighting.revision)

    def test_camera_exposes_only_typed_query_records(self) -> None:
        lighting = self.lighting()
        camera = Camera(default_brightness=0.8)
        lighting.add_local_light(self.light(), camera=camera)

        self.assertTrue(
            all(
                isinstance(value, CameraBrightnessArea)
                for value in camera.brightness_query_lights
            )
        )
        self.assertFalse(hasattr(camera, "brightness_areas"))
        self.assertFalse(hasattr(camera, "_brightness_areas_optimized"))
        self.assertFalse(hasattr(camera, "add_brightness_area"))
        self.assertFalse(hasattr(camera, "set_brightness_areas"))
        self.assertFalse(hasattr(camera, "clear_brightness_areas"))
        self.assertEqual(camera.brightness_query_lights[0].radius, 12.0)
        self.assertEqual(camera.get_brightness_at(Vector3()), 1.36)

    def test_authoritative_owners_reject_legacy_mutation_surfaces(self) -> None:
        lighting = self.lighting()
        camera = Camera(default_brightness=0.8)

        for name in (
            "set_brightness_modifiers",
            "extend_brightness_modifiers",
            "add_brightness_modifier",
            "update_brightness_modifier",
            "remove_brightness_modifiers",
            "install_brightness_modifiers_on_camera",
        ):
            self.assertFalse(hasattr(lighting, name), name)
        for name in (
            "brightness_areas",
            "_brightness_areas_optimized",
            "add_brightness_area",
            "set_brightness_areas",
            "clear_brightness_areas",
        ):
            self.assertFalse(hasattr(camera, name), name)

        with self.assertRaises(TypeError):
            lighting.add_local_light(self.modifier())
        with self.assertRaises(TypeError):
            camera.replace_brightness_query_lights([self.modifier()])

        self.assertIsInstance(lighting.local_lights, tuple)

    def test_typed_snapshot_packs_for_legacy_shader(self) -> None:
        lighting = self.lighting()
        lighting.add_local_light(self.light())
        snapshot = lighting.snapshot()

        packed = _brightness_area_uniforms(snapshot.local_lights)

        self.assertEqual(len(packed[0]), 1)
        self.assertEqual(packed[0][0], (0.0, 0.0, 12.0, 1.36))
        self.assertEqual(packed[3], (1.0,))
        self.assertEqual(packed[4], (0.0,))

    def test_legacy_projection_reads_direction_from_typed_snapshot(self) -> None:
        snapshot = self.lighting().snapshot()

        self.assertEqual(
            _light_direction_from(snapshot),
            snapshot.directional.light_direction,
        )
        self.assertNotEqual(_light_direction_from(snapshot), (0.0, 1.0, 0.0))

    def test_no_argument_shader_initialization_preserves_direction(self) -> None:
        snapshot = self.lighting().snapshot()
        set_texture_lighting_state(lighting=snapshot, compile_shader=False)

        set_texture_lighting_state(compile_shader=False)

        self.assertEqual(
            get_texture_lighting_state().light_direction,
            snapshot.directional.light_direction,
        )

    def test_controller_key_tracks_revision_without_replacing_typed_collection(self) -> None:
        lighting = self.lighting()
        camera = Camera(default_brightness=0.8)
        installed = lighting.add_local_light(
            self.light(),
            camera=camera,
        )
        scene = SimpleNamespace(
            lighting=lighting,
            camera=camera,
            _texture_lighting_sync_key=None,
            build_state=SimpleNamespace(doors=[]),
        )
        controller = StaticLightingController(
            scene,
            resources=SimpleNamespace(),
            build_state=scene.build_state,
        )
        legacy_projection = controller.legacy_brightness_modifiers()
        bridge = controller._legacy_bridge
        self.assertIsNotNone(bridge)
        before = bridge.texture_lighting_fast_key(
            brightness=0.8,
            lighting=lighting.snapshot(),
            sun_direction=lighting.sun_direction,
            brightness_areas=lighting.local_lights,
            covered_regions=[],
            compile_shader=False,
        )

        lighting.update_local_light(installed, camera=camera, radius=30.0)
        camera.add_brightness_query_light(
            LocalBrightnessLight(
                light_id="camera:test-only",
                center=(99.0, 0.0, 99.0),
                radius=5.0,
                value=9.0,
            )
        )
        controller.sync_local_lights_to_camera()
        after = bridge.texture_lighting_fast_key(
            brightness=0.8,
            lighting=lighting.snapshot(),
            sun_direction=lighting.sun_direction,
            brightness_areas=lighting.local_lights,
            covered_regions=[],
            compile_shader=False,
        )

        self.assertNotEqual(before, after)
        self.assertIsInstance(lighting.local_lights, tuple)
        projected_after = controller.legacy_brightness_modifiers()
        self.assertFalse(hasattr(lighting, "brightness_modifiers"))
        self.assertIsNot(projected_after, legacy_projection)
        self.assertEqual(projected_after[0]["radius"], 30.0)
        self.assertEqual(installed.radius, 12.0)
        self.assertEqual(len(camera.brightness_query_lights), 1)
        self.assertEqual(camera.brightness_query_lights[0].radius, 30.0)

    def test_controller_sends_snapshot_to_shader_adapter(self) -> None:
        lighting = self.lighting()
        lighting.add_local_light(self.light())
        camera = Camera(default_brightness=0.8)
        scene = SimpleNamespace(
            lighting=lighting,
            camera=camera,
            _texture_lighting_sync_key=None,
            _texture_lighting_sync_result=False,
            build_state=SimpleNamespace(doors=[]),
        )
        controller = StaticLightingController(
            scene,
            resources=SimpleNamespace(),
            build_state=scene.build_state,
        )

        with patch(
            "game.world.legacy_lighting_bridge.set_texture_lighting_state",
            return_value=True,
        ) as set_state:
            self.assertTrue(controller.sync_uniforms(compile_shader=False))

        shader_input = set_state.call_args.kwargs
        self.assertIsInstance(shader_input["lighting"], LightingSnapshot)
        self.assertEqual(
            shader_input["lighting"].revision,
            lighting.revision,
        )
        self.assertIs(
            shader_input["brightness_areas"][0],
            lighting.local_lights[0],
        )

    def test_door_update_reaches_snapshot_and_camera_projection(self) -> None:
        lighting = self.lighting()
        camera = Camera(default_brightness=0.8)
        installed = lighting.add_local_light(
            self.light(radius=0.0),
            camera=camera,
        )
        doorway_light = {
            **self.modifier(radius=0.0),
            "closed_radius": 0.0,
            "open_radius": 40.0,
            "open_value": 1.36,
        }
        region = {
            "factor": 0.34,
            "doorway": {
                "closed_edge_factor": 0.34,
                "open_edge_factor": 1.0,
                "edge_factor": 0.34,
            },
        }
        door = Door(
            Vector3(),
            camera=camera,
            texture=0,
            lighting=lighting,
        )
        door.bind_doorway_light(region, brightness_modifier=doorway_light)
        door.open_amount = 1.0
        door._sync_doorway_light()

        self.assertEqual(lighting.snapshot().local_lights[0].radius, 40.0)
        self.assertEqual(camera.brightness_query_lights[0].radius, 40.0)

    def test_receivers_keep_legacy_direction_only_without_scene_owner(self) -> None:
        lighting = self.lighting()
        copied_direction = Vector3(9.0, 8.0, 7.0)
        camera = Camera()
        door = Door(
            Vector3(),
            camera=camera,
            texture=0,
            lighting=lighting,
            sun_direction=copied_direction,
        )
        ground = TexturedGroundGridBuilder(
            count=1,
            tile_size=10.0,
            gap=0.0,
            texture=0,
            lighting=lighting,
            sun_direction=copied_direction,
        )
        legacy_door = Door(
            Vector3(),
            camera=camera,
            texture=0,
            sun_direction=copied_direction,
        )

        self.assertIs(door.lighting, lighting)
        self.assertIsNone(door.sun_direction)
        self.assertIsNone(ground.sun_direction)
        self.assertIs(legacy_door.sun_direction, copied_direction)


if __name__ == "__main__":
    unittest.main()
