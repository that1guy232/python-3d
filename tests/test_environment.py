from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from pygame.math import Vector3


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.camera.camera import Camera  # noqa: E402
from engine.rendering.lighting import (  # noqa: E402
    SceneLighting,
    covered_region_factor_at,
)
from engine.rendering.lighting_state import (  # noqa: E402
    LocalBrightnessLight,
    PointLight,
)
from game.world.environment import (  # noqa: E402
    EnvironmentPortal,
    EnvironmentVolume,
    environment_factor_at,
)
from game.world.objects.door import Door  # noqa: E402
from game.world.objects.torch import Torch  # noqa: E402
from game.world.world_lighting_plan import (  # noqa: E402
    AuthoredOpeningLight,
    apply_building_lighting,
    building_covered_regions,
    building_environment_volumes,
    opening_wall_lights_for_volumes,
)
from game.world.world_runtime import (  # noqa: E402
    _AMBIENT_BIRDS_INDOOR_VOLUME,
    _AMBIENT_BIRDS_OUTDOOR_VOLUME,
    _ambient_birds_volume,
)


class EnvironmentPortalTests(unittest.TestCase):
    def make_volume(self) -> EnvironmentVolume:
        portal = EnvironmentPortal(
            portal_id="building:0:doorway",
            kind="doorway",
            side="north",
            center_x=0.0,
            center_z=10.0,
            width=4.0,
            depth=6.0,
            side_fade=2.0,
            closed_factor=0.34,
            open_factor=1.0,
        )
        return EnvironmentVolume(
            volume_id="building:0",
            min_x=-10.0,
            max_x=10.0,
            min_z=-10.0,
            max_z=10.0,
            indoor_factor=0.34,
            portals=(portal,),
        )

    def test_portal_openness_updates_typed_factor(self) -> None:
        volume = self.make_volume()
        doorway = volume.doorway
        self.assertIsNotNone(doorway)
        assert doorway is not None

        self.assertAlmostEqual(volume.factor_at(0.0, 10.0), 0.34)
        doorway.set_openness(0.5)
        self.assertAlmostEqual(volume.factor_at(0.0, 10.0), 0.67)
        doorway.set_openness(1.0)
        self.assertAlmostEqual(volume.factor_at(0.0, 10.0), 1.0)
        self.assertAlmostEqual(volume.factor_at(0.0, 0.0), 0.34)

    def test_typed_query_matches_legacy_projection(self) -> None:
        volume = self.make_volume()
        assert volume.doorway is not None
        volume.doorway.set_openness(0.65)
        legacy = volume.to_legacy_dict()

        for x, z in ((0.0, 10.0), (1.0, 7.0), (5.0, 9.0), (0.0, 0.0)):
            typed_factor = environment_factor_at(x, z, volumes=[volume])
            legacy_factor = covered_region_factor_at(
                x,
                z,
                covered_regions=[legacy],
            )
            self.assertAlmostEqual(typed_factor, legacy_factor)

    def test_legacy_projection_reuses_opening_objects(self) -> None:
        legacy = self.make_volume().to_legacy_dict()
        self.assertIs(legacy["doorway"], legacy["openings"][0])

    def test_overlapping_volumes_choose_darkest_factor(self) -> None:
        first = self.make_volume()
        second = EnvironmentVolume(
            volume_id="building:1",
            min_x=-2.0,
            max_x=2.0,
            min_z=-2.0,
            max_z=2.0,
            indoor_factor=0.2,
        )
        self.assertAlmostEqual(
            environment_factor_at(0.0, 0.0, volumes=[first, second]),
            0.2,
        )

    def test_invalid_portal_side_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnvironmentPortal(
                portal_id="invalid",
                kind="doorway",
                side="up",
                center_x=0.0,
                center_z=0.0,
                width=1.0,
                depth=1.0,
                side_fade=1.0,
                closed_factor=0.34,
                open_factor=1.0,
            )

    def test_door_keeps_typed_and_legacy_portals_in_sync(self) -> None:
        volume = self.make_volume()
        legacy = volume.to_legacy_dict()
        door = Door(
            Vector3(0.0, 0.0, 10.0),
            camera=SimpleNamespace(position=Vector3()),
            texture=0,
        )
        door.bind_doorway_light(legacy, portal=volume.doorway)
        door.open_amount = 0.5
        door._sync_doorway_light()

        assert volume.doorway is not None
        self.assertAlmostEqual(volume.doorway.openness, 0.5)
        self.assertAlmostEqual(
            legacy["doorway"]["edge_factor"],
            volume.doorway.factor,
        )

    def test_door_updates_typed_portal_without_legacy_region(self) -> None:
        volume = self.make_volume()
        door = Door(
            Vector3(0.0, 0.0, 10.0),
            camera=SimpleNamespace(position=Vector3()),
            texture=0,
        )

        door.bind_doorway_light(None, portal=volume.doorway)
        door.open_amount = 0.5
        door._sync_doorway_light()

        assert volume.doorway is not None
        self.assertIsNone(door._doorway_light_region)
        self.assertAlmostEqual(volume.doorway.openness, 0.5)
        self.assertAlmostEqual(door._last_doorway_light_factor, volume.doorway.factor)

    def test_typed_authored_doorway_light_updates_without_dictionary_owner(self) -> None:
        volume = self.make_volume()
        doorway_lights, _window_lights = opening_wall_lights_for_volumes([volume])
        authored = doorway_lights[0]
        assert authored is not None
        camera = Camera()
        lighting = SceneLighting.from_world_center(
            Vector3(),
            sky_color=(0.7, 0.8, 1.0, 1.0),
        )
        lighting.add_local_light(authored.light, camera=camera)
        door = Door(
            Vector3(0.0, 0.0, 10.0),
            camera=camera,
            texture=0,
            lighting=lighting,
        )

        door.bind_doorway_light(
            None,
            brightness_modifier=authored,
            portal=volume.doorway,
        )
        door.open_amount = 1.0
        door._sync_doorway_light()

        self.assertIs(door._doorway_brightness_modifier, authored.light)
        self.assertAlmostEqual(
            lighting.snapshot().local_lights[0].radius,
            authored.open_radius,
        )
        self.assertAlmostEqual(
            camera.brightness_query_lights[0].radius,
            authored.open_radius,
        )


class EnvironmentAuthoringTests(unittest.TestCase):
    @staticmethod
    def building_specs() -> list[dict]:
        return [
            {
                "position": Vector3(100.0, 0.0, 200.0),
                "width": 80.0,
                "depth": 60.0,
                "doorway_side": "south",
                "doorway_width": 32.0,
                "windows": [
                    {"side": "north", "offset": 4.0, "width": 18.0}
                ],
            }
        ]

    def test_building_specs_create_stable_typed_and_legacy_views(self) -> None:
        specs = self.building_specs()
        volumes = building_environment_volumes(specs)
        regions = building_covered_regions(specs)

        self.assertEqual(len(volumes), 1)
        self.assertEqual(volumes[0].volume_id, "building:0")
        self.assertEqual(volumes[0].doorway.portal_id, "building:0:doorway")
        self.assertEqual(len(volumes[0].portals), 2)
        self.assertEqual(regions[0]["min_x"], volumes[0].min_x)
        self.assertEqual(len(regions[0]["openings"]), 2)
        self.assertEqual(
            regions[0]["doorway"]["portal_id"],
            "building:0:doorway",
        )

    def test_apply_building_lighting_installs_both_scene_views(self) -> None:
        scene = SimpleNamespace(
            building_specs=self.building_specs(),
            camera=SimpleNamespace(),
            lighting=None,
            brightness_modifiers=[],
        )
        regions = apply_building_lighting(scene)

        self.assertEqual(len(scene.environment_volumes), 1)
        self.assertIs(regions, scene.covered_regions)
        self.assertEqual(
            scene.environment_volumes[0].volume_id,
            "building:0",
        )
        self.assertEqual(len(scene.opening_light_modifiers), 2)
        self.assertEqual(
            scene.opening_light_modifiers[0]["light_id"],
            "building:0:doorway:wall-splash",
        )

    def test_torch_authoring_assigns_stable_light_ids(self) -> None:
        modifiers = Torch.brightness_modifiers_for_building_specs(
            self.building_specs()
        )
        self.assertEqual(modifiers[0]["light_id"], "building:0:torch:0")

        lights = Torch.local_lights_for_building_specs(self.building_specs())
        self.assertIsInstance(lights[0], LocalBrightnessLight)
        self.assertEqual(lights[0].light_id, "building:0:torch:0")
        lighting = SceneLighting.from_world_center(
            Vector3(),
            sky_color=(0.7, 0.8, 1.0, 1.0),
        )
        lighting.add_local_light(lights[0])
        self.assertIs(lighting.local_lights[0], lights[0])

        point_lights = Torch.point_lights_for_building_specs(
            self.building_specs()
        )
        self.assertIsInstance(point_lights[0], PointLight)
        self.assertEqual(point_lights[0].light_id, "building:0:torch:0")
        self.assertTrue(point_lights[0].casts_shadows)
        self.assertGreater(point_lights[0].position[1], 0.0)

    def test_opening_lights_are_authored_from_typed_portals(self) -> None:
        volumes = building_environment_volumes(self.building_specs())
        doorway_lights, window_lights = opening_wall_lights_for_volumes(volumes)

        self.assertIsInstance(doorway_lights[0], AuthoredOpeningLight)
        assert doorway_lights[0] is not None
        self.assertEqual(
            doorway_lights[0].light.light_id,
            "building:0:doorway:wall-splash",
        )
        self.assertEqual(doorway_lights[0].light.radius, 0.0)
        self.assertGreater(doorway_lights[0].open_radius, 0.0)
        self.assertEqual(
            window_lights[0].light.light_id,
            "building:0:window:0:wall-splash",
        )

    def test_scene_install_projects_typed_authoring_to_legacy_dicts(self) -> None:
        lighting = SceneLighting.from_world_center(
            Vector3(),
            sky_color=(0.7, 0.8, 1.0, 1.0),
        )
        scene = SimpleNamespace(
            building_specs=self.building_specs(),
            camera=Camera(),
            lighting=lighting,
            brightness_modifiers=[],
        )
        runtime_light = LocalBrightnessLight(
            light_id="runtime:test",
            center=(0.0, 0.0, 0.0),
            radius=10.0,
            value=1.1,
        )
        lighting.add_local_light(runtime_light, camera=scene.camera)

        apply_building_lighting(scene)

        self.assertTrue(
            all(
                isinstance(light, LocalBrightnessLight)
                for light in lighting.local_lights
            )
        )
        self.assertEqual(
            [light.light_id for light in lighting.local_lights],
            [
                "runtime:test",
                "building:0:doorway:wall-splash",
                "building:0:window:0:wall-splash",
                "building:0:torch:0",
            ],
        )
        self.assertTrue(
            all(isinstance(value, dict) for value in scene.brightness_modifiers)
        )

        first_revision = lighting.revision
        apply_building_lighting(scene)
        self.assertGreater(lighting.revision, first_revision)
        self.assertEqual(len(lighting.local_lights), 4)
        self.assertEqual(len(scene.camera.brightness_query_lights), 4)
        self.assertEqual(
            len({light.light_id for light in lighting.local_lights}),
            4,
        )
        self.assertIs(lighting.local_lights[0], runtime_light)

    def test_packet_scene_install_keeps_authored_lights_typed(self) -> None:
        lighting = SceneLighting.from_world_center(
            Vector3(),
            sky_color=(0.7, 0.8, 1.0, 1.0),
        )
        scene = SimpleNamespace(
            building_specs=self.building_specs(),
            camera=Camera(),
            lighting=lighting,
            lighting_backend="packet",
        )

        regions = apply_building_lighting(scene)

        self.assertEqual(regions, [])
        self.assertFalse(hasattr(scene, "brightness_modifiers"))
        self.assertFalse(hasattr(scene, "covered_regions"))
        self.assertEqual(scene.opening_light_modifiers, [])
        self.assertEqual(scene.torch_light_modifiers, [])
        self.assertEqual(scene.torch_point_lights, list(lighting.point_lights))
        self.assertEqual(lighting.local_lights, ())
        self.assertEqual(len(lighting.point_lights), 1)
        self.assertIsInstance(lighting.point_lights[0], PointLight)
        self.assertEqual(
            lighting.point_lights[0].light_id,
            "building:0:torch:0",
        )
        self.assertNotIn("brightness_modifiers", vars(lighting))
        self.assertFalse(hasattr(lighting, "brightness_modifiers"))

    def test_ambient_audio_uses_typed_portal_state(self) -> None:
        volume = EnvironmentPortalTests().make_volume()
        scene = SimpleNamespace(
            camera=SimpleNamespace(position=Vector3(0.0, 0.0, 10.0)),
            environment_volumes=[volume],
            covered_regions=[],
        )
        self.assertAlmostEqual(
            _ambient_birds_volume(scene),
            _AMBIENT_BIRDS_INDOOR_VOLUME,
        )
        assert volume.doorway is not None
        volume.doorway.set_openness(1.0)
        self.assertAlmostEqual(
            _ambient_birds_volume(scene),
            _AMBIENT_BIRDS_OUTDOOR_VOLUME,
        )


if __name__ == "__main__":
    unittest.main()
