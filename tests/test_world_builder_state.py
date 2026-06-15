from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from pygame.math import Vector3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world import building_pipeline, road_pipeline
from game.world.world_content import WorldContent, building


class FakeGroundBuilder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeRoad:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.render_batched = True


class FakeChest:
    def __init__(self, position, **kwargs) -> None:
        self.position = position
        self.kwargs = kwargs

    @staticmethod
    def texture_or_load(texture):
        return f"loaded:{texture}"

    def get_collision_meshes(self):
        return ["chest-mesh"]


class WorldBuilderStateTests(unittest.TestCase):
    def test_prepare_buildings_creates_specs_buildings_and_ground_builder(self) -> None:
        scene = SimpleNamespace(
            building_count=1,
            world_content=WorldContent.with_buildings(
                [
                    building(
                        (50.0, 0.0, 70.0),
                        width=40.0,
                        depth=30.0,
                        doorway_side="south",
                        windows=[{"side": "bad-side", "offset": 999.0}],
                    )
                ]
            ),
            ground_tex="ground",
            brightness_modifiers=["bright"],
            covered_regions=["covered"],
            camera=SimpleNamespace(brightness_default=0.75),
            lighting="lighting",
            sun_direction="sun",
        )

        with (
            patch.object(
                building_pipeline, "TexturedGroundGridBuilder", FakeGroundBuilder
            ),
            patch.object(
                building_pipeline, "apply_building_lighting"
            ) as apply_lighting,
        ):
            building_pipeline._prepare_buildings(scene, 10, 25, 2)

        self.assertEqual(len(scene.building_specs), 1)
        self.assertEqual(len(scene.buildings), 1)
        self.assertEqual(scene.building_specs[0]["windows"][0]["side"], "west")
        self.assertLessEqual(abs(scene.building_specs[0]["windows"][0]["offset"]), 8.0)
        self.assertIsInstance(scene.builder, FakeGroundBuilder)
        self.assertEqual(scene.builder.kwargs["count"], 10)
        self.assertEqual(scene.builder.kwargs["tile_size"], 25)
        self.assertEqual(scene.builder.kwargs["gap"], 2)
        self.assertEqual(scene.builder.kwargs["covered_regions"], ["covered"])
        apply_lighting.assert_called_once_with(scene, scene.building_specs)

    def test_build_roads_records_main_and_access_roads_and_batches(self) -> None:
        access_roads = [SimpleNamespace(name="driveway", render_batched=True)]
        scene = SimpleNamespace(
            ground_bounds=(0.0, 100.0, 0.0, 80.0),
            world_center=Vector3(50.0, 0.0, 40.0),
            camera=SimpleNamespace(
                position=Vector3(0.0, 0.0, 0.0), brightness_default=0.8
            ),
            road_tex="road",
            _ground_height_sampler="height-sampler",
            brightness_modifiers=["bright"],
            lighting="lighting",
            sun_direction="sun",
            others=[],
            roads=[],
            building_roads=[SimpleNamespace(disposed=False)],
            road_batches=[],
            building_road_segments=[("segment", 20.0)],
            ground_height_at=lambda x, z: 3.0,
            log_timing=lambda *args, **kwargs: None,
        )

        with (
            patch.object(road_pipeline, "Road", FakeRoad),
            patch.object(
                road_pipeline,
                "create_building_access_roads",
                return_value=access_roads,
            ),
            patch.object(
                road_pipeline,
                "build_road_render_batch",
                return_value="road-batch",
            ) as build_batch,
            redirect_stdout(io.StringIO()),
        ):
            road_pipeline._build_roads(scene)

        self.assertIsInstance(scene.road, FakeRoad)
        self.assertEqual(scene.road.kwargs["points"], [(0.0, 40.0), (100.0, 40.0)])
        self.assertEqual(scene.road.kwargs["ground_y"], 4.0)
        self.assertEqual(scene.camera.position, scene.world_center)
        self.assertEqual(scene.roads, [scene.road, *access_roads])
        self.assertEqual(scene.others, [scene.road, *access_roads])
        self.assertEqual(scene.road_batches, ["road-batch"])
        build_batch.assert_called_once_with(scene.roads)
        self.assertFalse(scene.road.render_batched)
        self.assertFalse(access_roads[0].render_batched)

    def test_build_showcase_chest_uses_scene_entity_registration(self) -> None:
        registered = []
        scene = SimpleNamespace(
            world_center=Vector3(20.0, 0.0, 30.0),
            ground_height_at=lambda x, z: 5.0,
            lighting=SimpleNamespace(sun_direction="lighting-sun"),
            sun_direction="scene-sun",
            showcase_chests=[],
            chests=[],
            entities=[],
            immediate_entities=[],
            wall_tiles=[],
            add_entity=registered.append,
        )

        with patch.object(building_pipeline, "Chest", FakeChest):
            building_pipeline._build_showcase_chest(scene, "wall-texture")

        self.assertEqual(len(scene.showcase_chests), 1)
        self.assertEqual(scene.chests, scene.showcase_chests)
        self.assertEqual(registered, scene.showcase_chests)
        chest = scene.showcase_chests[0]
        self.assertEqual(chest.position, Vector3(20.0, 5.0, 150.0))
        self.assertEqual(chest.kwargs["texture"], "loaded:wall-texture")
        self.assertEqual(chest.kwargs["sun_direction"], "lighting-sun")
        self.assertEqual(scene.entities, [])
        self.assertEqual(scene.wall_tiles, [])


if __name__ == "__main__":
    unittest.main()
