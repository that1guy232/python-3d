from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import game.world.scene_resources as resources_module
from game.world.scene_resources import SceneResourceDisposer


class Disposable:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def dispose(self) -> None:
        self.calls.append(self.name)


class SceneResourceDisposerTests(unittest.TestCase):
    def test_dispose_releases_each_resource_once_and_clears_scene_lists(self) -> None:
        stopped = []
        original_stop = resources_module.world_runtime.stop_ambient_birds
        resources_module.world_runtime.stop_ambient_birds = lambda: stopped.append(True)
        try:
            calls: list[str] = []
            shared = Disposable("shared", calls)
            unique = Disposable("unique", calls)
            scene = SimpleNamespace(
                ground_mesh=shared,
                road=shared,
                decal_batch=None,
                _hud=unique,
                _ground_height_sampler=object(),
                _collision_spatial_index=object(),
                fence_meshes=[shared],
                wall_tile_batches=[],
                road_batches=[],
                decal_batches=[],
                decals=[],
                roads=[],
                building_roads=[],
                door_batches=[],
                window_batches=[],
                polygon_batches=[],
                showcase_chests=[],
                chests=[],
                others=[],
                entities=[shared, unique],
                immediate_entities=[shared],
            )

            SceneResourceDisposer(scene).dispose()
        finally:
            resources_module.world_runtime.stop_ambient_birds = original_stop

        self.assertEqual(stopped, [True])
        self.assertEqual(calls, ["shared", "unique"])
        self.assertIsNone(scene.ground_mesh)
        self.assertIsNone(scene._ground_height_sampler)
        self.assertIsNone(scene.road)
        self.assertIsNone(scene.decal_batch)
        self.assertIsNone(scene._collision_spatial_index)
        self.assertEqual(scene.entities, [])
        self.assertEqual(scene.immediate_entities, [])
        self.assertEqual(scene.fence_meshes, [])

    def test_dispose_renderable_swallows_dispose_errors(self) -> None:
        bad = SimpleNamespace(dispose=lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        SceneResourceDisposer.dispose_renderable(bad)


if __name__ == "__main__":
    unittest.main()
