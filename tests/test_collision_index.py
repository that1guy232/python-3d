from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.collision_index import SceneCollisionIndex


class Mesh:
    def __init__(self, name: str, bounds=None, **attrs) -> None:
        self.name = name
        self._bounds = bounds
        for key, value in attrs.items():
            setattr(self, key, value)

    def get_bounding_box(self):
        return self._bounds

    def __repr__(self) -> str:
        return f"Mesh({self.name!r})"


class SceneCollisionIndexTests(unittest.TestCase):
    def make_scene(self, *, wall_tiles=(), polygons=(), cell_size=10.0):
        return SimpleNamespace(
            wall_tiles=list(wall_tiles),
            polygons=list(polygons),
            collision_cell_size=cell_size,
            _collision_spatial_index=None,
        )

    def test_meshes_for_bounds_returns_static_dynamic_and_fallback_candidates(
        self,
    ) -> None:
        static = Mesh("static", bounds=(0, 5, 0, 5))
        far = Mesh("far", bounds=(50, 60, 50, 60))
        dynamic = Mesh("dynamic", bounds=(100, 110, 100, 110), open_amount=0.5)
        fallback = Mesh("fallback")
        scene = self.make_scene(wall_tiles=[static, far, dynamic, fallback])

        candidates = SceneCollisionIndex(scene).meshes_for_bounds(0, 6, 0, 6)

        self.assertEqual(candidates, [static, dynamic, fallback])
        self.assertEqual(
            scene._collision_spatial_index["key"],
            SceneCollisionIndex(scene).source_key(),
        )

    def test_meshes_for_bounds_can_filter_polygons(self) -> None:
        wall = Mesh("wall", bounds=(0, 5, 0, 5))
        polygon = Mesh("polygon", bounds=(0, 5, 0, 5))
        scene = self.make_scene(wall_tiles=[wall], polygons=[polygon])
        index = SceneCollisionIndex(scene)

        self.assertEqual(
            index.meshes_for_bounds(0, 6, 0, 6, include_polygons=True),
            [wall, polygon],
        )
        self.assertEqual(
            index.meshes_for_bounds(0, 6, 0, 6, include_polygons=False),
            [wall],
        )

    def test_index_rebuilds_when_collision_sources_change(self) -> None:
        first = Mesh("first", bounds=(0, 5, 0, 5))
        second = Mesh("second", bounds=(20, 25, 20, 25))
        scene = self.make_scene(wall_tiles=[first])
        index = SceneCollisionIndex(scene)

        self.assertEqual(index.meshes_at(2, 2, 1), [first])
        old_index = scene._collision_spatial_index

        scene.wall_tiles.append(second)

        self.assertEqual(index.meshes_at(22, 22, 1), [second])
        self.assertIsNot(scene._collision_spatial_index, old_index)

    def test_invalidate_clears_cached_index(self) -> None:
        scene = self.make_scene(wall_tiles=[Mesh("wall", bounds=(0, 5, 0, 5))])
        index = SceneCollisionIndex(scene)
        index.rebuild()

        index.invalidate()

        self.assertIsNone(scene._collision_spatial_index)


if __name__ == "__main__":
    unittest.main()
