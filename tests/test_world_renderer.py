from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game.world.world_renderer import WorldRenderer


class WorldRendererObjectCullingTests(unittest.TestCase):
    def test_object_render_sphere_prefers_explicit_bounds(self) -> None:
        renderer = WorldRenderer(SimpleNamespace(camera=None))
        obj = SimpleNamespace(bounds_center=(1.0, 2.0, 3.0), bounds_radius=7)

        self.assertEqual(renderer._object_render_sphere(obj), ((1.0, 2.0, 3.0), 7.0))

    def test_object_render_sphere_uses_bounding_box_with_camera_height(self) -> None:
        scene = SimpleNamespace(camera=SimpleNamespace(position=SimpleNamespace(y=9.0)))
        renderer = WorldRenderer(scene)
        obj = SimpleNamespace(get_bounding_box=lambda: (0.0, 6.0, 10.0, 14.0))

        center, radius = renderer._object_render_sphere(obj)
        self.assertEqual(center, (3.0, 9.0, 12.0))
        self.assertAlmostEqual(radius, 13.0**0.5)

    def test_object_visible_uses_camera_frustum_when_available(self) -> None:
        calls = []

        def sphere_in_frustum(center, radius, *, far_distance):
            calls.append((center, radius, far_distance))
            return False

        scene = SimpleNamespace(
            camera=SimpleNamespace(sphere_in_frustum=sphere_in_frustum)
        )
        renderer = WorldRenderer(scene)
        obj = SimpleNamespace(bounds_center=(2.0, 3.0, 4.0), bounds_radius=5.0)

        self.assertFalse(renderer._object_visible(obj))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], ((2.0, 3.0, 4.0), 5.0))


if __name__ == "__main__":
    unittest.main()
