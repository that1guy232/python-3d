from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pygame.math import Vector3


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game.world.objects.chest import Chest  # noqa: E402
from game.world.objects.door import Door  # noqa: E402
from game.world.lighting_receivers import (  # noqa: E402
    DYNAMIC_POLYGON_LIGHTING_RECEIVER,
)
from game.world.objects.polygon import (  # noqa: E402
    Polygon,
    PolygonRenderBatch,
    _polygon_vertex_data,
)
from game.world.objects.wall_tile import WallTile, _tile_vertex_data  # noqa: E402
from game.world.objects.window import WindowRenderBatch  # noqa: E402


def lighting(sun_direction=(-1.0, -1.0, -1.0)):
    sun = Vector3(sun_direction)
    return SimpleNamespace(
        sun_direction=sun,
        light_direction=-sun,
        ambient=0.2,
        diffuse=0.8,
        max_factor=1.0,
    )


def set_sun_direction(scene_lighting, value) -> None:
    scene_lighting.sun_direction = Vector3(value)
    scene_lighting.light_direction = -scene_lighting.sun_direction


class DynamicLightingGeometryTests(unittest.TestCase):
    def test_dynamic_slab_vertices_are_sun_independent_material_data(self) -> None:
        scene_lighting = lighting()
        door = Door(
            Vector3(),
            camera=SimpleNamespace(),
            texture=1,
            lighting=scene_lighting,
            side="south",
        )
        vertices = door._visual_vertices()

        legacy = door._slab_vertex_data(
            vertices,
            as_quads=True,
            include_normals=True,
        )
        dynamic = door._slab_vertex_data(
            vertices,
            as_quads=True,
            include_normals=True,
            dynamic_lighting=True,
        )

        self.assertEqual(legacy.shape[1], 11)
        self.assertEqual(dynamic.shape[1], 14)
        self.assertFalse(np.allclose(legacy[:, 3:6], dynamic[:, 3:6]))
        self.assertTrue(np.allclose(dynamic[0:4, 3:6], 1.0))
        self.assertTrue(np.allclose(dynamic[0:4, 6:9], dynamic[0:4, 9:12]))
        self.assertTrue(np.allclose(dynamic[4:8, 3:6], 1.0))
        self.assertTrue(np.allclose(dynamic[:, 6:9], dynamic[:, 9:12]))

        set_sun_direction(scene_lighting, (1.0, -1.0, 1.0))
        changed_legacy = door._slab_vertex_data(
            vertices,
            as_quads=True,
            include_normals=True,
        )
        changed_dynamic = door._slab_vertex_data(
            vertices,
            as_quads=True,
            include_normals=True,
            dynamic_lighting=True,
        )
        self.assertFalse(np.allclose(legacy[:, 3:6], changed_legacy[:, 3:6]))
        self.assertTrue(np.array_equal(dynamic, changed_dynamic))

    def test_dynamic_indoor_wall_uses_material_white_and_real_normals(self) -> None:
        tile = WallTile(
            Vector3(),
            width=1.0,
            height=1.0,
            depth=1.0,
            texture=1,
            thickness=0.2,
        )
        tile.indoor_face_indices = tuple(range(len(tile.faces)))
        tile.indoor_light_factor = 0.34
        tile.indoor_normal_override = (0.0, -1.0, 0.0)
        with patch(
            "game.world.objects.wall_tile.get_texture_size",
            return_value=(16, 16),
        ):
            dynamic = _tile_vertex_data(tile, shader_lighting=True)

        self.assertTrue(np.allclose(dynamic[:, 3:6], 1.0))
        self.assertTrue(np.allclose(dynamic[0:6, 6:9], (1.0, 0.0, 0.0)))

    def test_transparent_window_slab_does_not_occlude_sun_shadow(self) -> None:
        groups = {(7, 11): [np.zeros((4, 11), dtype=np.float32)]}
        sentinel = object()
        with patch(
            "game.world.objects.window.BatchedMesh.from_vertex_data",
            return_value=sentinel,
        ) as create_mesh:
            meshes = WindowRenderBatch._make_meshes(
                groups,
                alpha_test=True,
                casts_shadows=False,
                dynamic_lighting=True,
            )

        self.assertEqual(meshes, [sentinel])
        self.assertFalse(create_mesh.call_args.kwargs["casts_shadows"])

    def test_dynamic_chest_vertices_add_normals_and_remove_sun_factor(self) -> None:
        scene_lighting = lighting()
        chest = Chest(Vector3(), texture=1, lighting=scene_lighting)

        legacy = chest._vertex_data()
        dynamic = chest._vertex_data(dynamic_lighting=True)

        self.assertEqual(legacy.shape[1], 8)
        self.assertEqual(dynamic.shape[1], 11)
        self.assertFalse(np.allclose(legacy[:, 3:6], dynamic[:, 3:6]))
        self.assertTrue(np.all(np.linalg.norm(dynamic[:, 6:9], axis=1) > 0.99))

        set_sun_direction(scene_lighting, (1.0, -1.0, 1.0))
        changed_legacy = chest._vertex_data()
        changed_dynamic = chest._vertex_data(dynamic_lighting=True)
        self.assertFalse(np.allclose(legacy[:, 3:6], changed_legacy[:, 3:6]))
        self.assertTrue(np.array_equal(dynamic, changed_dynamic))

    def test_dynamic_textured_polygon_removes_camera_brightness_and_sun(self) -> None:
        scene_lighting = lighting()
        polygon = Polygon(
            position=Vector3(),
            rotation=Vector3(),
            points_2d=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
            thickness=1.0,
            texture=1,
        )
        polygon.lighting = scene_lighting
        polygon.sun_direction = None
        get_brightness_batch = Mock(return_value=[0.75] * 8)
        camera = SimpleNamespace(
            brightness_default=0.75,
            get_brightness_batch=get_brightness_batch,
        )

        legacy = _polygon_vertex_data(polygon, camera=camera)
        get_brightness_batch.reset_mock()
        dynamic = _polygon_vertex_data(
            polygon,
            camera=camera,
            dynamic_lighting=True,
        )

        self.assertEqual(legacy.shape[1], 8)
        self.assertEqual(dynamic.shape[1], 11)
        self.assertTrue(np.allclose(dynamic[:, 3:6], 1.0))
        self.assertFalse(np.allclose(legacy[:, 3:6], dynamic[:, 3:6]))
        get_brightness_batch.assert_not_called()

        set_sun_direction(scene_lighting, (1.0, -1.0, 1.0))
        changed_dynamic = _polygon_vertex_data(
            polygon,
            camera=camera,
            dynamic_lighting=True,
        )
        self.assertTrue(np.array_equal(dynamic, changed_dynamic))

    def test_dynamic_polygon_cache_ignores_camera_brightness_and_uses_receiver(self) -> None:
        polygon = Polygon(
            position=Vector3(),
            rotation=Vector3(),
            points_2d=[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0)],
            thickness=1.0,
            texture=1,
        )
        batch = PolygonRenderBatch([polygon])
        dim = SimpleNamespace(brightness_default=0.4, _brightness_areas_optimized=())
        bright = SimpleNamespace(brightness_default=0.9, _brightness_areas_optimized=())

        self.assertNotEqual(
            batch._current_cache_key(dim),
            batch._current_cache_key(bright),
        )
        self.assertEqual(
            batch._current_cache_key(dim, dynamic_lighting=True),
            batch._current_cache_key(bright, dynamic_lighting=True),
        )

        with patch(
            "game.world.objects.polygon.BatchedMesh.from_vertex_data",
            return_value=object(),
        ) as create_mesh:
            batch._rebuild(dim, dynamic_lighting=True)

        self.assertIs(
            create_mesh.call_args.kwargs["lighting_receiver"],
            DYNAMIC_POLYGON_LIGHTING_RECEIVER,
        )


if __name__ == "__main__":
    unittest.main()
