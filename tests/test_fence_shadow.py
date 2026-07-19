from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game.world.lighting_receivers import FENCE_LIGHTING_RECEIVER  # noqa: E402
from game.world.objects.fence import build_textured_fence_ring  # noqa: E402


class FenceShadowMaterialTests(unittest.TestCase):
    def test_fence_panels_use_texture_alpha_for_visible_and_shadow_gaps(self) -> None:
        sentinel = object()
        with (
            patch(
                "game.world.objects.fence.get_texture_size",
                return_value=(26, 41),
            ),
            patch(
                "game.world.objects.fence.BatchedMesh.from_vertex_data",
                return_value=sentinel,
            ) as create_mesh,
        ):
            meshes = build_textured_fence_ring(
                0.0,
                26.0,
                0.0,
                26.0,
                textures=[7],
                dynamic_lighting=False,
            )

        self.assertEqual(meshes, [sentinel])
        options = create_mesh.call_args.kwargs
        self.assertTrue(options["alpha_test"])
        self.assertEqual(options["alpha_cutoff"], 0.5)
        self.assertIs(options["lighting_receiver"], FENCE_LIGHTING_RECEIVER)


if __name__ == "__main__":
    unittest.main()
