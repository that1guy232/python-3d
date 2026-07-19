from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from game.world.objects.ground import TerrainFlattenPad, TexturedGroundGridBuilder


class GroundMeshSeamTests(unittest.TestCase):
    def test_building_pad_tessellation_keeps_interior_edges_watertight(self) -> None:
        builder = TexturedGroundGridBuilder(
            count=30,
            tile_size=25,
            gap=0,
            texture=None,
        )
        pad = TerrainFlattenPad(
            min_x=330,
            max_x=420,
            min_z=330,
            max_z=420,
            height=10,
            blend_margin=96,
        )
        vertex_data, _ = builder._build_ground_vertex_rows(
            terrain_pads=[pad],
            apply_region_colors=False,
        )

        coords = vertex_data[:, [0, 2]]
        base_heights = (
            np.sin(coords[:, 0] / 37.0) * 9.0
            + np.cos(coords[:, 1] / 43.0) * 7.0
        ).astype(np.float32)
        vertex_data[:, 1] = builder._apply_terrain_pads_vectorized(
            coords,
            base_heights,
            [pad],
        )

        edge_counts: Counter[tuple[tuple[float, ...], tuple[float, ...]]] = Counter()
        for start in range(0, len(vertex_data), 3):
            triangle = vertex_data[start : start + 3, :3]
            for first, second in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge = tuple(sorted((tuple(first), tuple(second))))
                edge_counts[edge] += 1

        world_min = -builder.w
        world_max = (builder.count - 1) * builder.spacing + builder.w

        def is_outer_boundary(edge) -> bool:
            first, second = edge
            return (
                (first[0] == world_min and second[0] == world_min)
                or (first[0] == world_max and second[0] == world_max)
                or (first[2] == world_min and second[2] == world_min)
                or (first[2] == world_max and second[2] == world_max)
            )

        unmatched_interior_edges = [
            edge
            for edge, count in edge_counts.items()
            if count == 1 and not is_outer_boundary(edge)
        ]
        self.assertEqual([], unmatched_interior_edges)


if __name__ == "__main__":
    unittest.main()
